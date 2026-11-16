"""Retrieval techniques. Each one is a few lines over two score arrays, so they are easy to compare.

Families:
    Lexical     BM25: exact words, IDs and numbers
    Semantic    dense embeddings: meaning and paraphrase
    Hybrid      both, fused (RRF, weighted or query-routed), optionally reranked by a cross-encoder + MMR
    Generative  the LLM helps search: HyDE (search with an imagined answer) and RAG-Fusion (query rewrites)
The default "satyori" pipeline is the query-routed hybrid with rerank and MMR.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from satyori.index import Index
from satyori.ingest import Chunk
from satyori.llm import LLM, parse_json

# ---------------------------------------------------------------- results


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank of this chunk in each stage


@dataclass
class Retrieval:
    technique: str
    query: str
    hits: list[Hit]
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)  # router decision, HyDE passage, query variants...
    top_rerank: float | None = None  # best cross-encoder score, used by the confidence gate


# ---------------------------------------------------------------- the menu


@dataclass(frozen=True)
class Technique:
    key: str
    name: str
    family: str  # Lexical | Semantic | Hybrid | Generative
    blurb: str
    base: str  # dense | sparse | splade | rrf | rrf3 | weighted | adaptive | adaptive3 | hyde | multi_query
    rerank: bool = False
    mmr: bool = False
    needs_llm: bool = False
    needs_splade: bool = False
    blend: float = 1.0  # reranker share of the final score; the rest is the first-stage fusion score


_T = Technique
TECHNIQUES: dict[str, Technique] = {
    t.key: t
    for t in [
        _T("sparse", "BM25", "Lexical", "Exact keyword match. Great for IDs, codes, numbers.", "sparse"),
        _T("dense", "Dense", "Semantic", "Embedding similarity. Understands paraphrase.", "dense"),
        _T(
            "splade",
            "SPLADE",
            "Learned sparse",
            "BERT-weighted keywords plus expansion terms.",
            "splade",
            needs_splade=True,
        ),
        _T("rrf", "Hybrid · RRF", "Hybrid", "Dense + BM25 fused by reciprocal rank.", "rrf"),
        _T("weighted", "Hybrid · Weighted", "Hybrid", "Dense + BM25, normalised scores, alpha = 0.5.", "weighted"),
        _T("adaptive", "Adaptive hybrid", "Hybrid", "A query router picks the dense/BM25 balance.", "adaptive"),
        _T("rerank", "Hybrid + Rerank", "Hybrid", "RRF candidates rescored by a cross-encoder.", "rrf", rerank=True),
        _T("rrf3", "3-way RRF", "Hybrid", "Dense + BM25 + SPLADE fused by reciprocal rank.", "rrf3", needs_splade=True),
        _T(
            "adaptive3",
            "Adaptive 3-way",
            "Hybrid",
            "Dense + BM25 + SPLADE, z-score fusion weighted by the query router.",
            "adaptive3",
            needs_splade=True,
        ),
        _T(
            "satyori",
            "Satyori pipeline",
            "Hybrid",
            "Adaptive 3-way fusion, 50/50 blended rerank, MMR diversity.",
            "adaptive3",
            rerank=True,
            mmr=True,
            blend=0.5,
        ),
        _T(
            "hyde",
            "HyDE hybrid",
            "Generative",
            "Also searches with an answer the LLM imagines.",
            "hyde",
            rerank=True,
            needs_llm=True,
        ),
        _T(
            "multi_query",
            "RAG-Fusion",
            "Generative",
            "LLM rewrites the question 3 ways; results fused.",
            "multi_query",
            rerank=True,
            needs_llm=True,
        ),
    ]
}


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class FastReranker:
    """Cross-encoder (ms-marco MiniLM) via fastembed. Reads query and passage together, unlike embeddings."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, texts)), dtype=float)


# ---------------------------------------------------------------- building blocks


def top(scores: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order if not positive_only or scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = sum of w / (k + rank). Ignores raw scores, so scales never clash."""
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings), strict=True):
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))


def zscore(x: np.ndarray, top_n: int = 100) -> np.ndarray:
    """Distribution-based score normalisation: centre and scale by the query's own top-N scores, so a channel
    that is confident (one score far above the rest) counts for more than one that is unsure."""
    head = np.sort(x)[::-1][:top_n]
    sd = head.std()
    return (x - head.mean()) / sd if sd > 0 else np.zeros_like(x)


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


_ID = re.compile(r"\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z/-])[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+\b|\b\d{3,}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_CONCEPT = re.compile(
    r"\b(can i|should|is it|safe|risk|explain|why|what happens|allowed|worth|compare|difference|meaning|"
    r"implications?|good idea|pros|cons|summar\w*)\b",
    re.I,
)


def route(query: str) -> tuple[float, str]:
    """Query-aware fusion weight. Returns (alpha, reason): alpha is the dense share, 1 - alpha the BM25 share.

    Identifiers and quoted phrases need exact matching, open "can I / is it safe" questions need meaning.
    """
    ids = _ID.findall(query)
    quoted = re.findall(r"\"[^\"]+\"", query)
    acronyms = [a for a in _ACRONYM.findall(query) if a not in {"I", "A"}]
    concept = _CONCEPT.search(query)
    if ids or quoted:
        return 0.3, f"lexical: found identifier {(ids or quoted)[0]!r}, leaning on BM25"
    if concept and not acronyms:
        return 0.75, f"semantic: open question ({concept.group(0)!r}), leaning on embeddings"
    if acronyms:
        return 0.45, f"mixed: acronym {acronyms[0]!r} plus natural language"
    return 0.6, "balanced: no strong signal, slight lean to embeddings"


def mmr(query_vec: np.ndarray, candidates: list[int], vectors: np.ndarray, k: int, lam: float = 0.75) -> list[int]:
    """Maximal Marginal Relevance: pick relevant chunks that are not near-duplicates of ones already picked.
    The best-ranked candidate always stays first, so diversity never costs top-1 precision."""
    if not candidates:
        return []
    chosen: list[int] = [candidates[0]]
    pool = list(candidates[1:])
    while pool and len(chosen) < k:
        best, best_val = pool[0], -np.inf
        for i in pool:
            redundancy = max((float(vectors[i] @ vectors[j]) for j in chosen), default=0.0)
            relevance = 1.0 - candidates.index(i) / max(len(candidates), 1)  # keep the upstream ordering signal
            val = lam * (0.5 * relevance + 0.5 * float(vectors[i] @ query_vec)) - (1 - lam) * redundancy
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
        pool.remove(best)
    return chosen


# ---------------------------------------------------------------- the retriever


class Retriever:
    def __init__(self, index: Index, reranker: Reranker | None = None, llm: LLM | None = None, pool: int = 20):
        self.index, self.reranker, self.llm, self.pool = index, reranker, llm, pool

    def run(self, query: str, technique: str = "satyori", k: int = 5) -> Retrieval:
        t = TECHNIQUES[technique]
        if t.needs_llm and self.llm is None:
            raise RuntimeError(f"{t.name} needs an LLM. Set GROQ_API_KEY to enable it.")
        start = time.perf_counter()
        result = Retrieval(technique, query, [])
        if not len(self.index):
            return result

        ranks: dict[str, list[int]] = {}
        qvec = self.index.embedder.embed_query(query)
        dense = self.index.dense_scores(qvec)
        sparse = self.index.sparse_scores(query)
        ranks["dense"] = top(dense, self.pool)
        ranks["sparse"] = top(sparse, self.pool, positive_only=True)  # zero BM25 score = no shared words
        learned = self.index.splade_scores(query) if t.needs_splade or t.base == "adaptive3" else None
        if learned is not None:
            ranks["splade"] = top(learned, self.pool, positive_only=True)

        if t.base == "dense":
            order, scores = ranks["dense"], {i: float(dense[i]) for i in ranks["dense"]}
        elif t.base == "sparse":
            order, scores = ranks["sparse"], {i: float(sparse[i]) for i in ranks["sparse"]}
        elif t.base == "splade" and learned is not None:
            order, scores = ranks["splade"], {i: float(learned[i]) for i in ranks["splade"]}
        elif t.base == "rrf3" and learned is not None:
            scores = rrf([ranks["dense"], ranks["sparse"], ranks["splade"]])
            order = list(scores)[: self.pool]
        elif t.base == "adaptive3":  # falls back to dense + BM25 when SPLADE is switched off
            alpha, reason = route(query)
            lexical = zscore(sparse) + zscore(learned) if learned is not None else 2 * zscore(sparse)
            channels = "BM25 + SPLADE" if learned is not None else "BM25"
            result.notes.append(f"Router · {reason} (dense {alpha:.0%} / {channels} {1 - alpha:.0%})")
            blended = 2 * alpha * zscore(dense) + (1 - alpha) * lexical
            order = top(blended, self.pool)
            scores = {i: float(blended[i]) for i in order}
        elif t.base in ("weighted", "adaptive"):
            alpha = 0.5
            if t.base == "adaptive":
                alpha, reason = route(query)
                result.notes.append(f"Router · {reason} (dense {alpha:.0%} / BM25 {1 - alpha:.0%})")
            blended = alpha * minmax(dense) + (1 - alpha) * minmax(sparse)
            order = top(blended, self.pool)
            scores = {i: float(blended[i]) for i in order}
        elif t.base == "hyde":
            passage = self.llm.complete(  # type: ignore[union-attr]
                "You write short, factual-sounding passages from Indian property documents (sale agreements, "
                "RERA certificates, title reports). No preamble.",
                f"Write a 3-sentence passage that would answer: {query}",
                max_tokens=200,
                temperature=0.3,
            )
            result.notes.append(f"HyDE passage · {passage}")
            hyde_vec = self.index.embedder.embed_passages([passage])[0]
            ranks["hyde"] = top(self.index.dense_scores(hyde_vec), self.pool)
            scores = rrf([ranks["hyde"], ranks["dense"], ranks["sparse"]])
            order = list(scores)[: self.pool]
        elif t.base == "multi_query":
            variants = self._variants(query)
            result.notes.append("Query variants · " + " | ".join(variants))
            lists = [ranks["dense"], ranks["sparse"]]
            for v in variants:
                lists.append(top(self.index.dense_scores(self.index.embedder.embed_query(v)), self.pool))
                lists.append(top(self.index.sparse_scores(v), self.pool, positive_only=True))
            scores = rrf(lists)
            order = list(scores)[: self.pool]
        else:  # rrf
            scores = rrf([ranks["dense"], ranks["sparse"]])
            order = list(scores)[: self.pool]
        ranks["fused"] = order

        if t.rerank and self.reranker is not None and order:
            ce = self.reranker.score(query, [self.index.chunks[i].indexed_text for i in order])
            result.top_rerank = float(ce.max())  # raw cross-encoder confidence, used by the abstain gate
            final = ce
            if t.blend < 1.0:
                # Blend, don't replace: a reranker trained on web search can be worse than a strong hybrid on
                # unfamiliar text (BEIR SciFact nDCG@10: 0.742 hybrid, 0.687 reranked, 0.745 blended 50/50).
                first = np.array([scores[i] for i in order])
                final = t.blend * _standardize(ce) + (1 - t.blend) * _standardize(first)
            pairs = sorted(zip(order, final, strict=True), key=lambda p: p[1], reverse=True)
            order, scores = [i for i, _ in pairs], {i: float(s) for i, s in pairs}
            ranks["rerank"] = order

        if t.mmr and order:
            order = mmr(qvec, order[: max(k * 3, 10)], self.index.vectors, k)
            ranks["mmr"] = order

        for i in order[:k]:
            stage_ranks = {stage: lst.index(i) + 1 for stage, lst in ranks.items() if i in lst}
            result.hits.append(Hit(self.index.chunks[i], scores.get(i, 0.0), stage_ranks))
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _variants(self, query: str) -> list[str]:
        reply = self.llm.complete(  # type: ignore[union-attr]
            "You rewrite search queries for a property-document search engine. Reply with a JSON array of "
            "3 strings: a paraphrase, a version using legal/technical terms, and a short keyword query.",
            query,
            max_tokens=200,
            temperature=0.4,
        )
        parsed = parse_json(reply)
        return [str(v) for v in parsed][:3] if isinstance(parsed, list) else []
