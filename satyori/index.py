"""The searchable library: dense vectors and a BM25 index over the same chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from satyori.ingest import Chunk, Document, chunk_document

# ---------------------------------------------------------------- sparse: BM25

_STOP = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its me my of on or our so that
    the their there this to was what when where which who why will with would you your any about after before""".split()
)
# Keeps identifiers such as 88/2, cs-2023-8891, b-1204 and 194ia together as one token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.strip(".")
        if not tok or tok in _STOP:
            continue
        tokens.append(_stem(tok))
        if re.search(r"[-/.]", tok):  # also index the parts, so "88/2" matches "survey 88"
            tokens.extend(p for p in re.split(r"[-/.]", tok) if p and p not in _STOP)
    return tokens


class BM25:
    """Okapi BM25. Small enough to read in one sitting, fast enough for thousands of chunks."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(term for d in self.tf for term in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        if not self.tf:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.len / (self.avgdl or 1))
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tf], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


# ---------------------------------------------------------------- dense: embeddings


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


# Instructions these models were trained with. fastembed does not add them, and leaving them out costs accuracy
# (measured: nDCG@10 0.756 -> 0.787 on the golden set for bge-small).
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "snowflake/snowflake-arctic-embed-s": "Represent this sentence for searching relevant passages: ",
    "snowflake/snowflake-arctic-embed-m": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1.5-Q": "search_query: ",
}
PASSAGE_PREFIX = {"nomic-ai/nomic-embed-text-v1.5-Q": "search_document: "}


class FastEmbedder:
    """ONNX sentence embeddings via fastembed: no PyTorch, ~70 MB model, runs on CPU."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.query_prefix = QUERY_PREFIX.get(model_name, "")
        self.passage_prefix = PASSAGE_PREFIX.get(model_name, "")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.passage_embed([self.passage_prefix + t for t in texts]))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed(self.query_prefix + text)))[0])


# ---------------------------------------------------------------- learned sparse: SPLADE

SparseVec = dict[int, float]


class LearnedSparse(Protocol):
    def encode_passages(self, texts: list[str]) -> list[SparseVec]: ...
    def encode_query(self, text: str) -> SparseVec: ...


class SpladeEncoder:
    """SPLADE++: a BERT model that writes a weighted bag of words for each text, adding related terms that never
    appear in it ("demolish" also lights up "demolition", "knock"). Exact-match strength of BM25, some of the
    paraphrase strength of embeddings."""

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    @staticmethod
    def _dict(emb) -> SparseVec:
        return {int(i): float(v) for i, v in zip(emb.indices, emb.values, strict=True)}

    def encode_passages(self, texts: list[str]) -> list[SparseVec]:
        return [self._dict(e) for e in self.model.embed(texts, batch_size=32)]

    def encode_query(self, text: str) -> SparseVec:
        return self._dict(next(iter(self.model.query_embed(text))))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1, norm)


# ---------------------------------------------------------------- the library


class Index:
    def __init__(self, embedder: Embedder, splade: LearnedSparse | None = None):
        self.embedder, self.splade = embedder, splade
        self.splade_vecs: list[SparseVec] = []
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0))
        self.bm25 = BM25([])
        self.fingerprints: dict[str, str] = {}  # fingerprint -> document name

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def documents(self) -> list[str]:
        return list(dict.fromkeys(c.doc for c in self.chunks))

    def clone(self) -> Index:
        """A copy that shares the embedder but not the data, so each user session can add or remove documents."""
        other = Index(self.embedder, self.splade)
        other.chunks, other.vectors = list(self.chunks), self.vectors.copy()
        other.splade_vecs = list(self.splade_vecs)
        other.fingerprints, other.bm25 = dict(self.fingerprints), self.bm25
        return other

    def add(self, doc: Document, chunk_size: int = 700, overlap: int = 1) -> int:
        """Index a document. Returns the number of new chunks (0 if this exact file is already indexed)."""
        if doc.fingerprint in self.fingerprints:
            return 0
        chunks = chunk_document(doc, chunk_size, overlap)
        if not chunks:
            return 0
        texts = [c.indexed_text for c in chunks]
        vectors = self.embedder.embed_passages(texts)
        if self.splade is not None:
            self.splade_vecs.extend(self.splade.encode_passages(texts))
        self.vectors = vectors if not len(self.chunks) else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.fingerprints[doc.fingerprint] = doc.name
        self._rebuild_bm25()
        return len(chunks)

    def remove(self, doc_name: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc != doc_name]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 0))
        if self.splade_vecs:
            self.splade_vecs = [self.splade_vecs[i] for i in keep]
        self.fingerprints = {f: n for f, n in self.fingerprints.items() if n != doc_name}
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25([tokenize(c.indexed_text) for c in self.chunks])

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.vectors @ query_vec if len(self.chunks) else np.zeros(0)

    def sparse_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(tokenize(query))

    def splade_scores(self, query: str) -> np.ndarray | None:
        """Dot product of SPLADE term weights, or None when no SPLADE model is configured."""
        if self.splade is None or not self.splade_vecs:
            return None
        q = self.splade.encode_query(query)
        return np.array([sum(w * d.get(t, 0.0) for t, w in q.items()) for d in self.splade_vecs])
