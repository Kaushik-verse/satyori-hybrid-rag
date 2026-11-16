"""Retrieval benchmark: run every technique over a labelled question set and score the rankings.

A chunk counts as relevant when its text contains one of the question's `expect` phrases, so the labels
survive any change to chunk size. No LLM is needed to score, which keeps the benchmark cheap and repeatable.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from satyori.index import Index
from satyori.llm import LLM, parse_json
from satyori.retrieval import Retriever


@dataclass
class Question:
    q: str
    expect: list[str]
    type: str = "mixed"


@dataclass
class Row:
    technique: str
    question: str
    type: str
    first_hit: int | None  # 1-based rank of the first relevant chunk, None if missed
    ndcg: float
    latency_ms: float


def load_questions(path: Path) -> list[Question]:
    return [Question(**item) for item in json.loads(path.read_text())]


def relevant(text: str, expect: list[str]) -> bool:
    low = text.lower()
    return any(e.lower() in low for e in expect)


def ndcg(flags: list[bool], n_relevant: int) -> float:
    dcg = sum(1 / math.log2(r + 1) for r, f in enumerate(flags, start=1) if f)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(n_relevant, len(flags)) + 1))
    return dcg / ideal if ideal else 0.0


def run_benchmark(
    retriever: Retriever,
    questions: list[Question],
    techniques: Iterable[str],
    k: int = 5,
    on_progress: Callable[[float], None] | None = None,
) -> list[Row]:
    techniques = list(techniques)
    rows, total, done = [], len(techniques) * len(questions), 0
    for tech in techniques:
        for q in questions:
            n_rel = sum(relevant(c.text, q.expect) for c in retriever.index.chunks)
            res = retriever.run(q.q, tech, k)
            flags = [relevant(h.chunk.text, q.expect) for h in res.hits]
            first = flags.index(True) + 1 if any(flags) else None
            rows.append(Row(tech, q.q, q.type, first, ndcg(flags, n_rel), res.latency_ms))
            done += 1
            if on_progress:
                on_progress(done / total)
    return rows


def summarize(rows: list[Row]) -> list[dict]:
    """One line per technique: Hit@1, Hit@3, Hit@5, MRR, nDCG@5, latency, plus Hit@5 by question type."""
    out = []
    for tech in dict.fromkeys(r.technique for r in rows):
        rs = [r for r in rows if r.technique == tech]
        n = len(rs)
        line = {
            "technique": tech,
            "hit@1": sum(r.first_hit == 1 for r in rs) / n,
            "hit@3": sum(r.first_hit is not None and r.first_hit <= 3 for r in rs) / n,
            "hit@5": sum(r.first_hit is not None for r in rs) / n,
            "mrr": sum(1 / r.first_hit for r in rs if r.first_hit) / n,
            "ndcg@5": sum(r.ndcg for r in rs) / n,
            "latency_ms": sorted(r.latency_ms for r in rs)[n // 2],
        }
        for qtype in ("lexical", "semantic", "mixed"):
            typed = [r for r in rs if r.type == qtype]
            if typed:
                line[f"{qtype} hit@3"] = sum(r.first_hit is not None and r.first_hit <= 3 for r in typed) / len(typed)
        out.append(line)
    return out


def generate_questions(index: Index, llm: LLM, n: int = 10, seed: int = 7) -> list[Question]:
    """Build a test set from the user's own documents: the LLM writes one question per sampled chunk and
    names a short phrase copied from it. Phrases that do not appear verbatim are discarded."""
    pool = [c for c in index.chunks if len(c.text) > 150]
    sample = random.Random(seed).sample(pool, min(n, len(pool)))  # noqa: S311 - sampling, not security
    questions = []
    for chunk in sample:
        reply = llm.complete(
            "You create evaluation questions for a document search engine. Reply only with JSON: "
            '{"question": "...", "phrase": "...", "type": "lexical|semantic"}. The question must be answerable '
            "from the passage. Make half of them paraphrased (semantic) without copying the passage's words, and "
            "half that mention an exact ID, number or name (lexical). `phrase` is 2 to 6 words copied verbatim from "
            "the passage that contain the answer.",
            f"Document: {chunk.doc}\nPassage: {chunk.text}",
            max_tokens=200,
            temperature=0.5,
        )
        data = parse_json(reply)
        if (
            isinstance(data, dict)
            and data.get("question")
            and data.get("phrase")
            and relevant(chunk.text, [data["phrase"]])
        ):
            questions.append(Question(data["question"], [data["phrase"]], data.get("type", "mixed")))
    return questions
