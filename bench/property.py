"""Benchmark every offline technique on the property due-diligence question sets.

    uv run python bench/property.py            # golden (30) + held-out (24) sets
    uv run python bench/property.py --llm      # also HyDE and RAG-Fusion (needs GROQ_API_KEY)

The held-out set was written before any tuning and is never used to choose settings.
Results are written to bench/results/property.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from satyori.config import DEMO_CORPUS_DIR, GOLDEN_SET_PATH, get_settings  # noqa: E402
from satyori.evaluate import load_questions, run_benchmark, summarize  # noqa: E402
from satyori.index import FastEmbedder, Index, SpladeEncoder  # noqa: E402
from satyori.ingest import load_file  # noqa: E402
from satyori.llm import GroqLLM  # noqa: E402
from satyori.retrieval import TECHNIQUES, FastReranker, Retriever  # noqa: E402


def main() -> None:
    s = get_settings()
    index = Index(FastEmbedder(s.embedding_model), SpladeEncoder(s.splade_model) if s.splade_model else None)
    for path in sorted(DEMO_CORPUS_DIR.iterdir()):
        index.add(load_file(path), s.chunk_size, s.chunk_overlap)
    use_llm = "--llm" in sys.argv and s.llm_enabled
    llm = GroqLLM(s.groq_api_key, s.answer_model) if use_llm else None
    retriever = Retriever(index, FastReranker(s.reranker_model), llm)
    techniques = [
        k for k, t in TECHNIQUES.items() if (use_llm or not t.needs_llm) and (s.splade_model or not t.needs_splade)
    ]

    results = {}
    for name, path in [("golden", GOLDEN_SET_PATH), ("holdout", GOLDEN_SET_PATH.with_name("holdout_set.json"))]:
        questions = load_questions(path)
        summary = summarize(run_benchmark(retriever, questions, techniques))
        results[name] = summary
        print(f"\n{name} set · {len(questions)} questions · {len(index)} passages")
        print(f"{'technique':13} hit@1  hit@3  MRR   nDCG@5  ms")
        for r in summary:
            print(
                f"{r['technique']:13} {r['hit@1']:.2f}   {r['hit@3']:.2f}   {r['mrr']:.2f}  {r['ndcg@5']:.2f}   "
                f"{r['latency_ms']:.0f}"
            )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "property.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
