from pathlib import Path

from satyori.answer import Answer, ask, grounding_check, tidy
from satyori.config import DEMO_CORPUS_DIR, GOLDEN_SET_PATH
from satyori.evaluate import Question, load_questions, ndcg, relevant, run_benchmark, summarize
from satyori.ingest import chunk_document, load_file
from satyori.retrieval import Retrieval, Retriever
from tests.conftest import OverlapReranker, ScriptedLLM


def test_tidy_normalises_model_typography():
    assert tidy("Due 30 June 2027【2】 for CS‑2023‑8891") == "Due 30 June 2027[2] for CS-2023-8891"


def test_conflict_line_is_parsed_and_none_is_ignored():
    r = Retrieval("satyori", "q", [])
    assert Answer("x\n**Conflict:** brochure says 2026", r).conflict == "brochure says 2026"
    assert Answer("x\nConflict: None; sources agree", r).conflict is None
    assert Answer("x", r).conflict is None


def test_answer_cites_sources_and_passes_grounding(index):
    retriever = Retriever(index, OverlapReranker())
    llm = ScriptedLLM("The Promoter shall hand over possession on or before 30 June 2027 [1].")
    a = ask(retriever, llm, "When is possession due?", technique="rerank", k=3)
    assert not a.abstained
    assert "Sources:" in llm.calls[0][1] and "[1]" in llm.calls[0][1]
    assert a.groundedness == 1.0


def test_invented_figures_are_flagged(index, embedder):
    hits = Retriever(index, OverlapReranker()).run("possession date", "rerank", k=2).hits
    checks = grounding_check(embedder, "The Promoter shall hand over possession on or before 30 June 2031 [1].", hits)
    assert checks and "2031" in checks[0].missing_numbers and not checks[0].ok


def test_weak_retrieval_is_rewritten_then_abstains(index):
    llm = ScriptedLLM("NONE")
    a = ask(Retriever(index, OverlapReranker()), llm, "Who won the cricket final?", technique="rerank")
    assert a.abstained and "couldn't find" in a.text
    assert len(llm.calls) == 1  # only the rewrite; no answer was generated


def test_follow_up_is_condensed_with_history(index):
    llm = ScriptedLLM("What is the cancellation forfeit for the Agreement?", "10 percent is forfeited [1].")
    a = ask(
        Retriever(index, OverlapReranker()),
        llm,
        "and if I cancel?",
        technique="rerank",
        history=[("user", "When is possession due?"), ("assistant", "30 June 2027")],
    )
    assert a.rewritten_query == "What is the cancellation forfeit for the Agreement?"
    assert "forfeit" in a.retrieval.hits[0].chunk.text


def test_council_drafts_then_chair_merges(index):
    llm = ScriptedLLM("Draft A [1].", "Draft B [1].", "Merged answer: 30 June 2027 [1].")
    a = ask(
        Retriever(index, OverlapReranker()),
        llm,
        "When is possession due?",
        technique="rerank",
        council=("m1", "m2"),
        chair="chair",
    )
    assert set(a.drafts) == {"m1", "m2"}
    assert llm.calls[-1][2] == "chair" and "Draft from analyst" in llm.calls[-1][1]


def test_ndcg_and_summary():
    assert ndcg([True, False], 1) == 1.0
    assert 0 < ndcg([False, True], 1) < 1
    assert ndcg([False, False], 1) == 0.0


def test_benchmark_runs_on_small_set(index):
    qs = [
        Question("Case CS-2023-8891", ["CS-2023-8891"], "lexical"),
        Question("neighbours use the well", ["shared borewell"], "semantic"),
    ]
    rows = run_benchmark(Retriever(index, OverlapReranker()), qs, ["sparse", "rrf"], k=3)
    summary = summarize(rows)
    assert [s["technique"] for s in summary] == ["sparse", "rrf"]
    assert all(0 <= s["mrr"] <= 1 for s in summary)


def test_golden_set_labels_exist_in_demo_corpus():
    chunks = [c for p in sorted(DEMO_CORPUS_DIR.iterdir()) for c in chunk_document(load_file(Path(p)))]
    questions = load_questions(GOLDEN_SET_PATH)
    assert len(questions) == 30
    for q in questions:
        assert any(relevant(c.text, q.expect) for c in chunks), q.q
