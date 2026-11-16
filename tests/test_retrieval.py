import numpy as np
import pytest

from satyori.retrieval import TECHNIQUES, Retriever, minmax, mmr, route, rrf
from tests.conftest import OverlapReranker, ScriptedLLM


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert next(iter(fused)) == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert set(fused) == {1, 2, 3, 4}


def test_minmax_handles_flat_input():
    assert minmax(np.array([2.0, 2.0])).tolist() == [0.0, 0.0]
    assert minmax(np.array([1.0, 3.0])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    ("query", "alpha"),
    [
        ("What is case CS-2023-8891 about?", 0.3),
        ('Find the clause "earnest money"', 0.3),
        ("Can I knock down the old house?", 0.75),
        ("What does RERA say about escrow", 0.45),
        ("possession timeline", 0.6),
    ],
)
def test_router_balances_dense_and_sparse(query, alpha):
    assert route(query)[0] == alpha


def test_mmr_skips_near_duplicates():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    picked = mmr(np.array([1.0, 0.2]) / np.linalg.norm([1.0, 0.2]), [0, 1, 2], vecs, k=2, lam=0.5)
    assert picked == [0, 2]
    assert mmr(np.array([0.0, 1.0]), [1, 2, 0], vecs, k=3)[0] == 1  # the top candidate is never displaced


@pytest.mark.parametrize("technique", [k for k, t in TECHNIQUES.items() if not t.needs_llm])
def test_every_offline_technique_finds_the_case_number(index, technique):
    r = Retriever(index, OverlapReranker()).run("What is Case Ref CS-2023-8891 about?", technique, k=3)
    assert r.hits, technique
    assert any("CS-2023-8891" in h.chunk.text for h in r.hits)
    assert all(h.ranks for h in r.hits)


def test_rerank_records_confidence_and_stage_ranks(index):
    r = Retriever(index, OverlapReranker()).run("shared borewell neighbours", "rerank", k=2)
    assert r.top_rerank is not None
    assert {"dense", "fused", "rerank"} <= set(r.hits[0].ranks)


def test_adaptive_explains_its_choice(index):
    r = Retriever(index, OverlapReranker()).run("CS-2023-8891", "adaptive")
    assert r.notes and "lexical" in r.notes[0]


def test_llm_techniques_need_an_llm(index):
    with pytest.raises(RuntimeError, match="needs an LLM"):
        Retriever(index, OverlapReranker()).run("pets", "hyde")


def test_hyde_and_multi_query_use_the_llm(index):
    llm = ScriptedLLM(
        "The neighbours may use the borewell under a perpetual easement.",
        '["borewell access rights", "easement over well", "neighbour well"]',
    )
    r = Retriever(index, OverlapReranker(), llm)
    hyde = r.run("Can the neighbours use my well?", "hyde", k=2)
    assert hyde.notes[0].startswith("HyDE passage") and "borewell" in hyde.hits[0].chunk.text
    fusion = r.run("Can the neighbours use my well?", "multi_query", k=2)
    assert "borewell access rights" in fusion.notes[0]


def test_bm25_ignores_chunks_without_shared_words(index):
    r = Retriever(index).run("zzz nonexistent", "sparse")
    assert r.hits == []


def test_zscore_rewards_a_confident_channel():
    from satyori.retrieval import zscore

    confident, unsure = zscore(np.array([10.0, 1.0, 1.0, 1.0])), zscore(np.array([2.0, 1.9, 1.8, 1.7]))
    assert confident[0] > unsure[0]


@pytest.mark.parametrize("technique", ["splade", "rrf3", "adaptive3", "satyori"])
def test_splade_techniques(splade_index, technique):
    r = Retriever(splade_index, OverlapReranker()).run("Case Ref CS-2023-8891", technique, k=3)
    assert "CS-2023-8891" in r.hits[0].chunk.text
    assert "splade" in r.hits[0].ranks


def test_splade_survives_remove_and_clone(splade_index):
    copy = splade_index.clone()
    copy.remove("Brochure")
    assert len(copy.splade_vecs) == len(copy) and len(splade_index.splade_vecs) == len(splade_index)


def test_blended_rerank_keeps_first_stage_signal(index):
    class Undecided:  # a reranker with no opinion: every passage scores the same
        def score(self, query, texts):
            return np.zeros(len(texts))

    first_stage = Retriever(index, Undecided()).run("Case Ref CS-2023-8891", "adaptive3", k=3)
    blended = Retriever(index, Undecided()).run("Case Ref CS-2023-8891", "satyori", k=3)
    assert blended.hits[0].chunk.id == first_stage.hits[0].chunk.id
    assert "CS-2023-8891" in blended.hits[0].chunk.text
