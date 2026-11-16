from pathlib import Path

from satyori.config import DEMO_CORPUS_DIR
from satyori.index import BM25, Index, tokenize
from satyori.ingest import Document, chunk_document, load_file
from tests.conftest import make_doc


def test_chunks_follow_section_headings():
    chunks = chunk_document(
        make_doc(
            "Agreement",
            "# Agreement\n\n## 1. Possession\n\nDate is June.\n\n## 2. Pets\n\nDogs are allowed in the park.",
        )
    )
    assert [c.section for c in chunks] == ["1. Possession", "2. Pets"]
    assert chunks[0].indexed_text.startswith("Agreement | 1. Possession")


def test_sentence_crossing_a_pdf_page_break_stays_whole():
    doc = Document(
        "Report",
        [
            "4. HERITAGE STATUS The bungalow is under review for a heritage",
            "tag. If approved, demolition is prohibited.",
        ],
        "fp",
    )
    chunks = chunk_document(doc)
    assert any("heritage tag. If approved" in c.text for c in chunks)
    assert chunks[0].section == "4. Heritage Status"


def test_chunks_respect_size_and_never_split_sentences():
    text = " ".join(f"Sentence number {i} talks about clause {i}." for i in range(60))
    chunks = chunk_document(make_doc("Long", text), size=300, overlap=1)
    assert len(chunks) > 3
    assert all(c.text.endswith(".") for c in chunks)
    assert all(len(c.text) <= 300 + 60 for c in chunks)


def test_tokenizer_keeps_identifiers_and_their_parts():
    tokens = tokenize("Case CS-2023-8891 on Survey No. 88/2")
    assert "cs-2023-8891" in tokens and "8891" in tokens
    assert "88/2" in tokens and "88" in tokens


def test_bm25_prefers_rare_matching_terms():
    bm = BM25([tokenize("possession date june"), tokenize("pets allowed in park"), tokenize("possession of pets")])
    scores = bm.scores(tokenize("park"))
    assert scores.argmax() == 1 and scores[0] == 0


def test_index_dedupes_and_removes(index, embedder):
    before = len(index)
    assert (
        index.add(
            make_doc(
                "Brochure",
                "# Brochure\n\n## 1. Possession\n\nMove in by December 2026 at our "
                "lakeside towers.\n\n## 2. Pets\n\nA large pet park for dogs and cats.",
            )
        )
        == 0
    )
    index.remove("Brochure")
    assert "Brochure" not in index.documents and len(index) < before
    assert index.vectors.shape[0] == len(index)


def test_clone_is_independent(index):
    copy = index.clone()
    copy.remove("Agreement")
    assert "Agreement" in index.documents and "Agreement" not in copy.documents


def test_demo_corpus_loads():
    files = sorted(p for p in DEMO_CORPUS_DIR.iterdir() if p.suffix in {".md", ".pdf"})
    assert len(files) == 9
    chunks = [c for p in files for c in chunk_document(load_file(Path(p)))]
    assert len(chunks) > 40
    assert all(c.text and c.section for c in chunks)


def test_empty_index_returns_nothing(embedder):
    assert Index(embedder).dense_scores(embedder.embed_query("x")).size == 0
