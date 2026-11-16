"""Shared resources, settings and small render helpers for every page."""

from __future__ import annotations

import html

import streamlit as st

from satyori.config import DEMO_CORPUS_DIR, get_settings
from satyori.index import FastEmbedder, Index, SpladeEncoder
from satyori.ingest import load_file
from satyori.llm import GroqLLM
from satyori.retrieval import TECHNIQUES, FastReranker, Hit, Retrieval, Retriever

SETTINGS = get_settings()

INK, VERMILION, INDIGO, MATCHA, SAFFRON = "#1C1C1E", "#C8412B", "#23395B", "#5E7F36", "#C98F2B"
FAMILY_COLOR = {
    "Lexical": INDIGO,
    "Learned sparse": "#3F7CAC",
    "Semantic": MATCHA,
    "Hybrid": VERMILION,
    "Generative": SAFFRON,
}
STAGE = {  # label and colour for each step of a chunk's journey through the pipeline
    "sparse": ("BM25", INDIGO),
    "dense": ("Dense", MATCHA),
    "splade": ("SPLADE", "#3F7CAC"),
    "hyde": ("HyDE", SAFFRON),
    "fused": ("Fused", VERMILION),
    "rerank": ("Rerank", INK),
    "mmr": ("MMR", "#6B6B70"),
}


def logo(size: int = 52) -> str:
    """An ensō (Zen brush circle) around a bindu (the Indian point of origin): Japan meets India."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 52 52" fill="none" aria-hidden="true">'
        f'<path d="M38.2 11.4 A19 19 0 1 1 29.3 7.3" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
        f'<path d="M38.2 11.4 A19 19 0 0 1 44.5 22" stroke="{INK}" stroke-width="2.5" stroke-linecap="round"/>'
        f'<circle cx="26" cy="26" r="4.5" fill="{VERMILION}"/></svg>'
    )


# ---------------------------------------------------------------- cached resources


@st.cache_resource(show_spinner="Loading the embedding model…")
def embedder() -> FastEmbedder:
    return FastEmbedder(SETTINGS.embedding_model)


@st.cache_resource(show_spinner="Loading the reranker…")
def reranker() -> FastReranker:
    return FastReranker(SETTINGS.reranker_model)


@st.cache_resource(show_spinner="Loading the SPLADE model…")
def splade() -> SpladeEncoder | None:
    return SpladeEncoder(SETTINGS.splade_model) if SETTINGS.splade_model else None


@st.cache_resource(show_spinner="Reading the sample documents…")
def demo_index() -> Index:
    index = Index(embedder(), splade())
    for path in sorted(DEMO_CORPUS_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".md", ".txt"}:
            index.add(load_file(path), SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    return index


@st.cache_resource
def llm() -> GroqLLM | None:
    return GroqLLM(SETTINGS.groq_api_key, SETTINGS.answer_model) if SETTINGS.llm_enabled else None


def index() -> Index:
    """Each browser session gets its own copy of the library, so uploads never leak between users."""
    if "index" not in st.session_state:
        st.session_state.index = demo_index().clone()
    return st.session_state.index


def retriever() -> Retriever:
    return Retriever(index(), reranker(), llm())


def cfg() -> dict:
    """Pipeline settings, changed in the Developer lab and read by the Ask page."""
    return st.session_state.setdefault("cfg", {"technique": "satyori", "top_k": 5, "council": False, "trace": False})


def available(key: str) -> bool:
    t = TECHNIQUES[key]
    return (not t.needs_llm or llm() is not None) and (not t.needs_splade or index().splade is not None)


def tech_label(key: str) -> str:
    t = TECHNIQUES[key]
    if available(key):
        return t.name
    return t.name + (" · needs API key" if t.needs_llm else " · SPLADE off")


# ---------------------------------------------------------------- render helpers


def esc(text: str) -> str:
    return html.escape(text)


def chip(text: str, color: str | None = None, cls: str = "") -> str:
    dot = f'<span class="sy-dot" style="background:{color}"></span>' if color else ""
    return f'<span class="sy-chip {cls}">{dot}{esc(text)}</span>'


def journey(ranks: dict[str, int]) -> str:
    parts = [f'<b style="background:{STAGE[s][1]}">{STAGE[s][0]} #{r}</b>' for s, r in ranks.items() if s in STAGE]
    return '<div class="sy-journey">' + '<span class="arrow">→</span>'.join(parts) + "</div>"


def source_card(n: int, hit: Hit, *, trace: bool = False, snippet: int | None = None) -> str:
    c = hit.chunk
    body = c.text if snippet is None or len(c.text) <= snippet else c.text[:snippet].rsplit(" ", 1)[0] + "…"
    page = f" · page {c.page}" if c.page else ""
    score = f'<span class="score">{hit.score:.3f}</span>' if trace else ""
    return (
        f'<div class="sy-src"><div class="hd"><div><span class="n">{n}</span><span class="doc">{esc(c.doc)}</span>'
        f'<div class="sec">{esc(c.section)}{page}</div></div>{score}</div>'
        f'<div class="body">{esc(body)}</div>{journey(hit.ranks) if trace else ""}</div>'
    )


def sources_html(hits: list[Hit], *, trace: bool = False, snippet: int | None = None) -> str:
    return "".join(source_card(n, h, trace=trace, snippet=snippet) for n, h in enumerate(hits, 1))


def retrieval_notes(r: Retrieval) -> None:
    for note in r.notes:
        label, _, text = note.partition(" · ")
        st.markdown(f'<div class="sy-note"><b>{esc(label)}</b> · {esc(text)}</div>', unsafe_allow_html=True)


def page_header(kicker: str, title: str, subtitle: str = "") -> None:
    sub = f'<p class="sy-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="sy-kicker">{esc(kicker)}</div><h2 class="sy-title">{title}</h2>{sub}', unsafe_allow_html=True
    )
