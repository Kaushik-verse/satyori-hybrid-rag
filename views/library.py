"""Library: add and remove the documents Satyori answers from."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from satyori.index import Index
from satyori.ingest import load_pdf, load_text, load_url
from views.common import SETTINGS, demo_index, embedder, index, page_header, splade


def _add(lib: Index, files, url: str) -> None:
    added, failed = 0, []
    with st.status("Reading documents…", expanded=True) as status:
        for f in files or []:
            try:
                data = f.getvalue()
                doc = load_pdf(f.name, data) if f.name.lower().endswith(".pdf") else load_text(f.name, data)
                n = lib.add(doc, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
                st.write(f"**{doc.name}**: {n} passages" if n else f"**{doc.name}**: already added or empty")
                added += n
            except Exception as exc:
                failed.append(f"{f.name} ({type(exc).__name__})")
        if url.strip():
            try:
                doc = load_url(url.strip())
                n = lib.add(doc, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
                st.write(f"**{doc.name}**: {n} passages")
                added += n
            except Exception as exc:
                failed.append(f"{url} ({type(exc).__name__})")
        status.update(label=f"Added {added} passages", state="complete")
    if failed:
        st.warning("Could not read: " + ", ".join(failed))


def page() -> None:
    lib = index()
    page_header(
        "Library",
        "Your documents",
        "Satyori ships with nine fictional sample documents. Add your own sale agreements, RERA certificates, "
        "title reports or brochures. Files stay in your browser session and are never stored.",
    )

    c1, c2 = st.columns([3, 2])
    files = c1.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
    url = c2.text_input("Or a web page", placeholder="https://")
    if st.button("Add to library", type="primary", disabled=not (files or url)):
        _add(lib, files, url)

    docs = lib.documents
    if docs:
        counts = pd.Series([c.doc for c in lib.chunks]).value_counts()
        st.markdown(f"##### {len(docs)} documents")
        for d in docs:
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.markdown(f"📄 **{d}** <span class='sy-note'>· {counts[d]} passages</span>", unsafe_allow_html=True)
            if b.button("Remove", key=f"rm-{d}", use_container_width=True):
                lib.remove(d)
                st.rerun()
    else:
        st.info("Your library is empty. Upload a document above, or restore the samples.")

    st.divider()
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Restore samples", use_container_width=True):
        st.session_state.index = demo_index().clone()
        st.session_state.pop("bench", None)
        st.rerun()
    if b2.button("Remove all", use_container_width=True):
        st.session_state.index = Index(embedder(), splade())
        st.session_state.pop("bench", None)
        st.rerun()
