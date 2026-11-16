"""Satyori: insight grounded in truth. Streamlit entry point; pages live in `views/`, logic in `satyori/`."""

from __future__ import annotations

import logging

import streamlit as st

from satyori.config import PROJECT_ROOT
from views import ask, lab, library
from views.common import index, llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Satyori · insight grounded in truth", page_icon="⭕", layout="wide")
st.html(f"<style>{(PROJECT_ROOT / 'assets' / 'style.css').read_text()}</style>")

nav = st.navigation(
    {
        "Satyori": [
            st.Page(ask.page, title="Ask", icon=":material/forum:", default=True),
            st.Page(library.page, title="Library", icon=":material/folder_open:", url_path="library"),
        ],
        "Developer lab": [
            st.Page(lab.pipeline, title="Pipeline", icon=":material/tune:", url_path="pipeline"),
            st.Page(lab.arena, title="Arena", icon=":material/compare:", url_path="arena"),
            st.Page(lab.benchmark, title="Benchmark", icon=":material/monitoring:", url_path="benchmark"),
        ],
    }
)

st.logo(str(PROJECT_ROOT / "assets" / "logo.svg"), icon_image=str(PROJECT_ROOT / "assets" / "mark.svg"), size="large")
with st.sidebar:
    lib = index()
    st.markdown(
        f"<div class='sy-note'>{len(lib.documents)} documents · {len(lib)} passages"
        + ("" if llm() else " · <b>retrieval-only</b> (no API key)")
        + "</div>",
        unsafe_allow_html=True,
    )

nav.run()

st.markdown(
    "<div class='sy-foot'>Satyori · Satya (Sanskrit, truth) + Satori (Japanese, insight) · "
    "built by Ch Sai Kaushik · sample documents are fictional</div>",
    unsafe_allow_html=True,
)
