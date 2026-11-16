"""Developer lab: pipeline settings, the retriever Arena, the Benchmark and the index inspector."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from satyori.config import GOLDEN_SET_PATH
from satyori.evaluate import generate_questions, load_questions, run_benchmark, summarize
from satyori.retrieval import TECHNIQUES, Retrieval
from views.common import (
    FAMILY_COLOR,
    SETTINGS,
    available,
    cfg,
    chip,
    esc,
    index,
    llm,
    page_header,
    retrieval_notes,
    retriever,
    sources_html,
    tech_label,
)

# ---------------------------------------------------------------- Pipeline


def pipeline() -> None:
    page_header(
        "Developer lab",
        "Pipeline",
        "Choose how the Ask page retrieves and answers. Changes apply to your next question.",
    )
    c = cfg()
    keys = [k for k in TECHNIQUES if available(k)]
    c["technique"] = st.selectbox(
        "Retrieval technique", keys, index=keys.index(c["technique"]), format_func=lambda k: TECHNIQUES[k].name
    )
    c["top_k"] = st.slider("Sources per answer", 3, 8, c["top_k"])
    c["council"] = st.toggle(
        "Council mode",
        value=c["council"],
        disabled=llm() is None,
        help=f"{len(SETTINGS.council_models)} models draft answers in parallel and a chair model "
        "keeps only what the sources support. Slower, more careful.",
    )
    c["trace"] = st.toggle(
        "Show retrieval trace on the Ask page",
        value=c["trace"],
        help="Scores, the router's decision, and each source's rank at every pipeline stage.",
    )

    st.markdown("##### Techniques")
    rows = [
        {
            "Technique": t.name,
            "Family": t.family,
            "How it works": t.blurb,
            "Rerank": "✓" if t.rerank else "",
            "MMR": "✓" if t.mmr else "",
            "Needs LLM": "✓" if t.needs_llm else "",
        }
        for t in TECHNIQUES.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("##### Models")
    st.markdown(
        f"- Embeddings: `{SETTINGS.embedding_model}` (ONNX via fastembed)\n"
        f"- Learned sparse: `{SETTINGS.splade_model or 'off'}` (SPLADE++)\n"
        f"- Reranker: `{SETTINGS.reranker_model}` (cross-encoder)\n"
        f"- Answers: `{SETTINGS.answer_model}` on Groq" + ("" if llm() else " · *no API key set*") + "\n"
        f"- Council: {', '.join(f'`{m}`' for m in SETTINGS.council_models)}\n"
        f"- Chunking: about {SETTINGS.chunk_size} characters, sentence-aware, "
        f"{SETTINGS.chunk_overlap} sentence overlap, with a document › section header"
    )

    lib = index()
    if lib.documents:
        st.markdown("##### Index inspector")
        st.caption(
            "Each passage is indexed together with its document and section header, so short passages remain findable."
        )
        pick = st.selectbox("Document", lib.documents, key="inspect")
        for ch in [ch for ch in lib.chunks if ch.doc == pick]:
            with st.container(border=True):
                page = f" · p.{ch.page}" if ch.page else ""
                st.markdown(
                    f"<span class='sy-note'><code>{ch.id}</code> · {esc(ch.section)}{page} · "
                    f"{len(ch.text)} chars</span>",
                    unsafe_allow_html=True,
                )
                st.write(ch.text)


# ---------------------------------------------------------------- Arena


def arena() -> None:
    page_header(
        "Developer lab",
        "Arena",
        "One question, several retrievers side by side: what each finds, how fast, and where they agree. "
        "Try an ID like <i>HNR-4-00311-2024-25</i>, then a paraphrase like <i>can the neighbours use my "
        "well?</i>",
    )
    q = st.text_input("Question", value="Can the neighbours walk onto the Malabar property to use the well?")
    picks = st.multiselect(
        "Techniques (up to 4)",
        list(TECHNIQUES),
        default=["sparse", "dense", "satyori"],
        format_func=tech_label,
        max_selections=4,
    )
    k = st.slider("Top k", 3, 8, 5)
    if st.button("Run", type="primary", disabled=not (q and picks)):
        with st.spinner("Running retrievers…"):
            st.session_state.arena = {key: retriever().run(q, key, k) for key in picks if available(key)}

    results: dict[str, Retrieval] = st.session_state.get("arena", {})
    if not results:
        return
    for col, (key, r) in zip(st.columns(len(results)), results.items(), strict=True):
        t = TECHNIQUES[key]
        with col:
            st.markdown(
                f"**{t.name}**<br>" + chip(t.family, FAMILY_COLOR[t.family]) + chip(f"{r.latency_ms:.0f} ms"),
                unsafe_allow_html=True,
            )
            retrieval_notes(r)
            st.markdown(sources_html(r.hits, trace=True, snippet=170), unsafe_allow_html=True)

    st.markdown("##### Rank agreement")
    rows: dict[str, dict] = {}
    for key, r in results.items():
        for rank, h in enumerate(r.hits, 1):
            rows.setdefault(h.chunk.id, {"Passage": f"{h.chunk.doc} › {h.chunk.section}"})[TECHNIQUES[key].name] = rank
    table = pd.DataFrame(rows.values())
    st.dataframe(table, hide_index=True, use_container_width=True)
    everyone = int(table.drop(columns="Passage").notna().all(axis=1).sum())
    st.caption(
        f"{everyone} of {len(table)} passages were found by every technique. A blank cell means that "
        "technique missed the passage."
    )


# ---------------------------------------------------------------- Benchmark


def benchmark() -> None:
    page_header(
        "Developer lab",
        "Benchmark",
        "Every technique answers every question in a labelled set. A passage counts as relevant if it "
        "contains the expected phrase, so scoring needs no LLM and is repeatable.",
    )
    source = st.radio(
        "Question set",
        ["Golden set · 30 questions on the samples", "Generate from my documents"],
        horizontal=True,
        disabled=llm() is None,
    )
    n_gen = st.slider("Questions to generate", 6, 30, 12) if source.startswith("Generate") else 0
    default = [
        k
        for k in ("sparse", "dense", "splade", "rrf", "rrf3", "adaptive", "adaptive3", "rerank", "satyori")
        if available(k)
    ]
    picks = st.multiselect("Techniques", list(TECHNIQUES), default=default, format_func=tech_label)
    st.caption("HyDE and RAG-Fusion call the LLM for every question, about 1 to 3 seconds each.")

    if st.button("Run benchmark", type="primary", disabled=not picks):
        if n_gen:
            with st.spinner("Writing questions from your documents…"):
                questions = generate_questions(index(), llm(), n_gen)
        else:
            questions = load_questions(GOLDEN_SET_PATH)
        bar = st.progress(0.0, text="Benchmarking…")
        rows = run_benchmark(
            retriever(),
            questions,
            [k for k in picks if available(k)],
            on_progress=lambda p: bar.progress(p, text=f"Benchmarking… {p:.0%}"),
        )
        bar.empty()
        st.session_state.bench = (rows, len(questions))

    if "bench" not in st.session_state:
        return
    rows, n_q = st.session_state.bench
    summary = pd.DataFrame(summarize(rows))
    summary["technique"] = summary["technique"].map(lambda k: TECHNIQUES[k].name)
    best = summary.sort_values(["mrr", "latency_ms"], ascending=[False, True]).iloc[0]
    near_best = summary[summary["hit@3"] >= summary["hit@3"].max() - 0.05].sort_values("latency_ms").iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Best MRR", f"{best['mrr']:.2f}", best["technique"], delta_color="off")
    m2.metric("Best Hit@3", f"{summary['hit@3'].max():.0%}")
    m3.metric("Fastest near-best", f"{near_best['latency_ms']:.0f} ms", near_best["technique"], delta_color="off")

    typed = [c for c in summary.columns if c.endswith(" hit@3")]
    table = summary.drop(columns=typed)
    for c in ("hit@1", "hit@3", "hit@5"):
        table[c] = (table[c] * 100).round()
    bars = {
        c: st.column_config.ProgressColumn(c.replace("hit", "Hit"), min_value=0, max_value=100, format="%d%%")
        for c in ("hit@1", "hit@3", "hit@5")
    }
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            **bars,
            "technique": "Technique",
            "mrr": st.column_config.NumberColumn("MRR", format="%.2f"),
            "ndcg@5": st.column_config.NumberColumn("nDCG@5", format="%.2f"),
            "latency_ms": st.column_config.NumberColumn("Median ms", format="%.0f"),
        },
    )

    if typed:
        st.markdown("##### Hit@3 by question type")
        long = summary.melt(id_vars="technique", value_vars=typed, var_name="question type", value_name="score")
        long["question type"] = long["question type"].str.replace(" hit@3", "")
        long["score"] = (long["score"] * 100).round()
        st.bar_chart(
            long,
            x="technique",
            y="score",
            color="question type",
            stack=False,
            height=320,
            y_label="Hit@3 (%)",
            x_label="",
            sort=False,
        )
        st.caption(
            "Lexical questions quote an ID or number, semantic ones paraphrase, mixed ones do both. "
            "BM25 and dense each win one type; hybrids aim to win both."
        )
    misses = [r for r in rows if r.first_hit is None]
    with st.expander(f"Misses: {len(misses)} across {n_q} questions"):
        for r in misses:
            st.markdown(f"- **{TECHNIQUES[r.technique].name}** · {r.question} · *{r.type}*")
