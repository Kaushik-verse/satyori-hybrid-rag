"""Ask: the everyday page. A question in, a cited answer out, with the evidence one click away."""

from __future__ import annotations

import logging

import streamlit as st

from satyori.answer import SUPPORTED, Answer, ask
from satyori.config import PROJECT_ROOT
from satyori.retrieval import TECHNIQUES, Retrieval
from views.common import (
    FAMILY_COLOR,
    SAFFRON,
    SETTINGS,
    cfg,
    chip,
    esc,
    index,
    llm,
    logo,
    retrieval_notes,
    retriever,
    sources_html,
)

MARK = PROJECT_ROOT / "assets" / "mark.svg"
EXAMPLES = [
    ("Possession", "When do I get possession of flat B-1204?"),
    ("Heritage", "Can I demolish the Malabar bungalow?"),
    ("Litigation", "What is case CS-2023-8891 about?"),
    ("Title", "Is Tower B still mortgaged to the bank?"),
]


def hero() -> None:
    st.markdown(
        f"""<div class="sy-hero">{logo(64)}
<div><div class="sy-kicker">Property due diligence, answered from your documents</div>
<h1>Satyori</h1>
<div class="sy-meaning"><span><b>Satya</b> <i>Sanskrit</i> · truth</span><span class="plus">+</span>
<span><b>Satori</b> <i>Japanese</i> · sudden insight</span><span class="plus">=</span>
<span class="res">insight grounded in truth</span></div></div></div>
<p class="sy-sub">Ask about sale agreements, RERA certificates, title and encumbrance records, loans or bylaws.
Every answer cites its source, flags documents that contradict each other, and says so when the answer
isn't there.</p>""",
        unsafe_allow_html=True,
    )


def render_answer(a: Answer) -> None:
    trace = cfg()["trace"]
    body = a.text
    if a.conflict:  # shown as a callout instead of a trailing line
        body = "\n".join(line for line in body.splitlines() if "conflict" not in line.lower()[:14])
    st.markdown(body)
    if a.conflict:
        st.markdown(
            f'<div class="sy-conflict"><b>Documents disagree.</b> {esc(a.conflict)}</div>', unsafe_allow_html=True
        )

    chips = []
    if a.abstained:
        chips.append(chip("Not found in your documents", cls="warn"))
    else:
        chips.append(chip(f"{len(a.retrieval.hits)} sources"))
    if a.groundedness is not None:
        g = a.groundedness
        chips.append(chip(f"{g:.0%} of sentences backed by sources", cls="ok" if g >= 0.75 else "warn"))
    if trace:
        t = TECHNIQUES[a.retrieval.technique]
        chips += [chip(t.name, FAMILY_COLOR[t.family]), chip(f"{a.retrieval.latency_ms:.0f} ms retrieval")]
        if a.rewritten_query:
            chips.append(chip("query rewritten"))
        if a.drafts:
            chips.append(chip(f"council of {len(a.drafts)}", SAFFRON))
    st.markdown("".join(chips), unsafe_allow_html=True)

    if a.retrieval.hits:
        with st.expander("Sources"):
            if trace:
                if a.rewritten_query:
                    st.caption(f"Searched for: *{a.rewritten_query}*")
                retrieval_notes(a.retrieval)
            st.markdown(sources_html(a.retrieval.hits, trace=trace), unsafe_allow_html=True)
    if a.checks:
        with st.expander("Evidence check"):
            st.caption(
                "Each sentence is compared with the documents it cites, and every figure in it must appear "
                "in those documents."
            )
            for c in a.checks:
                why = []
                if c.missing_numbers:
                    why.append("figures not found in sources: " + ", ".join(c.missing_numbers))
                if c.support < SUPPORTED:
                    why.append("weak match with sources")
                if not c.cited:
                    why.append("no citation")
                extra = f" <span class='sy-note'>({'; '.join(why)})</span>" if why else ""
                score = f"<code>{c.support:.2f}</code> " if trace else ""
                st.markdown(
                    f"<div class='sy-check'>{'✅' if c.ok else '⚠️'} {score}{esc(c.sentence)}{extra}</div>",
                    unsafe_allow_html=True,
                )
    if trace and a.drafts:
        with st.expander("Council drafts"):
            for model, draft in a.drafts.items():
                st.markdown(f"**{model}**")
                st.markdown(draft)
                st.divider()


def render_message(m: dict) -> None:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else str(MARK)):
        if isinstance(m.get("answer"), Answer):
            render_answer(m["answer"])
        elif isinstance(m.get("retrieval"), Retrieval):
            st.markdown(m["content"])
            st.markdown(sources_html(m["retrieval"].hits, trace=cfg()["trace"]), unsafe_allow_html=True)
        else:
            st.markdown(m["content"])


def respond(prompt: str, messages: list[dict]) -> dict | None:
    c = cfg()
    if not len(index()):
        return {"role": "assistant", "content": "Your library is empty. Add documents on the Library page first."}
    if llm() is None:  # retrieval-only mode: still useful, shows the passages an answer would come from
        r = retriever().run(prompt, c["technique"], c["top_k"])
        return {
            "role": "assistant",
            "content": "Answers need a `GROQ_API_KEY`. Here are the most relevant passages from your documents:",
            "retrieval": r,
        }
    history = [(m["role"], m["content"]) for m in messages[-6:]]
    with st.spinner("Three models are cross-checking…" if c["council"] else "Reading your documents…"):
        try:
            answer = ask(
                retriever(),
                llm(),
                prompt,
                technique=c["technique"],
                k=c["top_k"],
                history=history,
                council=SETTINGS.council_models if c["council"] else (),
                chair=SETTINGS.answer_model,
            )
        except Exception as exc:
            logging.exception("Answer failed")
            if type(exc).__name__ == "RateLimitError":
                st.warning(
                    "Groq's free tier allows a limited number of tokens per minute and that limit was hit. "
                    "Wait about a minute and ask again" + (", or turn off Council mode." if c["council"] else ".")
                )
            else:
                st.error(f"The model call failed ({type(exc).__name__}). Check your API key or try again.")
            return None
    return {"role": "assistant", "content": answer.text, "answer": answer}


def page() -> None:
    messages: list[dict] = st.session_state.setdefault("messages", [])
    conversation, footer = st.container(), st.container()
    # Created last so it sits under the conversation; the containers above are filled afterwards.
    prompt = st.chat_input("Ask about possession dates, title risks, case numbers, clauses…") or st.session_state.pop(
        "pending", None
    )

    with conversation:
        if not messages and not prompt:
            hero()
            st.markdown(
                '<div class="sy-kicker" style="margin-top:8px">Try the sample documents</div>', unsafe_allow_html=True
            )
            cols = st.columns(2)
            for i, (tag, example) in enumerate(EXAMPLES):
                if cols[i % 2].button(f"**{tag}** · {example}", key=f"ex{i}", use_container_width=True):
                    st.session_state.pending = example
                    st.rerun()
            if llm() is None:
                st.info(
                    "No `GROQ_API_KEY` set, so Satyori will show the most relevant passages instead of writing "
                    "an answer. Get a free key at [console.groq.com](https://console.groq.com/keys).",
                    icon="🔑",
                )
        for m in messages:
            render_message(m)
        if prompt:
            render_message({"role": "user", "content": prompt})
            with st.empty():
                reply = respond(prompt, messages)
            messages.append({"role": "user", "content": prompt})
            if reply:
                messages.append(reply)
                render_message(reply)

    if messages:
        with footer:
            c1, c2, _ = st.columns([1, 1, 3])
            report = "\n\n".join(f"**{m['role'].title()}**\n\n{m['content']}" for m in messages)
            c1.download_button("Download chat", report, "satyori_chat.md", "text/markdown", use_container_width=True)
            if c2.button("New chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
