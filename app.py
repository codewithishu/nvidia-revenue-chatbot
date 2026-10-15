"""Streamlit chat UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from nvrag.config import load_config  # noqa: E402
from nvrag.pipeline import RagPipeline  # noqa: E402

st.set_page_config(page_title="NVIDIA Filings Q&A", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading index and models...")
def get_pipeline() -> RagPipeline:
    return RagPipeline(load_config())


def render_sources(sources: list[dict], ids: list[int], title: str) -> None:
    if not ids:
        return
    st.markdown(f"**{title}**")
    for n in ids:
        s = sources[n - 1]
        label = f"[S{n}] {s['period']} · {s['doc_type']} · PDF p.{s['page']} · {s['kind']} · {s['file']}"
        with st.expander(label):
            if s["units"]:
                st.caption(f"Units: {s['units']}")
            if s["section"]:
                st.caption(f"Section: {s['section']}")
            st.code(s["text"], language="markdown" if s["kind"] == "table" else None, wrap_lines=True)
            st.caption(f"dense={s['dense']}  bm25={s['bm25']}  rerank={s['rerank']}")


def render_answer(a: dict) -> None:
    st.markdown(a["answer"])
    if a["abstained"]:
        st.info("Abstained: the indexed documents do not contain enough evidence.")
    for w in a["warnings"]:
        st.warning(w)
    render_sources(a["sources"], a["cited_ids"], "Cited sources")
    others = [s["id"] for s in a["sources"] if s["id"] not in a["cited_ids"]]
    if others and st.session_state.get("show_all"):
        render_sources(a["sources"], others, "Other retrieved passages (not cited)")


try:
    pipe = get_pipeline()
except Exception as e:
    st.error(f"Could not start: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", 3, 20, pipe.cfg.top_k)
    years = st.multiselect("Restrict to years (optional)", pipe.retriever.available_years)
    st.checkbox("Show uncited retrieved passages", key="show_all")
    cov = pipe.cfg.index_dir / "COVERAGE.md"
    if cov.exists():
        with st.expander("Document coverage"):
            st.markdown(cov.read_text())
    if st.button("Clear chat"):
        st.session_state.pop("chat", None)
        st.rerun()
    st.caption(f"Embeddings: {pipe.cfg.embed_model} · Reranker: {'on' if pipe.cfg.use_reranker else 'off'} · "
               f"LLM: {pipe.cfg.llm_provider}")

st.title("NVIDIA annual reports & proxy statements")
st.caption("Answers come only from the ingested filings, with page citations. "
           "Fiscal years end in late January (fiscal 2024 = year ended Jan 28, 2024).")

chat = st.session_state.setdefault("chat", [])
for m in chat:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m["data"])

if q := st.chat_input("Ask about revenue, segments, risks, executive compensation, board..."):
    with st.chat_message("user"):
        st.markdown(q)
    history = [{"role": m["role"], "content": m["content"]} for m in chat]
    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            try:
                a = pipe.ask(q, history, top_k=top_k, years=years or None)
                data = {k: getattr(a, k) for k in ("answer", "abstained", "cited_ids", "sources", "warnings")}
            except Exception as e:
                data = {"answer": f"Error: {e}", "abstained": False, "cited_ids": [], "sources": [], "warnings": []}
        render_answer(data)
    chat += [{"role": "user", "content": q}, {"role": "assistant", "content": data["answer"], "data": data}]
