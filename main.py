import uuid
from typing import Any, Dict, List

import streamlit as st

from backend import run_llm

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
    "sources": [],
}


def _new_session() -> str:
    session_id = uuid.uuid4().hex
    st.session_state.sessions[session_id] = {
        "title": "Untitled chat",
        "messages": [dict(WELCOME_MESSAGE)],
    }
    st.session_state.current_session = session_id
    return session_id


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper -- This is going to be chainging as I add statement parser", layout="centered")
st.title("LangChain Documentation Helper")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if st.session_state.get("current_session") not in st.session_state.sessions:
    _new_session()

with st.sidebar:
    st.subheader("Chat sessions")
    if st.button("➕ New chat", use_container_width=True):
        _new_session()
        st.rerun()

    for sid, sess in list(st.session_state.sessions.items()):
        is_current = sid == st.session_state.current_session
        if st.button(
            sess["title"],
            key=f"session_{sid}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state.current_session = sid
            st.rerun()

    st.divider()
    if st.button("Clear current chat", use_container_width=True):
        current = st.session_state.sessions[st.session_state.current_session]
        current["title"] = "New chat"
        current["messages"] = [dict(WELCOME_MESSAGE)]
        st.rerun()
    if st.button("Delete current chat", use_container_width=True):
        del st.session_state.sessions[st.session_state.current_session]
        st.session_state.pop("current_session", None)
        st.rerun()

current_session = st.session_state.sessions[st.session_state.current_session]
messages: List[Dict[str, Any]] = current_session["messages"]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    if current_session["title"] == "New chat":
        current_session["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")
    messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
