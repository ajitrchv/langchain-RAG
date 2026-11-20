import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from backend import run_llm
from ingestion import delete_pdfs, ingest_pdf

DEFAULT_TITLE = "Untitled chat"
HISTORY_FILE = Path(__file__).resolve().parent / "chat_history.json"

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
    "sources": [],
}


def _load_history() -> Dict[str, Any]:
    """Load saved chat sessions from disk; fall back to no history if the file is unusable."""
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        sessions = data["sessions"]
        if not isinstance(sessions, dict) or not all(
            isinstance(s, dict) and isinstance(s.get("messages"), list) and "title" in s
            for s in sessions.values()
        ):
            return {"sessions": {}, "current_session": None}
        return {"sessions": sessions, "current_session": data.get("current_session")}
    except (OSError, ValueError, KeyError, TypeError):
        return {"sessions": {}, "current_session": None}


def _save_history() -> None:
    """Persist all chat sessions to disk (atomic write, owner-only permissions)."""
    payload = {
        "sessions": st.session_state.sessions,
        "current_session": st.session_state.get("current_session"),
    }
    tmp_path = HISTORY_FILE.with_suffix(".tmp")
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, HISTORY_FILE)
    except OSError:
        st.warning("Could not save chat history.")


def _new_session() -> str:
    # Reuse an existing untouched chat (only the welcome message) instead of creating another.
    for existing_id, existing in st.session_state.sessions.items():
        if len(existing["messages"]) <= 1:
            st.session_state.current_session = existing_id
            return existing_id

    session_id = uuid.uuid4().hex
    st.session_state.sessions[session_id] = {
        "title": DEFAULT_TITLE,
        "messages": [dict(WELCOME_MESSAGE)],
        "pdfs": [],
    }
    st.session_state.current_session = session_id
    return session_id


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        + (f" (page {meta['page']})" if meta.get("page") is not None else "")
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


def _drop_pdfs(session_id: str) -> None:
    """Delete the PDF vectors of a session from Pinecone and forget them locally."""
    session = st.session_state.sessions.get(session_id)
    if not session or not session.get("pdfs"):
        return
    try:
        delete_pdfs(session_id)
    except Exception:
        st.warning("Could not remove the uploaded PDF(s) from the index.")
    session["pdfs"] = []


st.set_page_config(page_title="LangChain Documentation Helper -- This is going to be chainging as I add statement parser", layout="centered")
st.title("LangChain Documentation Helper")

if "sessions" not in st.session_state:
    _history = _load_history()
    st.session_state.sessions = _history["sessions"]
    st.session_state.current_session = _history["current_session"]
if st.session_state.get("current_session") not in st.session_state.sessions:
    _new_session()
# Save at the start of every run so changes made right before an st.rerun() are kept.
_save_history()

# Read the prompt before the sidebar renders so a new chat's title shows up immediately.
chat_value = st.chat_input(
    "Ask a question, or attach a PDF to chat with it…",
    accept_file="multiple",
    file_type=["pdf"],
)
prompt = chat_value.text.strip() if chat_value and chat_value.text else None
uploaded_files = list(chat_value.files) if chat_value else []
if prompt:
    _active = st.session_state.sessions[st.session_state.current_session]
    if _active["title"] == DEFAULT_TITLE:
        _active["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")

with st.sidebar:
    st.markdown(
        """
        <style>
        [class*="st-key-active_session_"] button {
            border: 2px solid #e5484d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Chat sessions")
    if st.button("➕ New chat", use_container_width=True):
        _new_session()
        st.rerun()

    for sid, sess in list(st.session_state.sessions.items()):
        is_current = sid == st.session_state.current_session
        if st.button(
            sess["title"],
            key=f"active_session_{sid}" if is_current else f"session_{sid}",
            use_container_width=True,
        ):
            st.session_state.current_session = sid
            st.rerun()

    st.divider()
    if st.button("Clear current chat", use_container_width=True):
        _drop_pdfs(st.session_state.current_session)
        current = st.session_state.sessions[st.session_state.current_session]
        current["title"] = DEFAULT_TITLE
        current["messages"] = [dict(WELCOME_MESSAGE)]
        st.rerun()
    if st.button("Delete current chat", use_container_width=True):
        _drop_pdfs(st.session_state.current_session)
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

if uploaded_files:
    with st.chat_message("assistant"):
        for uploaded in uploaded_files:
            try:
                with st.spinner(f"Indexing {uploaded.name}…"):
                    chunk_count = ingest_pdf(
                        uploaded.getvalue(), uploaded.name, st.session_state.current_session
                    )
                if uploaded.name not in current_session.setdefault("pdfs", []):
                    current_session["pdfs"].append(uploaded.name)
                note = (
                    f"Indexed **{uploaded.name}** ({chunk_count} chunks). "
                    "Answers in this chat are now based only on the uploaded file(s)."
                )
            except ValueError as e:
                note = f"Could not index **{uploaded.name}**: {e}"
            except Exception:
                note = f"Could not index **{uploaded.name}** due to an unexpected error."
            st.markdown(note)
            messages.append({"role": "assistant", "content": note, "sources": []})

if prompt:
    messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(
                    prompt,
                    session_id=st.session_state.current_session,
                    use_pdf=bool(current_session.get("pdfs")),
                )
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

_save_history()
