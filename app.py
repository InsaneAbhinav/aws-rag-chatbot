"""
app.py – Streamlit chat UI for the AWS RAG Chatbot.

Streams answers from rag_chain.py and displays source citations.
Run with: uv run streamlit run app.py
"""

import os

import streamlit as st

from rag_chain import load_resources, stream_answer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AWS RAG Chatbot",
    page_icon="☁️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("☁️ AWS RAG Chatbot")
    st.markdown(
        "Ask questions about AWS services and get answers grounded in "
        "official AWS documentation."
    )
    st.divider()
    st.markdown(
        "**Data source:** "
        "[AWS Public PDF Chunks]"
        "(https://huggingface.co/datasets/semihk1/aws-public-pdf-chunked-dataset) "
        "(first 5 000 chunks)"
    )
    st.markdown(
        "**Model:** `openai/gpt-oss-120b` via [Groq](https://groq.com)"
    )
    st.divider()
    st.caption("Built with Streamlit · LangChain · FAISS · Sentence Transformers")


# ---------------------------------------------------------------------------
# Cache: pre-load FAISS index + embedding model once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FAISS index & embedding model…")
def init():
    load_resources()
    return True


init()


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """Resolve Groq API key: st.secrets → env var."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

if not api_key or api_key == "your-groq-api-key-here":
    st.warning(
        "⚠️ Please set your Groq API key in `.streamlit/secrets.toml` or "
        "as the `GROQ_API_KEY` environment variable.",
        icon="🔑",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}**: {s['text'][:200]}…")


# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask about AWS…"):
    # 1. Show & save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream assistant response
    with st.chat_message("assistant"):
        sources: list[dict] = []
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in stream_answer(prompt, api_key=api_key):
                if isinstance(chunk, dict):
                    # First yield is the sources dict
                    sources = chunk["sources"]
                else:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            # Final render (remove cursor)
            response_placeholder.markdown(full_response)

            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['source']}**: {s['text'][:200]}…")

            # 3. Save assistant message only on success
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })
        except Exception as e:
            response_placeholder.empty()
            st.error(f"⚠️ Groq API Error: {e}")
