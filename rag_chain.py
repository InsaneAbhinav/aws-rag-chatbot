"""
rag_chain.py – Core RAG pipeline for the AWS documentation chatbot.

Loads a pre-built FAISS index + metadata, provides retrieval,
non-streaming answers (for eval), and streaming answers (for the UI).
"""

import json
import os
from pathlib import Path
from typing import Generator

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths (relative to this file so it works from any cwd)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_INDEX_PATH = _BASE_DIR / "aws_index.faiss"
_META_PATH = _BASE_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Lazy-loaded singleton resources
# ---------------------------------------------------------------------------
_resources: dict | None = None


def load_resources():
    """Load FAISS index, metadata, and embedding model (cached after first call)."""
    global _resources
    if _resources is not None:
        return _resources

    index = faiss.read_index(str(_INDEX_PATH))

    with open(_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    _resources = {
        "index": index,
        "texts": meta["texts"],
        "sources": meta["sources"],
        "embedder": model,
    }
    return _resources


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(api_key: str) -> ChatGroq:
    """Return a ChatGroq instance. Caller supplies the API key."""
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# API key resolution helper
# ---------------------------------------------------------------------------

def _resolve_api_key(api_key: str | None) -> str:
    """Resolve API key: explicit arg → st.secrets → env var."""
    if api_key:
        return api_key
    # Try Streamlit secrets (may not be available outside Streamlit)
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in .streamlit/secrets.toml "
            "or as an environment variable."
        )
    return key


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Encode *query* and return the top-k nearest chunks from FAISS.

    Returns a list of dicts: {"text": str, "source": str, "score": float}
    """
    res = load_resources()
    embedding = res["embedder"].encode([query]).astype("float32")
    distances, indices = res["index"].search(embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": res["texts"][idx],
            "source": res["sources"][idx],
            "score": float(dist),
        })
    return results


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an AWS documentation expert. Answer the user's question using ONLY \
the provided context. If the context doesn't contain enough information, say so. \
Cite the source PDF when possible.

Context:
{context}"""


def _build_messages(
    query: str, contexts: list[dict], max_chars_per_chunk: int = 1200, max_total_chars: int = 4500
) -> list:
    """
    Build the LangChain message list for the RAG prompt.
    Caps chunk length and total context size to stay safely within Groq's 8 000 TPM limit.
    """
    context_blocks = []
    total_chars = 0

    for c in contexts:
        text = c["text"][:max_chars_per_chunk].strip()
        if total_chars + len(text) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 150:
                text = text[:remaining].strip() + "…"
                context_blocks.append(f"[Source: {c['source']}]\n{text}")
            break
        context_blocks.append(f"[Source: {c['source']}]\n{text}")
        total_chars += len(text)

    context_block = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."
    return [
        SystemMessage(content=_SYSTEM_TEMPLATE.format(context=context_block)),
        HumanMessage(content=query),
    ]


# ---------------------------------------------------------------------------
# Non-streaming answer (used by eval.py)
# ---------------------------------------------------------------------------

def answer(query: str, api_key: str | None = None, k: int = 5) -> tuple[str, list[dict]]:
    """
    Return (response_text, sources) for a given query.

    Non-streaming – waits for the full response before returning.
    """
    api_key = _resolve_api_key(api_key)
    contexts = retrieve(query, k)
    messages = _build_messages(query, contexts)
    llm = get_llm(api_key)
    response = llm.invoke(messages)
    return response.content, contexts


# ---------------------------------------------------------------------------
# Streaming answer (used by app.py)
# ---------------------------------------------------------------------------

def stream_answer(
    query: str, api_key: str | None = None, k: int = 5
) -> Generator:
    """
    Yield sources (once) then token chunks from the LLM.

    First yield:  dict  → {"sources": [{"text": ..., "source": ...}, ...]}
    Next yields:  str   → individual token strings
    """
    api_key = _resolve_api_key(api_key)
    contexts = retrieve(query, k)

    # Yield sources first so the UI can display them immediately
    yield {"sources": [{"text": c["text"], "source": c["source"]} for c in contexts]}

    messages = _build_messages(query, contexts)
    llm = get_llm(api_key)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
