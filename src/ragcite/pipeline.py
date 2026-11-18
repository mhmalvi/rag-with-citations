"""Factories that wire settings, index, local models and the Claude client together."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from ragcite.answer import Answerer
from ragcite.config import Settings
from ragcite.encoders import OnnxCrossEncoder, OnnxEmbedder, Reranker
from ragcite.llm import LLMClient, make_llm_client
from ragcite.retrieval import HybridRetriever, RetrievalConfig
from ragcite.store import connect, get_meta


@lru_cache(maxsize=4)
def embedder_for(model: str) -> OnnxEmbedder:
    """Load (once per process) the bi-encoder for a model id."""
    return OnnxEmbedder(model)


@lru_cache(maxsize=4)
def reranker_for(model: str) -> Reranker | None:
    """Load (once per process) the cross-encoder, or None when model is "none"."""
    return None if model.lower() == "none" else OnnxCrossEncoder(model)


def open_index(settings: Settings) -> sqlite3.Connection:
    """Open the index and refuse to query it with a different embedding model.

    Args:
        settings: Application settings.

    Returns:
        Index connection.

    Raises:
        RuntimeError: If the index is empty or was built with another embedder.
    """
    conn = connect(settings.db_path)
    built_with = get_meta(conn, "embedder")
    if built_with is None:
        raise RuntimeError(f"index {settings.db_path} is empty - run `ragcite ingest <folder>` first")
    if built_with != settings.embed_model:
        raise RuntimeError(f"index was built with {built_with}, but RAG_EMBED_MODEL is {settings.embed_model}")
    return conn


def build_retriever(settings: Settings, conn: sqlite3.Connection) -> HybridRetriever:
    """Create the default hybrid + rerank retriever for an open index."""
    return HybridRetriever(
        conn,
        embedder_for(settings.embed_model),
        reranker_for(settings.rerank_model),
        RetrievalConfig(final_k=settings.final_k),
    )


def build_answerer(settings: Settings, conn: sqlite3.Connection, llm: LLMClient | None = None) -> Answerer:
    """Create the full answer pipeline (retriever + Claude + citation verification)."""
    llm = llm or make_llm_client(**settings.llm_kwargs())
    return Answerer(build_retriever(settings, conn), llm, settings.answer_model, conn)
