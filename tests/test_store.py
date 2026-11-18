"""Tests for the SQLite/FTS5 index."""

from __future__ import annotations

from ragcite.ingest import ingest_folder
from ragcite.store import (
    bm25_search,
    fts_query,
    get_chunks,
    get_documents,
    get_meta,
    load_matrix,
    location_text,
)
from tests.conftest import FakeEmbedder


def test_fts_query_quotes_tokens_and_drops_stopwords():
    """Fts query quotes tokens and drops stopwords."""
    # Arrange
    question = 'What is the "leave" policy? DROP TABLE chunks; -- NEAR(a b)'
    # Act
    expression = fts_query(question)
    # Assert
    assert expression.startswith('"leave" OR "policy"')
    assert all(part.startswith('"') and part.endswith('"') for part in expression.split(" OR "))


def test_bm25_search_ranks_matching_chunk_first(index):
    """Bm25 search ranks matching chunk first."""
    # Arrange / Act
    results = bm25_search(index, "travel expenses receipt", limit=5)
    # Assert
    top = get_chunks(index, [results[0][0]])[results[0][0]]
    assert "Travel expenses" in top.text


def test_bm25_search_with_only_stopwords_returns_nothing(index):
    """Bm25 search with only stopwords returns nothing."""
    # Arrange / Act / Assert
    assert bm25_search(index, "what is the", limit=5) == []


def test_ingest_records_documents_meta_and_vectors(index):
    """Ingest records documents meta and vectors."""
    # Arrange / Act
    documents = get_documents(index)
    ids, matrix = load_matrix(index)
    # Assert
    assert set(documents) == {"handbook", "security"}
    assert documents["handbook"].kind == "markdown" and documents["security"].kind == "html"
    assert matrix.shape == (len(ids), 128)
    assert get_meta(index, "embedder") == "fake-embedder"


def test_location_text_returns_whole_section(index):
    """Location text returns whole section."""
    # Arrange / Act
    text = location_text(index, "security", None, "Security policy > Incidents")
    # Assert
    assert "within one hour" in text


def test_reingest_is_idempotent(index, corpus_dir):
    """Reingest is idempotent."""
    # Arrange
    before = len(load_matrix(index)[0])
    # Act
    ingest_folder(index, corpus_dir, FakeEmbedder())
    # Assert
    assert len(load_matrix(index)[0]) == before
    assert len(bm25_search(index, "passwords", limit=50)) >= 1
