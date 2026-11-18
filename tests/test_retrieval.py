"""Tests for hybrid retrieval, fusion and reranking."""

from __future__ import annotations

import numpy as np
import pytest

from ragcite.retrieval import HybridRetriever, RetrievalConfig, dense_search, reciprocal_rank_fusion
from tests.conftest import FakeEmbedder, FakeReranker


def test_rrf_rewards_items_ranked_high_in_both_lists():
    """Rrf rewards items ranked high in both lists."""
    # Arrange
    bm25, dense = ["a", "b", "c"], ["b", "c", "a"]
    # Act
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    # Assert
    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_dense_search_orders_by_cosine():
    """Dense search orders by cosine."""
    # Arrange
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    # Act
    results = dense_search(matrix, ["x", "y", "z"], np.array([1.0, 0.0], dtype=np.float32), k=2)
    # Assert
    assert [r[0] for r in results] == ["x", "z"]


def test_dense_search_on_empty_index_returns_nothing():
    """Dense search on empty index returns nothing."""
    # Arrange / Act / Assert
    assert dense_search(np.zeros((0, 0)), [], np.zeros(2), k=3) == []


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_the_leave_policy(index, mode):
    """Each mode finds the leave policy."""
    # Arrange
    retriever = HybridRetriever(index, FakeEmbedder(), None, RetrievalConfig(mode=mode, rerank_k=0, final_k=3))
    # Act
    results = retriever.retrieve("How many days of paid annual leave do staff get?")
    # Assert
    assert any("28 days" in r.chunk.text for r in results)
    assert all(r.title for r in results)


def test_reranker_scores_replace_fusion_scores(index):
    """Reranker scores replace fusion scores."""
    # Arrange
    retriever = HybridRetriever(index, FakeEmbedder(), FakeReranker(), RetrievalConfig(final_k=2))
    # Act
    results = retriever.retrieve("report security incident within one hour")
    # Assert
    assert "within one hour" in results[0].chunk.text
    assert results[0].stages["rerank"] == results[0].score
    assert results[0].score >= results[1].score


def test_with_config_shares_index_but_changes_settings(index):
    """With config shares index but changes settings."""
    # Arrange
    base = HybridRetriever(index, FakeEmbedder(), FakeReranker())
    # Act
    clone = base.with_config(final_k=1, mode="bm25")
    # Assert
    assert clone.matrix is base.matrix
    assert base.config.final_k == 6 and clone.config.final_k == 1
    assert len(clone.retrieve("passwords characters")) == 1
