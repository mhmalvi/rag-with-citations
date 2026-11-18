"""Hybrid retrieval: BM25 (FTS5) + dense vectors, fused with RRF, then cross-encoder reranking."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ragcite.encoders import Embedder, Reranker
from ragcite.models import RetrievedChunk
from ragcite.store import bm25_search, get_chunks, get_documents, load_matrix


@dataclass(frozen=True)
class RetrievalConfig:
    """Knobs for the retrieval pipeline.

    Attributes:
        mode: "hybrid", "bm25" or "dense" (first-stage candidates).
        bm25_k: Candidates taken from BM25.
        dense_k: Candidates taken from dense search.
        rrf_k: Reciprocal-rank-fusion constant.
        rerank_k: Fused candidates sent to the reranker (0 disables reranking).
        final_k: Passages returned.
    """

    mode: str = "hybrid"
    bm25_k: int = 30
    dense_k: int = 30
    rrf_k: int = 60
    rerank_k: int = 20
    final_k: int = 6


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id lists with RRF: score(d) = sum 1 / (k + rank).

    Args:
        rankings: Ranked lists of ids (best first).
        k: Damping constant (60 is the value from the original RRF paper).

    Returns:
        (id, fused score) pairs, best first; ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


def dense_search(matrix: np.ndarray, ids: Sequence[str], query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Cosine top-k over normalised vectors.

    Args:
        matrix: (n, dim) normalised chunk vectors.
        ids: Chunk ids aligned with matrix rows.
        query_vector: (dim,) normalised query vector.
        k: Results to return.

    Returns:
        (chunk_id, cosine) pairs, best first.
    """
    if not len(ids):
        return []
    similarities = matrix @ query_vector
    top = np.argsort(-similarities)[:k]
    return [(ids[i], float(similarities[i])) for i in top]


class HybridRetriever:
    """Retrieves passages for a question from one index."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Load the embedding matrix and document titles.

        Args:
            conn: Index connection.
            embedder: Must be the model the index was built with.
            reranker: Cross-encoder, or None to skip reranking.
            config: Retrieval settings.
        """
        self.conn, self.embedder, self.reranker = conn, embedder, reranker
        self.config = config or RetrievalConfig()
        self.ids, self.matrix = load_matrix(conn)
        self.titles = {d.doc_id: d.title for d in get_documents(conn).values()}

    def with_config(self, **changes: object) -> HybridRetriever:
        """Return a retriever sharing this one's index and models but with changed settings."""
        clone = object.__new__(HybridRetriever)
        clone.__dict__.update(self.__dict__)
        clone.config = replace(self.config, **changes)
        return clone

    def candidates(self, question: str) -> list[tuple[str, dict[str, float]]]:
        """First stage: BM25 and/or dense candidates, fused with RRF.

        Args:
            question: User question.

        Returns:
            (chunk_id, stage info) pairs in fused order.
        """
        cfg, lists, info = self.config, [], {}
        if cfg.mode in ("hybrid", "bm25"):
            bm25 = bm25_search(self.conn, question, cfg.bm25_k)
            lists.append([cid for cid, _ in bm25])
            for rank, (cid, _) in enumerate(bm25, 1):
                info.setdefault(cid, {})["bm25_rank"] = rank
        if cfg.mode in ("hybrid", "dense"):
            dense = dense_search(self.matrix, self.ids, self.embedder.embed([question])[0], cfg.dense_k)
            lists.append([cid for cid, _ in dense])
            for rank, (cid, _) in enumerate(dense, 1):
                info.setdefault(cid, {})["dense_rank"] = rank
        fused = reciprocal_rank_fusion(lists, cfg.rrf_k)
        return [(cid, {**info[cid], "rrf": score}) for cid, score in fused]

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Run the full pipeline and return the final passages.

        Args:
            question: User question.

        Returns:
            Up to config.final_k passages, best first.
        """
        cfg = self.config
        pool = self.candidates(question)
        if self.reranker is not None and cfg.rerank_k > 0:
            pool = pool[: cfg.rerank_k]
        chunks = get_chunks(self.conn, [cid for cid, _ in pool])
        results = [
            RetrievedChunk(chunks[cid], self.titles.get(chunks[cid].doc_id, ""), stages["rrf"], stages)
            for cid, stages in pool
            if cid in chunks
        ]
        if self.reranker is not None and cfg.rerank_k > 0 and results:
            scores = self.reranker.score(question, [f"{r.title}\n{r.chunk.text}" for r in results])
            for result, score in zip(results, scores, strict=True):
                result.score = float(score)
                result.stages["rerank"] = float(score)
            results.sort(key=lambda r: -r.score)
        return results[: cfg.final_k]
