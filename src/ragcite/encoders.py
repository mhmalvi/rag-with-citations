"""Local sentence-transformers models run through ONNX Runtime (no PyTorch needed).

* OnnxEmbedder: bi-encoder (default sentence-transformers/all-MiniLM-L6-v2), mean pooled
  and L2-normalised, used for dense retrieval.
* OnnxCrossEncoder: cross-encoder reranker (default cross-encoder/ms-marco-MiniLM-L6-v2).

Both download their ONNX export once from the Hugging Face Hub into the standard cache.
The Embedder/Reranker protocols let tests swap in tiny deterministic fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]


class Embedder(Protocol):
    """Anything that maps texts to L2-normalised vectors."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 matrix of shape (len(texts), dim)."""
        ...


class Reranker(Protocol):
    """Anything that scores (query, passage) pairs; higher means more relevant."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Return one float score per passage."""
        ...


def resolve_model_dir(repo_id: str) -> Path:
    """Return a local directory holding the ONNX export of a Hub model.

    Args:
        repo_id: Hugging Face repo id, or a local directory path.

    Returns:
        Directory containing onnx/model.onnx and tokenizer.json.
    """
    local = Path(repo_id).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, allow_patterns=MODEL_FILES))


class _OnnxModel:
    """Shared tokenizer + ONNX session plumbing."""

    def __init__(self, repo_id: str, max_length: int, batch_size: int) -> None:
        """Load tokenizer and inference session.

        Args:
            repo_id: Hub repo id or local model directory.
            max_length: Token truncation length.
            batch_size: Texts per inference batch.
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = resolve_model_dir(repo_id)
        self.name = repo_id
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _run(self, batch: list) -> tuple[np.ndarray, np.ndarray]:
        """Tokenise a batch (strings or (a, b) pairs) and run the model.

        Args:
            batch: Inputs for Tokenizer.encode_batch.

        Returns:
            (first model output, attention mask).
        """
        encodings = self.tokenizer.encode_batch(batch)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        return self.session.run(None, feeds)[0], feeds["attention_mask"]


class OnnxEmbedder(_OnnxModel):
    """Mean-pooled, normalised sentence embeddings."""

    def __init__(
        self, repo_id: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 256, batch_size: int = 32
    ) -> None:
        """Load the bi-encoder (see _OnnxModel for arguments)."""
        super().__init__(repo_id, max_length, batch_size)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            float32 matrix (len(texts), dim) with unit-length rows.
        """
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            hidden, mask = self._run(list(texts[start : start + self.batch_size]))
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            vectors.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None))
        return np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), np.float32)


class OnnxCrossEncoder(_OnnxModel):
    """Cross-encoder relevance scores for (query, passage) pairs."""

    def __init__(
        self, repo_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2", max_length: int = 512, batch_size: int = 16
    ) -> None:
        """Load the cross-encoder (see _OnnxModel for arguments)."""
        super().__init__(repo_id, max_length, batch_size)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Score passages against a query.

        Args:
            query: User question.
            passages: Candidate passage texts.

        Returns:
            Raw relevance logits, one per passage.
        """
        scores = []
        for start in range(0, len(passages), self.batch_size):
            pairs = [(query, p) for p in passages[start : start + self.batch_size]]
            logits, _ = self._run(pairs)
            scores.append(np.asarray(logits).reshape(len(pairs), -1)[:, 0])
        return np.concatenate(scores) if scores else np.zeros(0, np.float32)
