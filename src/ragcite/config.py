"""Runtime settings, read from environment variables with sensible defaults.

No secret is ever stored here: the Anthropic SDK reads ANTHROPIC_API_KEY from the
environment itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Return an environment variable or a default (empty values count as unset)."""
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Attributes:
        db_path: SQLite index file.
        answer_model: Claude model that writes answers.
        judge_model: Claude model used as the correctness judge in evals.
        embed_model: Hub id (or local dir) of the sentence-transformers bi-encoder.
        rerank_model: Hub id (or local dir) of the cross-encoder; "none" disables reranking.
        final_k: Passages shown to the model per question.
        llm_backend: "api" (Anthropic SDK) or "claude-cli" (signed-in Claude Code CLI).
        base_url: Anthropic-compatible endpoint for the SDK ("" = api.anthropic.com / ANTHROPIC_BASE_URL).
        api_key_env: Name of the env var holding the API key (default ANTHROPIC_API_KEY).
    """

    db_path: Path = field(default_factory=lambda: Path(_env("RAG_DB_PATH", ".ragcite/index.sqlite")))
    answer_model: str = field(default_factory=lambda: _env("RAG_ANSWER_MODEL", "claude-sonnet-5-5"))
    judge_model: str = field(default_factory=lambda: _env("RAG_JUDGE_MODEL", "claude-sonnet-5-5"))
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    rerank_model: str = field(default_factory=lambda: _env("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    final_k: int = field(default_factory=lambda: int(_env("RAG_FINAL_K", "6")))
    llm_backend: str = field(default_factory=lambda: _env("RAG_LLM_BACKEND", "api"))
    base_url: str = field(default_factory=lambda: _env("RAG_BASE_URL", ""))
    api_key_env: str = field(default_factory=lambda: _env("RAG_API_KEY_ENV", "ANTHROPIC_API_KEY"))

    def llm_kwargs(self) -> dict[str, str]:
        """Arguments for llm.make_llm_client built from these settings."""
        return {"backend": self.llm_backend, "base_url": self.base_url, "api_key_env": self.api_key_env}
