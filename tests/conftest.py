"""Shared offline fakes and fixtures: no network, no ONNX models, no API key."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragcite.ingest import ingest_folder
from ragcite.llm import LLMResult
from ragcite.store import connect

TOKEN = re.compile(r"\w+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (unit-normalised)."""

    name = "fake-embedder"

    def __init__(self, dim: int = 128) -> None:
        """Set the vector size."""
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Hash each lower-cased token into a bucket and normalise."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN.findall(text.lower()):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class FakeReranker:
    """Scores passages by how many distinct query tokens they contain."""

    name = "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Return the token-overlap count per passage."""
        wanted = set(TOKEN.findall(query.lower()))
        return np.array([len(wanted & set(TOKEN.findall(p.lower()))) for p in passages], dtype=np.float32)


class FakeLLM:
    """Records calls and returns a scripted JSON payload (or the result of a callable)."""

    def __init__(self, reply: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Store the scripted reply."""
        self.reply, self.calls = reply, []

    def complete_json(self, **kwargs: Any) -> LLMResult:
        """Record the request and return the scripted reply."""
        self.calls.append(kwargs)
        data = self.reply(kwargs) if callable(self.reply) else self.reply
        return LLMResult(data, kwargs["model"], {"input_tokens": 100, "output_tokens": 20})


HANDBOOK_MD = """---
title: Staff handbook
---

Intro paragraph about the handbook and who it is for, written for every employee of the company.

## Annual leave

Full-time staff get 28 days of paid annual leave each year, including bank holidays.
Requests must be made at least two weeks in advance through the HR portal.

## Expenses

Travel expenses are reimbursed within 30 days when a receipt is attached to the claim.
"""

SECURITY_HTML = """<html><head><title>Security policy</title></head><body>
<nav>Home | About</nav>
<main>
<h1>Security policy</h1>
<p>Printable version</p>
<h2>Passwords</h2>
<p>Passwords must be at least 14 characters long and must never be shared with anyone.</p>
<ul><li>Use a password manager.</li><li>Turn on multi-factor authentication.</li></ul>
<h2>Incidents</h2>
<p>Report a suspected security incident to the security team within one hour of noticing it.</p>
</main><footer>Footer text</footer></body></html>"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A tiny two-document corpus (Markdown + HTML) on disk."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text(HANDBOOK_MD)
    (folder / "security.html").write_text(SECURITY_HTML)
    return folder


@pytest.fixture
def index(corpus_dir: Path, tmp_path: Path):
    """An index built from corpus_dir with the fake embedder."""
    conn = connect(tmp_path / "index.sqlite")
    ingest_folder(conn, corpus_dir, FakeEmbedder())
    yield conn
    conn.close()
