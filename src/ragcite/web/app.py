"""FastAPI app: one HTML page plus a small JSON API.

Routes:
    GET  /                          the single-page UI
    POST /api/ask                   {"question": "..."} -> answer with verified citations
    GET  /api/documents             indexed documents
    GET  /api/documents/{id}/file   the original file (PDF links can add #page=N)
    GET  /healthz                   liveness probe
"""

from __future__ import annotations

import threading
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ragcite.answer import Answerer
from ragcite.config import Settings
from ragcite.store import connect, get_documents, get_meta

MEDIA_TYPES = {"pdf": "application/pdf", "markdown": "text/plain; charset=utf-8", "html": "text/html; charset=utf-8"}


class AskRequest(BaseModel):
    """Body of POST /api/ask (length-limited to keep prompts bounded)."""

    question: str = Field(min_length=3, max_length=500)


class AppState:
    """Lazily built, thread-safe holder for the answer pipeline."""

    def __init__(self, settings: Settings, answerer: Answerer | None) -> None:
        """Store settings; an answerer may be injected (tests) or is built on first use."""
        self.settings, self._answerer, self._lock = settings, answerer, threading.Lock()

    def answerer(self) -> Answerer:
        """Return the pipeline, building it once (loads ONNX models and the index)."""
        with self._lock:
            if self._answerer is None:
                from ragcite.pipeline import build_answerer, open_index

                self._answerer = build_answerer(self.settings, open_index(self.settings))
            return self._answerer

    def ask(self, question: str) -> dict[str, Any]:
        """Answer one question; the lock also serialises SQLite access across request threads."""
        answerer = self.answerer()
        with self._lock:
            return answerer.answer(question).to_dict()


def resolve_document_file(settings: Settings, doc_id: str) -> tuple[Path, str]:
    """Find the original file of an indexed document, refusing paths outside the ingest root.

    Args:
        settings: Application settings (index location).
        doc_id: Document id from the index.

    Returns:
        (absolute file path, document kind).

    Raises:
        HTTPException: 404 when unknown or missing, 403 if the stored path escapes the root.
    """
    conn = connect(settings.db_path)
    document, root = get_documents(conn).get(doc_id), get_meta(conn, "root")
    conn.close()
    if document is None or root is None:
        raise HTTPException(status_code=404, detail="unknown document")
    base = Path(root).resolve()
    path = (base / document.source).resolve()
    if not path.is_relative_to(base):
        raise HTTPException(status_code=403, detail="forbidden")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file missing")
    return path, document.kind


def _page_routes(app: FastAPI, page: str) -> None:
    """Register the HTML page and the health probe.

    Args:
        app: Application to extend.
        page: HTML of the single-page UI.
    """

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the single-page UI."""
        return page

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}


def _api_routes(app: FastAPI, state: AppState) -> None:
    """Register the JSON API and the original-file route.

    Args:
        app: Application to extend.
        state: Shared pipeline holder.
    """

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        """Answer a question with verified citations."""
        return state.ask(body.question.strip())

    @app.get("/api/documents")
    def documents() -> list[dict[str, str]]:
        """List indexed documents with their public source URL and licence."""
        conn = connect(state.settings.db_path)
        docs = [vars(d) for d in get_documents(conn).values()]
        conn.close()
        return docs

    @app.get("/api/documents/{doc_id}/file")
    def document_file(doc_id: str) -> FileResponse:
        """Serve the original file; HTML is sandboxed so a stored page cannot run script on this origin."""
        path, kind = resolve_document_file(state.settings, doc_id)
        headers = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}
        return FileResponse(path, media_type=MEDIA_TYPES.get(kind, "application/octet-stream"), headers=headers)


def create_app(settings: Settings | None = None, answerer: Answerer | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Settings (defaults from environment).
        answerer: Optional pre-built pipeline (used by tests to inject fakes).

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="rag-with-citations", docs_url="/api/docs")
    page = resources.files("ragcite").joinpath("web", "static", "index.html").read_text(encoding="utf-8")
    _page_routes(app, page)
    _api_routes(app, AppState(settings or Settings(), answerer))
    return app
