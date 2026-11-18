"""SQLite index: documents, citable sections, chunks with embeddings, and an FTS5 BM25 index.

One file holds everything, so the whole knowledge base is a single portable artefact.
Dense search loads the embedding matrix into memory once (fine up to ~10^5 chunks).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ragcite.models import Chunk, Document, Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
    kind TEXT NOT NULL, url TEXT NOT NULL DEFAULT '', licence TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS sections (
    doc_id TEXT NOT NULL, ordinal INTEGER NOT NULL, page INTEGER, section TEXT NOT NULL,
    text TEXT NOT NULL, PRIMARY KEY (doc_id, ordinal));
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, ordinal INTEGER NOT NULL, page INTEGER,
    section TEXT NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, body, tokenize = 'porter unicode61');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or "
    "our should that the their them there these this to was what when where which who why will "
    "with you your".split()
)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (and create if needed) an index database.

    Args:
        path: SQLite file path, or ":memory:".

    Returns:
        Connection with the schema applied and Row factory set.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def delete_document(conn: sqlite3.Connection, doc_id: str) -> None:
    """Remove a document and everything derived from it (re-ingest is idempotent).

    Args:
        conn: Index connection.
        doc_id: Document to remove.
    """
    ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
    conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(i,) for i in ids])
    for table in ("chunks", "sections", "documents"):
        conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))  # noqa: S608 - fixed names


def add_document(
    conn: sqlite3.Connection,
    document: Document,
    sections: Sequence[Section],
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    search_texts: Sequence[str],
) -> None:
    """Insert one document with its sections, chunks, vectors and BM25 rows.

    Args:
        conn: Index connection.
        document: Document metadata.
        sections: Citable units (used for the page-level faithfulness check).
        chunks: Retrievable passages.
        embeddings: float32 matrix aligned with chunks.
        search_texts: Text indexed by BM25 for each chunk (title + heading + passage).
    """
    delete_document(conn, document.doc_id)
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
        (document.doc_id, document.title, document.source, document.kind, document.url, document.licence),
    )
    conn.executemany(
        "INSERT INTO sections VALUES (?, ?, ?, ?, ?)",
        [(s.doc_id, i, s.page, s.section, s.text) for i, s in enumerate(sections)],
    )
    _insert_chunks(conn, chunks, embeddings, search_texts)
    conn.commit()


def _insert_chunks(
    conn: sqlite3.Connection, chunks: Sequence[Chunk], embeddings: np.ndarray, search_texts: Sequence[str]
) -> None:
    """Insert chunk rows (with float32 vector blobs) and their BM25 rows; the caller commits."""
    rows = [
        (c.chunk_id, c.doc_id, c.ordinal, c.page, c.section, c.text, np.asarray(v, dtype=np.float32).tobytes())
        for c, v in zip(chunks, embeddings, strict=True)
    ]
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    fts_rows = [(c.chunk_id, t) for c, t in zip(chunks, search_texts, strict=True)]
    conn.executemany("INSERT INTO chunks_fts VALUES (?, ?)", fts_rows)


def fts_query(text: str) -> str:
    """Build a safe FTS5 OR-query from free text (every token quoted, stopwords dropped).

    Args:
        text: User question.

    Returns:
        FTS5 MATCH expression, or "" when nothing searchable remains.
    """
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


def bm25_search(conn: sqlite3.Connection, query: str, limit: int) -> list[tuple[str, float]]:
    """Rank chunks with SQLite FTS5's built-in BM25.

    Args:
        conn: Index connection.
        query: User question (free text; it is tokenised and quoted, never interpolated).
        limit: Maximum results.

    Returns:
        (chunk_id, score) pairs, best first; score is -bm25 so higher is better.
    """
    expression = fts_query(query)
    if not expression:
        return []
    rows = conn.execute(
        "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
        (expression, limit),
    )
    return [(r["chunk_id"], -float(r["s"])) for r in rows]


def load_matrix(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    """Load every chunk embedding.

    Args:
        conn: Index connection.

    Returns:
        (chunk ids, float32 matrix with one row per id).
    """
    rows = conn.execute("SELECT chunk_id, embedding FROM chunks ORDER BY doc_id, ordinal").fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return [r["chunk_id"] for r in rows], matrix


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    """Convert a chunks row to a Chunk."""
    return Chunk(row["chunk_id"], row["doc_id"], row["ordinal"], row["page"], row["section"], row["text"])


def get_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
    """Fetch chunks by id.

    Args:
        conn: Index connection.
        chunk_ids: Ids to fetch.

    Returns:
        Mapping id -> Chunk (missing ids are absent).
    """
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", ids)  # noqa: S608 - placeholders
    return {r["chunk_id"]: _row_to_chunk(r) for r in rows}


def get_documents(conn: sqlite3.Connection) -> dict[str, Document]:
    """Return every document keyed by id."""
    rows = conn.execute("SELECT * FROM documents ORDER BY doc_id")
    return {
        r["doc_id"]: Document(r["doc_id"], r["title"], r["source"], r["kind"], r["url"], r["licence"]) for r in rows
    }


def location_text(conn: sqlite3.Connection, doc_id: str, page: int | None, section: str) -> str:
    """Return the full text of a citable location: a whole PDF page, or a whole section.

    Args:
        conn: Index connection.
        doc_id: Document id.
        page: PDF page number (None for Markdown/HTML).
        section: Section path (used when page is None).

    Returns:
        Concatenated text of every stored section at that location ("" if none).
    """
    if page is not None:
        rows = conn.execute("SELECT text FROM sections WHERE doc_id = ? AND page = ? ORDER BY ordinal", (doc_id, page))
    else:
        rows = conn.execute(
            "SELECT text FROM sections WHERE doc_id = ? AND page IS NULL AND section = ? ORDER BY ordinal",
            (doc_id, section),
        )
    return "\n".join(r["text"] for r in rows)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Store an index-level setting (e.g. the embedding model id)."""
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Read an index-level setting, or None."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
