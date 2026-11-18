"""Ingest a folder of PDF/Markdown/HTML files into the SQLite index."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ragcite.chunking import chunk_sections, merge_small_sections
from ragcite.encoders import Embedder
from ragcite.models import Chunk, Document
from ragcite.parsing import SUPPORTED_SUFFIXES, kind_of, parse_file
from ragcite.store import add_document, set_meta


@dataclass
class IngestStats:
    """Counters reported after an ingest run."""

    documents: int = 0
    sections: int = 0
    chunks: int = 0


def discover_files(folder: Path) -> list[Path]:
    """List supported files under a folder, recursively, in a stable order.

    Args:
        folder: Root folder.

    Returns:
        Sorted file paths with a supported suffix.
    """
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_manifest(folder: Path) -> dict[str, dict]:
    """Read optional manifest.json metadata (title/url/licence) keyed by file name.

    The manifest is looked for in the folder and its parent, so both `corpus/` and
    `corpus/files/` work.

    Args:
        folder: Ingest folder.

    Returns:
        Mapping file name -> manifest entry (empty when there is no manifest).
    """
    for candidate in (folder / "manifest.json", folder.parent / "manifest.json"):
        if candidate.exists():
            entries = json.loads(candidate.read_text()).get("files", [])
            return {Path(e["path"]).name: e for e in entries}
    return {}


def search_text(chunk: Chunk, title: str) -> str:
    """Text used for embeddings and BM25: the passage prefixed with its title and heading.

    Args:
        chunk: The passage.
        title: Document title.

    Returns:
        Contextualised passage text.
    """
    heading = f" | {chunk.section}" if chunk.section else ""
    return f"{title}{heading}\n{chunk.text}"


def ingest_file(
    conn: sqlite3.Connection, path: Path, root: Path, embedder: Embedder, meta: dict, max_words: int = 180
) -> tuple[int, int]:
    """Parse, chunk, embed and store one file.

    Args:
        conn: Index connection.
        path: File to ingest.
        root: Ingest root (used for the stored relative source path).
        embedder: Embedding model.
        meta: Manifest entry for the file (may be empty).
        max_words: Target chunk size.

    Returns:
        (number of sections, number of chunks) stored.
    """
    doc_id = path.stem
    parsed_title, parsed = parse_file(path, doc_id)
    sections = merge_small_sections(parsed, min_words=25)
    chunks = chunk_sections(sections, max_words=max_words)
    title = meta.get("title") or parsed_title or doc_id
    texts = [search_text(c, title) for c in chunks]
    document = Document(
        doc_id, title, str(path.relative_to(root)), kind_of(path), meta.get("url", ""), meta.get("licence", "")
    )
    add_document(conn, document, sections, chunks, embedder.embed(texts), texts)
    return len(sections), len(chunks)


def ingest_folder(conn: sqlite3.Connection, folder: Path, embedder: Embedder, max_words: int = 180) -> IngestStats:
    """Ingest every supported file in a folder.

    Args:
        conn: Index connection.
        folder: Folder to scan recursively.
        embedder: Embedding model (its name is recorded so queries use the same one).
        max_words: Target chunk size.

    Returns:
        IngestStats with totals.
    """
    manifest, stats = load_manifest(folder), IngestStats()
    for path in discover_files(folder):
        sections, chunks = ingest_file(conn, path, folder, embedder, manifest.get(path.name, {}), max_words)
        stats.documents += 1
        stats.sections += sections
        stats.chunks += chunks
    set_meta(conn, "embedder", embedder.name)
    set_meta(conn, "root", str(folder.resolve()))
    return stats
