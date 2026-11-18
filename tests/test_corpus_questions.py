"""Guards for the shipped eval set: every gold evidence phrase must exist in its document's chunks.

Runs offline on the committed public corpus (parse + chunk only, no models), so a
changed upstream file or a chunking regression shows up in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.chunking import chunk_sections, merge_small_sections
from ragcite.evaluation import load_questions
from ragcite.faithfulness import normalize
from ragcite.ingest import discover_files
from ragcite.parsing import parse_file

ROOT = Path(__file__).resolve().parent.parent
FILES = {p.stem: p for p in discover_files(ROOT / "corpus" / "files")}
QUESTIONS = load_questions(ROOT / "evals" / "questions.jsonl")


def test_question_set_shape():
    """Question set shape."""
    # Arrange
    ids = [q.id for q in QUESTIONS]
    # Act
    unanswerable = [q for q in QUESTIONS if not q.answerable]
    # Assert
    assert len(ids) == len(set(ids)) and len(QUESTIONS) >= 30
    assert len(unanswerable) >= 5
    assert all(q.doc_id in FILES and q.evidence and q.reference for q in QUESTIONS if q.answerable)


@pytest.mark.parametrize("item", [q for q in QUESTIONS if q.answerable], ids=lambda q: q.id)
def test_evidence_exists_in_gold_document_chunks(item):
    """Evidence exists in gold document chunks."""
    # Arrange
    _, sections = parse_file(FILES[item.doc_id], item.doc_id)
    chunks = chunk_sections(merge_small_sections(sections, min_words=25))
    # Act
    texts = [normalize(c.text) for c in chunks]
    # Assert
    assert any(normalize(e) in t for e in item.evidence for t in texts)
