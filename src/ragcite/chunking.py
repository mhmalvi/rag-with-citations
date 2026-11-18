"""Structure-aware chunking: pack paragraphs of one Section into overlapping passages."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ragcite.models import Chunk, Section

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def word_count(text: str) -> int:
    """Count whitespace-separated words.

    Args:
        text: Any text.

    Returns:
        Number of words.
    """
    return len(text.split())


def split_units(text: str, max_words: int) -> list[str]:
    """Split section text into paragraph-sized units no longer than max_words.

    Paragraphs are separated by blank lines; PDF text without blank lines falls
    back to lines. Over-long paragraphs are split at sentence boundaries.

    Args:
        text: Section text.
        max_words: Upper bound for a unit (sentences longer than this stay whole).

    Returns:
        Non-empty units in order.
    """
    paragraphs = re.split(r"\n\s*\n", text) if "\n\n" in text else text.splitlines()
    units: list[str] = []
    for paragraph in (p.strip() for p in paragraphs):
        if not paragraph:
            continue
        if word_count(paragraph) <= max_words:
            units.append(paragraph)
            continue
        units.extend(_pack(SENTENCE_END.split(paragraph), max_words, overlap=0))
    return units


def _pack(units: list[str], max_words: int, overlap: int) -> list[str]:
    """Greedily join units into passages of up to max_words, repeating `overlap` units.

    Args:
        units: Ordered text units.
        max_words: Target passage size in words.
        overlap: How many trailing units of a passage start the next one.

    Returns:
        Passages joined with newlines.
    """
    passages: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and word_count(" ".join(current + [unit])) > max_words:
            passages.append("\n".join(current))
            current = current[-overlap:] if overlap else []
            if current and word_count(" ".join(current + [unit])) > max_words:
                current = []
        current.append(unit)
    if current:
        passages.append("\n".join(current))
    return passages


def merge_small_sections(sections: list[Section], min_words: int) -> list[Section]:
    """Merge a tiny section into the next one when both share the same page.

    Headings followed by one short line would otherwise become useless chunks.

    Args:
        sections: Sections in reading order.
        min_words: Sections shorter than this are merged forward.

    Returns:
        New list of sections.
    """
    merged: list[Section] = []
    carry: Section | None = None
    for section in sections:
        if carry and carry.page == section.page and carry.doc_id == section.doc_id:
            section = Section(
                section.doc_id, section.page, section.section or carry.section, f"{carry.text}\n\n{section.text}"
            )
        elif carry:
            merged.append(carry)
        carry = section if word_count(section.text) < min_words else None
        if carry is None:
            merged.append(section)
    if carry:
        merged.append(carry)
    return merged


def chunk_sections(sections: Iterable[Section], max_words: int = 180, overlap_units: int = 1) -> list[Chunk]:
    """Cut sections into chunks that never cross a page or heading boundary.

    Run merge_small_sections first if tiny heading-only sections should be folded in;
    the same merged sections must then be stored as the citable units.

    Args:
        sections: Sections of one document, in order.
        max_words: Target chunk size in words.
        overlap_units: Paragraph units shared between consecutive chunks of a section.

    Returns:
        Chunks with ordinals unique within the document.
    """
    chunks: list[Chunk] = []
    for section in sections:
        for text in _pack(split_units(section.text, max_words), max_words, overlap_units):
            chunk_id = f"{section.doc_id}:{len(chunks):04d}"
            chunks.append(Chunk(chunk_id, section.doc_id, len(chunks), section.page, section.section, text))
    return chunks
