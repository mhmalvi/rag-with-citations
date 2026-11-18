"""Tests for structure-aware chunking."""

from __future__ import annotations

from ragcite.chunking import chunk_sections, merge_small_sections, split_units, word_count
from ragcite.models import Section


def _section(text: str, section: str = "S", page: int | None = 1) -> Section:
    """Build a Section for tests."""
    return Section("doc", page, section, text)


def test_chunks_never_cross_sections_or_pages():
    """Chunks never cross sections or pages."""
    # Arrange
    sections = [_section("alpha " * 50, "A", 1), _section("beta " * 50, "B", 2)]
    # Act
    chunks = chunk_sections(sections, max_words=40)
    # Assert
    for chunk in chunks:
        assert ("alpha" in chunk.text) != ("beta" in chunk.text)
        assert chunk.page == (1 if "alpha" in chunk.text else 2)


def test_chunks_respect_max_words_and_overlap_one_unit():
    """Chunks respect max words and overlap one unit."""
    # Arrange
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(6))
    # Act
    chunks = chunk_sections([_section(paragraphs)], max_words=50, overlap_units=1)
    # Assert
    assert all(word_count(c.text) <= 50 for c in chunks)
    first_last_unit = chunks[0].text.split("\n")[-1]
    assert chunks[1].text.startswith(first_last_unit)


def test_chunk_ids_are_unique_and_ordered():
    """Chunk ids are unique and ordered."""
    # Arrange
    sections = [_section("one two three. " * 30, "A"), _section("four five six. " * 30, "B")]
    # Act
    chunks = chunk_sections(sections, max_words=30)
    # Assert
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_split_units_breaks_long_paragraph_at_sentences():
    """Split units breaks long paragraph at sentences."""
    # Arrange
    paragraph = " ".join(f"Sentence number {i} has five words." for i in range(20))
    # Act
    units = split_units(paragraph, max_words=20)
    # Assert
    assert len(units) > 1
    assert all(u.endswith(".") for u in units)


def test_merge_small_sections_folds_tiny_heading_into_next_on_same_page():
    """Merge small sections folds tiny heading into next on same page."""
    # Arrange
    sections = [
        _section("Tiny intro", "Intro", 1),
        _section("Real content " * 20, "Body", 1),
        _section("Other page text " * 10, "Next", 2),
    ]
    # Act
    merged = merge_small_sections(sections, min_words=5)
    # Assert
    assert len(merged) == 2
    assert merged[0].text.startswith("Tiny intro") and merged[0].section == "Body"


def test_merge_small_sections_keeps_tiny_section_when_page_changes():
    """Merge small sections keeps tiny section when page changes."""
    # Arrange
    sections = [_section("Tiny", "A", 1), _section("Big text " * 20, "B", 2)]
    # Act
    merged = merge_small_sections(sections, min_words=5)
    # Assert
    assert [s.text.split()[0] for s in merged] == ["Tiny", "Big"]
