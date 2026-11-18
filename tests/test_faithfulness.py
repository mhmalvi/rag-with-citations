"""Tests for the code-level quote verification."""

from __future__ import annotations

from ragcite.faithfulness import find_span, normalize, quote_in_text

PAGE = (
    "Every Organizational Profile includes one or both of the following:\n"
    "1. A Cur\nrent Profile specifies the Core outcomes that an organization is currently achieving.\n"
    "Workers get 28 days’ paid annual leave — including bank holidays."
)


def test_exact_quote_is_found():
    """Exact quote is found."""
    # Arrange
    quote = "Every Organizational Profile includes one or both of the following"
    # Act / Assert
    assert quote_in_text(quote, PAGE)


def test_pdf_line_break_artefacts_and_typography_are_tolerated():
    """Pdf line break artefacts and typography are tolerated."""
    # Arrange
    quote = '"A Current Profile specifies the Core outcomes"'
    curly = "Workers get 28 days' paid annual leave - including bank holidays"
    # Act / Assert
    assert quote_in_text(quote, PAGE)
    assert quote_in_text(curly, PAGE)


def test_changed_word_or_number_is_rejected():
    """Changed word or number is rejected."""
    # Arrange
    wrong_number = "Workers get 30 days' paid annual leave"
    paraphrase = "Each Organizational Profile includes one or both"
    # Act / Assert
    assert not quote_in_text(wrong_number, PAGE)
    assert not quote_in_text(paraphrase, PAGE)


def test_ellipsis_parts_must_appear_in_order():
    """Ellipsis parts must appear in order."""
    # Arrange
    in_order = "Every Organizational Profile ... paid annual leave"
    reversed_order = "paid annual leave ... Every Organizational Profile"
    # Act / Assert
    assert quote_in_text(in_order, PAGE)
    assert not quote_in_text(reversed_order, PAGE)


def test_too_short_quotes_do_not_count():
    """Too short quotes do not count."""
    # Arrange / Act / Assert
    assert not quote_in_text("the", PAGE)
    assert not quote_in_text("", PAGE)


def test_find_span_maps_back_to_original_offsets():
    """Find span maps back to original offsets."""
    # Arrange
    quote = "28 days' paid annual leave"
    # Act
    start, end = find_span(quote, PAGE)
    # Assert
    assert PAGE[start:end] == "28 days’ paid annual leave"


def test_normalize_drops_whitespace_and_case():
    """Normalize drops whitespace and case."""
    # Arrange / Act / Assert
    assert normalize("A  B\nc D") == "abcd"
