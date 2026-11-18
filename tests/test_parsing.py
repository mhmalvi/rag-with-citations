"""Tests for the structure-aware parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.parsing import (
    clean_markdown_inline,
    is_pdf_heading,
    kind_of,
    parse_html,
    parse_markdown,
    parse_pdf_pages,
    repeated_lines,
)
from tests.conftest import HANDBOOK_MD, SECURITY_HTML


def test_markdown_uses_front_matter_title_and_heading_paths():
    """Markdown uses front matter title and heading paths."""
    # Arrange / Act
    title, sections = parse_markdown(HANDBOOK_MD, "handbook")
    # Assert
    assert title == "Staff handbook"
    assert [s.section for s in sections] == [
        "Staff handbook",
        "Staff handbook > Annual leave",
        "Staff handbook > Expenses",
    ]
    assert all(s.page is None for s in sections)


def test_markdown_without_front_matter_takes_first_h1_as_title():
    """Markdown without front matter takes first h1 as title."""
    # Arrange
    text = "# Guide\n\nHello there.\n\n## Part\n\nMore text."
    # Act
    title, sections = parse_markdown(text, "g")
    # Assert
    assert title == "Guide"
    assert sections[-1].section == "Guide > Part"


def test_clean_markdown_inline_strips_links_emphasis_and_liquid():
    """Clean markdown inline strips links emphasis and liquid."""
    # Arrange
    line = "Use [Google Trends](https://x.y) and _can't_ or **bold** {% include x %}`code`"
    # Act
    cleaned = clean_markdown_inline(line)
    # Assert
    assert cleaned == "Use Google Trends and can't or bold code"


def test_html_sections_skip_navigation_and_noise():
    """Html sections skip navigation and noise."""
    # Arrange / Act
    title, sections = parse_html(SECURITY_HTML, "security")
    # Assert
    assert title == "Security policy"
    text = " ".join(s.text for s in sections)
    assert "Home | About" not in text and "Footer text" not in text and "Printable version" not in text
    passwords = next(s for s in sections if s.section.endswith("Passwords"))
    assert passwords.section == "Security policy > Passwords"
    assert "- Use a password manager." in passwords.text


def test_html_nested_blocks_are_not_duplicated():
    """Html nested blocks are not duplicated."""
    # Arrange
    html = "<main><h2>A</h2><ul><li><p>Only once</p></li></ul></main>"
    # Act
    _, sections = parse_html(html, "x")
    # Assert
    assert sections[0].text.count("Only once") == 1


def test_pdf_pages_drop_running_headers_and_split_on_headings():
    """Pdf pages drop running headers and split on headings."""
    # Arrange
    header = "ACME Policy Manual"
    pages = [
        f"{header}\nIntro text on page one.\n1",
        f"{header}\n2. Access Control Rules\nAccess is granted by role.\n2",
        f"{header}\nStill about access on page three.\n3",
    ]
    # Act
    sections = parse_pdf_pages(pages, "manual")
    # Assert
    assert [s.page for s in sections] == [1, 2, 3]
    assert all(header not in s.text for s in sections)
    assert sections[1].section == "2. Access Control Rules"
    assert sections[2].section == "2. Access Control Rules"


def test_repeated_lines_detects_boilerplate_and_page_numbers():
    """Repeated lines detects boilerplate and page numbers."""
    # Arrange
    pages = [["Header", "a", "7"], ["Header", "b"], ["Header", "c"]]
    # Act
    found = repeated_lines(pages)
    # Assert
    assert {"Header", "7"} <= found
    assert "a" not in found


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("3.1. CSF Profiles", True),
        ("Appendix A. CSF Core", True),
        ("3. CSF 2.0 Community Profile for Cyber Incident Risk Management", True),
        ("1. Introduction ........................ 2", False),
        ("100 Bureau Drive (Mail Stop 2000) Gaithersburg, MD 20899-2000", False),
        ("2. Gather the information needed to prepare the Organizational Profile. Examples of", False),
        ("1. A Cur", False),
    ],
)
def test_is_pdf_heading(line, expected):
    """Is pdf heading."""
    # Arrange / Act / Assert
    assert is_pdf_heading(line) is expected


def test_kind_of_rejects_unsupported_files():
    """Kind of rejects unsupported files."""
    # Arrange
    path = Path("notes.docx")
    # Act / Assert
    with pytest.raises(ValueError):
        kind_of(path)
