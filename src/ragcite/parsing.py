"""Structure-aware parsers for PDF, Markdown and HTML.

Each parser returns a list of Section objects. A Section never spans two PDF pages
and never spans two headings, so every chunk cut from it has an exact citable
location (page for PDFs, heading path for Markdown/HTML).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ragcite.models import Section

PDF_SUFFIXES = {".pdf"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | MARKDOWN_SUFFIXES | HTML_SUFFIXES

PDF_HEADING = re.compile(r"^(?:[1-9]\d?(?:\.\d+)*\.?|Appendix [A-Z]\.?)\s+[A-Z](?!.*\.\.)[^!?]{4,78}[^.\d\s,;]$")
MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
HTML_BLOCKS = ["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "td", "th", "pre", "blockquote"]
HTML_HEADINGS = {"h1", "h2", "h3", "h4"}
HTML_DROP = ["script", "style", "nav", "header", "footer", "noscript", "svg", "form"]
HTML_NOISE = {"Printable version", "Print this page"}
CONTINUATION_WORDS = {"a", "an", "and", "as", "for", "in", "of", "on", "or", "that", "the", "to", "with"}


def kind_of(path: Path) -> str:
    """Map a file suffix to a document kind.

    Args:
        path: File path.

    Returns:
        "pdf", "markdown" or "html".

    Raises:
        ValueError: For unsupported suffixes.
    """
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in HTML_SUFFIXES:
        return "html"
    raise ValueError(f"unsupported file type: {path.name}")


def parse_file(path: Path, doc_id: str) -> tuple[str, list[Section]]:
    """Parse any supported file.

    Args:
        path: File to parse.
        doc_id: Id to stamp on every Section.

    Returns:
        (title found in the document or "", sections).
    """
    kind = kind_of(path)
    if kind == "pdf":
        from pypdf import PdfReader

        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        return "", parse_pdf_pages(pages, doc_id)
    text = path.read_text(encoding="utf-8", errors="replace")
    if kind == "markdown":
        return parse_markdown(text, doc_id)
    return parse_html(text, doc_id)


# ---------------------------------------------------------------- PDF


def is_pdf_heading(line: str) -> bool:
    """Heuristic: a short numbered line ("2.1. Scope") that does not read like a wrapped sentence.

    Args:
        line: One stripped line of PDF text.

    Returns:
        True when the line looks like a section heading.
    """
    if not PDF_HEADING.match(line):
        return False
    return line.rsplit(maxsplit=1)[-1].lower() not in CONTINUATION_WORDS


def repeated_lines(pages: list[list[str]], min_share: float = 0.5) -> set[str]:
    """Find running headers/footers: lines that appear on at least min_share of pages.

    Args:
        pages: Lines per page.
        min_share: Fraction of pages a line must appear on to count as boilerplate.

    Returns:
        Set of stripped boilerplate lines (pure page numbers are always included).
    """
    counts = Counter(line for lines in pages for line in {ln.strip() for ln in lines if ln.strip()})
    threshold = max(2, int(len(pages) * min_share))
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    boilerplate |= {ln.strip() for lines in pages for ln in lines if re.fullmatch(r"\d{1,4}", ln.strip())}
    return boilerplate


def parse_pdf_pages(page_texts: list[str], doc_id: str) -> list[Section]:
    """Turn raw per-page PDF text into heading-aware Sections.

    Running headers/footers are removed, and a page is split wherever a numbered
    heading line (e.g. "3.1. CSF Profiles") starts, so sections carry the heading.

    Args:
        page_texts: Extracted text of every page, in order.
        doc_id: Document id.

    Returns:
        Sections in reading order; empty pages are skipped.
    """
    pages = [text.splitlines() for text in page_texts]
    boilerplate = repeated_lines(pages)
    sections: list[Section] = []
    heading = ""
    for number, lines in enumerate(pages, start=1):
        buffer: list[str] = []
        for line in (ln.rstrip() for ln in lines):
            if line.strip() in boilerplate:
                continue
            if is_pdf_heading(line.strip()):
                _flush(sections, doc_id, number, heading, buffer)
                heading, buffer = line.strip(), [line.strip()]
                continue
            buffer.append(line)
        _flush(sections, doc_id, number, heading, buffer)
    return sections


def _flush(sections: list[Section], doc_id: str, page: int | None, heading: str, lines: list[str]) -> None:
    """Append a Section built from buffered lines if it has any text.

    Args:
        sections: Output list (mutated).
        doc_id: Document id.
        page: Page number or None.
        heading: Current heading path.
        lines: Buffered text lines (cleared afterwards).
    """
    text = "\n".join(lines).strip()
    if text:
        sections.append(Section(doc_id=doc_id, page=page, section=heading, text=text))
    lines.clear()


# ---------------------------------------------------------------- Markdown


def strip_front_matter(text: str) -> tuple[str, str]:
    """Remove YAML front matter and return its title.

    Args:
        text: Raw Markdown.

    Returns:
        (title from front matter or "", body without front matter).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not match:
        return "", text
    title = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", match.group(1), flags=re.MULTILINE)
    return (title.group(1) if title else ""), text[match.end() :]


def clean_markdown_inline(line: str) -> str:
    """Strip inline Markdown/Liquid markup so text reads (and quotes) naturally.

    Args:
        line: One Markdown line.

    Returns:
        The line with links, emphasis, inline HTML and template tags removed.
    """
    line = re.sub(r"\{%.*?%\}|\{\{.*?\}\}", "", line)
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
    line = re.sub(r"(?<![\w*])[*_](\S(?:.*?\S)?)[*_](?![\w*])", r"\1", line)
    return line.replace("`", "")


def parse_markdown(text: str, doc_id: str) -> tuple[str, list[Section]]:
    """Split Markdown into one Section per heading (H1-H4), keeping the heading path.

    Args:
        text: Raw Markdown (front matter allowed).
        doc_id: Document id.

    Returns:
        (title, sections). Title comes from front matter, else the first H1.
    """
    title, body = strip_front_matter(text)
    sections: list[Section] = []
    path: list[str] = [title] if title else []
    buffer: list[str] = []
    for raw in body.splitlines():
        heading = MD_HEADING.match(raw)
        if heading:
            _flush(sections, doc_id, None, " > ".join(path), buffer)
            level, name = len(heading.group(1)), clean_markdown_inline(heading.group(2))
            title = title or name
            path = path[: level - 1] + [name] if level > 1 or not path else [name]
            continue
        buffer.append(clean_markdown_inline(raw))
    _flush(sections, doc_id, None, " > ".join(path), buffer)
    return title, sections


# ---------------------------------------------------------------- HTML


def _block_text(element: Tag) -> str:
    """Return the whitespace-collapsed text of an HTML element (list items get a bullet)."""
    text = re.sub(r"\s+", " ", element.get_text()).strip()
    return f"- {text}" if element.name == "li" and text else text


def html_blocks(html: str) -> tuple[str, list[Tag]]:
    """Find the main content and its top-level block elements in document order.

    Args:
        html: Raw HTML.

    Returns:
        (document <title> text, list of block elements not nested in another block).
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(HTML_DROP):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    non_heading = [name for name in HTML_BLOCKS if name not in HTML_HEADINGS]
    blocks = [el for el in root.find_all(HTML_BLOCKS) if not el.find_parent(non_heading)]
    return title, blocks


def parse_html(html: str, doc_id: str) -> tuple[str, list[Section]]:
    """Split HTML into one Section per heading, keeping the heading path (h1 > h2 > h3).

    Args:
        html: Raw HTML.
        doc_id: Document id.

    Returns:
        (title, sections).
    """
    title, blocks = html_blocks(html)
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []
    for element in blocks:
        text = _block_text(element)
        if not text or text in HTML_NOISE:
            continue
        if element.name in HTML_HEADINGS:
            _flush(sections, doc_id, None, " > ".join(path), buffer)
            level = int(element.name[1])
            path = path[: level - 1] + [text]
            continue
        buffer.append(text)
    _flush(sections, doc_id, None, " > ".join(path), buffer)
    return title, sections
