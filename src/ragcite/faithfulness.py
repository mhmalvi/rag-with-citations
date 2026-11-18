"""Code-level citation checks: does a quoted span really occur in the cited text?

Matching is exact after a light normalisation that only removes differences a PDF
text layer or a model's typography would introduce: Unicode compatibility forms,
curly quotes, dash variants, case and whitespace (including line-break hyphenation
artefacts such as "Cur\\nrent"). Words, numbers and punctuation must match.
An ellipsis ("...") in a quote splits it into parts that must appear in order.
"""

from __future__ import annotations

import re
import unicodedata

SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x2033)
DASHES = (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)
TRANSLATION = str.maketrans(
    {
        **dict.fromkeys(map(chr, SINGLE_QUOTES), "'"),
        **dict.fromkeys(map(chr, DOUBLE_QUOTES), '"'),
        **dict.fromkeys(map(chr, DASHES), "-"),
        chr(0x00A0): " ",  # no-break space
        chr(0x00AD): "",  # soft hyphen
    }
)
ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")
MIN_PART_CHARS = 8


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise text for matching and remember where each kept character came from.

    Args:
        text: Original text.

    Returns:
        (normalised string, list mapping each normalised char index to an index in `text`).
    """
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        folded = unicodedata.normalize("NFKC", char).translate(TRANSLATION).lower()
        for piece in folded:
            if not piece.isspace():
                chars.append(piece)
                positions.append(index)
    return "".join(chars), positions


def normalize(text: str) -> str:
    """Return only the normalised form of text (see normalize_with_map)."""
    return normalize_with_map(text)[0]


def quote_parts(quote: str) -> list[str]:
    """Split a quote at ellipses into normalised parts, dropping empty pieces.

    Args:
        quote: The quoted span, possibly with "..." elisions and surrounding quote marks.

    Returns:
        Normalised parts in order.
    """
    stripped = quote.strip().strip("\"'“”‘’")
    return [normalize(part) for part in ELLIPSIS.split(stripped) if normalize(part)]


def find_span(quote: str, text: str) -> tuple[int, int] | None:
    """Locate a quote inside text.

    Args:
        quote: Quoted span (ellipses allowed).
        text: Text that should contain it.

    Returns:
        (start, end) character offsets in the original text covering the whole quote,
        or None when any part is missing, out of order, or the quote is too short to mean anything.
    """
    parts = quote_parts(quote)
    if not parts or sum(len(p) for p in parts) < MIN_PART_CHARS:
        return None
    haystack, positions = normalize_with_map(text)
    cursor, start = 0, None
    for part in parts:
        found = haystack.find(part, cursor)
        if found < 0:
            return None
        start = found if start is None else start
        cursor = found + len(part)
    return positions[start], positions[cursor - 1] + 1


def quote_in_text(quote: str, text: str) -> bool:
    """Return True when the quote occurs in text under the normalisation rules above."""
    return find_span(quote, text) is not None
