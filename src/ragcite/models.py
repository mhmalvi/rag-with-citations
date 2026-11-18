"""Plain data types shared by ingestion, retrieval, answering and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    """A structural unit of a parsed document (a PDF page segment or a heading section).

    Attributes:
        doc_id: Stable document id (the file stem).
        page: 1-based PDF page number, or None for Markdown/HTML.
        section: Heading path such as "3. Eligibility > Exceptions" ("" when unknown).
        text: Cleaned text of the unit.
    """

    doc_id: str
    page: int | None
    section: str
    text: str

    @property
    def location(self) -> str:
        """Return the citable location key: "page N" for PDFs, else the section path."""
        return f"page {self.page}" if self.page is not None else self.section


@dataclass(frozen=True)
class Document:
    """Metadata for one ingested source file.

    Attributes:
        doc_id: Stable id (file stem).
        title: Human readable title.
        source: Path of the original file relative to the ingest folder.
        kind: "pdf", "markdown" or "html".
        url: Optional public URL of the original.
        licence: Optional licence label.
    """

    doc_id: str
    title: str
    source: str
    kind: str
    url: str = ""
    licence: str = ""


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage carved out of exactly one Section.

    Attributes:
        chunk_id: "<doc_id>:<ordinal>".
        doc_id: Owning document id.
        ordinal: Position of the chunk inside the document.
        page: PDF page (None for Markdown/HTML).
        section: Heading path of the chunk.
        text: Passage text shown to the model and quoted in citations.
    """

    chunk_id: str
    doc_id: str
    ordinal: int
    page: int | None
    section: str
    text: str

    @property
    def location(self) -> str:
        """Return the citable location key, matching Section.location."""
        return f"page {self.page}" if self.page is not None else self.section


@dataclass
class RetrievedChunk:
    """A chunk with the scores each retrieval stage gave it.

    Attributes:
        chunk: The passage.
        title: Title of the owning document.
        score: Final score used for ordering (rerank score when reranked).
        stages: Per-stage ranks/scores, e.g. {"bm25_rank": 3, "dense_rank": 1}.
    """

    chunk: Chunk
    title: str
    score: float
    stages: dict[str, float] = field(default_factory=dict)


@dataclass
class Citation:
    """One inline citation [n] in an answer.

    Attributes:
        number: The marker number used in the answer text.
        source: 1-based index into the passages shown to the model.
        quote: Exact span the model quoted as evidence.
        chunk_id: Chunk the source index resolved to ("" if out of range).
        doc_id: Document of the chunk.
        title: Document title.
        location: "page N" or section path.
        verified: True when the quote was found verbatim (normalised) on the cited page/section.
        highlight: [start, end) character offsets of the quote inside the chunk text, if found.
    """

    number: int
    source: int
    quote: str
    chunk_id: str = ""
    doc_id: str = ""
    title: str = ""
    location: str = ""
    verified: bool = False
    highlight: tuple[int, int] | None = None


@dataclass
class Answer:
    """The result of answering one question.

    Attributes:
        question: The user question.
        text: Answer text with inline [n] markers (or the abstention message).
        abstained: True when the model said the documents do not contain the answer.
        citations: Parsed and verified citations.
        passages: Passages that were shown to the model, in source order.
        model: Model id that produced the answer.
        usage: Token usage counters.
    """

    question: str
    text: str
    abstained: bool
    citations: list[Citation]
    passages: list[RetrievedChunk]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def all_citations_verified(self) -> bool:
        """Return True when the answer has citations and every one of them verified."""
        return bool(self.citations) and all(c.verified for c in self.citations)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the answer (including passages) to JSON-friendly primitives."""
        return {
            "question": self.question,
            "text": self.text,
            "abstained": self.abstained,
            "all_citations_verified": self.all_citations_verified,
            "citations": [asdict(c) for c in self.citations],
            "passages": [
                {"source": i + 1, "title": p.title, "score": p.score, **asdict(p.chunk), "location": p.chunk.location}
                for i, p in enumerate(self.passages)
            ],
            "model": self.model,
            "usage": self.usage,
        }
