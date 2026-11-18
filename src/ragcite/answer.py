"""Answer a question from retrieved passages with Claude, then verify every citation in code."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from importlib import resources
from typing import Any, Protocol

from ragcite.faithfulness import find_span, quote_in_text
from ragcite.llm import LLMClient, LLMRefusal
from ragcite.models import Answer, Citation, RetrievedChunk
from ragcite.store import location_text

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["source", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}
MARKER = re.compile(r"\[(\d+)\]")


class Retriever(Protocol):
    """Anything with retrieve(question) -> passages."""

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return passages for a question, best first."""
        ...


def load_prompt(name: str) -> str:
    """Read a prompt file shipped inside the package (src/ragcite/prompts/)."""
    return resources.files("ragcite").joinpath("prompts", name).read_text(encoding="utf-8")


def format_sources(passages: Sequence[RetrievedChunk]) -> str:
    """Render passages as numbered XML-ish source blocks for the prompt.

    Args:
        passages: Retrieved passages in the order they will be numbered.

    Returns:
        Prompt text with one <source> element per passage.
    """
    blocks = []
    for number, passage in enumerate(passages, start=1):
        chunk = passage.chunk
        blocks.append(
            f'<source id="{number}" document="{passage.title}" location="{chunk.location}">\n{chunk.text}\n</source>'
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, passages: Sequence[RetrievedChunk]) -> str:
    """Combine sources and question into the user turn."""
    return f"<sources>\n{format_sources(passages)}\n</sources>\n\n<question>{question}</question>"


def verify_citation(citation: Citation, passages: Sequence[RetrievedChunk], conn: sqlite3.Connection | None) -> None:
    """Resolve a citation to its passage and check the quote exists on the cited page/section.

    The quote must occur in the passage that was shown to the model *and* in the stored
    text of the cited location (whole PDF page or whole section). Mutates the citation.

    Args:
        citation: Citation with source and quote filled in.
        passages: Passages shown to the model.
        conn: Index connection for the page-level check (None checks the passage only).
    """
    if not 1 <= citation.source <= len(passages):
        return
    passage = passages[citation.source - 1]
    chunk = passage.chunk
    citation.chunk_id, citation.doc_id, citation.title = chunk.chunk_id, chunk.doc_id, passage.title
    citation.location = chunk.location
    citation.highlight = find_span(citation.quote, chunk.text)
    on_page = True
    if conn is not None:
        on_page = quote_in_text(citation.quote, location_text(conn, chunk.doc_id, chunk.page, chunk.section))
    citation.verified = citation.highlight is not None and on_page


def parse_answer(
    question: str, payload: dict[str, Any], passages: Sequence[RetrievedChunk], conn: sqlite3.Connection | None
) -> Answer:
    """Turn the model's JSON into an Answer with verified citations.

    Args:
        question: The user question.
        payload: Object matching ANSWER_SCHEMA.
        passages: Passages shown to the model.
        conn: Index connection for page-level verification.

    Returns:
        Answer; abstained answers carry no citations.
    """
    abstained = not payload.get("answerable", False)
    citations = (
        []
        if abstained
        else [
            Citation(number=i, source=int(c.get("source", 0)), quote=str(c.get("quote", "")))
            for i, c in enumerate(payload.get("citations", []), start=1)
        ]
    )
    for citation in citations:
        verify_citation(citation, passages, conn)
    return Answer(question, str(payload.get("answer", "")).strip(), abstained, citations, list(passages))


class Answerer:
    """Retrieval + generation + verification for one index."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        model: str,
        conn: sqlite3.Connection | None = None,
        effort: str = "low",
    ) -> None:
        """Wire the pipeline.

        Args:
            retriever: Passage retriever.
            llm: JSON LLM client.
            model: Claude model id for answers.
            conn: Index connection for page-level citation checks.
            effort: output_config.effort for answer calls.
        """
        self.retriever, self.llm, self.model, self.conn, self.effort = retriever, llm, model, conn, effort
        self.system = load_prompt("answer_system.md")

    def answer(self, question: str) -> Answer:
        """Answer a question.

        Args:
            question: User question.

        Returns:
            Answer with verified citations, or an abstention.
        """
        passages = self.retriever.retrieve(question)
        if not passages:
            return Answer(question, "The documents provided do not cover this.", True, [], [], self.model)
        try:
            result = self.llm.complete_json(
                model=self.model,
                system=self.system,
                user=build_user_message(question, passages),
                schema=ANSWER_SCHEMA,
                max_tokens=8000,
                effort=self.effort,
            )
        except LLMRefusal:
            return Answer(question, "The model declined to answer this question.", True, [], passages, self.model)
        answer = parse_answer(question, result.data, passages, self.conn)
        answer.model, answer.usage = result.model, result.usage
        return answer


def dangling_markers(answer: Answer) -> list[int]:
    """Return marker numbers used in the answer text that have no matching citation."""
    known = {c.number for c in answer.citations}
    return sorted({int(n) for n in MARKER.findall(answer.text)} - known)
