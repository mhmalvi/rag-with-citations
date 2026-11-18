"""Evaluation: retrieval hit@k, citation faithfulness, LLM-judged correctness, abstention."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragcite.answer import Retriever, load_prompt
from ragcite.faithfulness import normalize
from ragcite.llm import LLMClient
from ragcite.models import Answer, RetrievedChunk

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}
VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
DEFAULT_KS = (1, 3, 5)


@dataclass(frozen=True)
class EvalQuestion:
    """One eval item.

    Attributes:
        id: Stable id.
        question: Question text.
        answerable: False for questions the corpus cannot answer.
        doc_id: Gold document (answerable only).
        evidence: Short exact phrases; a retrieved chunk of doc_id containing any of them is a hit.
        reference: Reference answer for the judge.
        category: Free label, e.g. "fact", "list", "unanswerable".
    """

    id: str
    question: str
    answerable: bool
    doc_id: str = ""
    evidence: tuple[str, ...] = ()
    reference: str = ""
    category: str = ""


@dataclass
class QuestionResult:
    """Everything measured for one question."""

    item: EvalQuestion
    answer: dict[str, Any]
    hit_rank: int | None
    verdict: str = ""
    judge_reasoning: str = ""
    latency_s: float = 0.0


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a JSONL question set.

    Args:
        path: JSONL file, one object per line.

    Returns:
        Questions in file order.
    """
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            raw["evidence"] = tuple(raw.get("evidence", ()))
            items.append(EvalQuestion(**raw))
    return items


def hit_rank(item: EvalQuestion, passages: Sequence[RetrievedChunk]) -> int | None:
    """Return the 1-based rank of the first passage that contains gold evidence, or None.

    Args:
        item: Answerable question with doc_id and evidence.
        passages: Retrieved passages, best first.

    Returns:
        Rank of the first hit, None if no passage hits.
    """
    phrases = [normalize(e) for e in item.evidence]
    for rank, passage in enumerate(passages, start=1):
        if passage.chunk.doc_id == item.doc_id and any(p in normalize(passage.chunk.text) for p in phrases):
            return rank
    return None


def retrieval_metrics(ranks: Sequence[int | None], ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    """Compute hit@k for each k and mean reciprocal rank.

    Args:
        ranks: First-hit rank per question (None = miss).
        ks: Cut-offs.

    Returns:
        {"hit@1": ..., "hit@3": ..., "mrr": ..., "n": ...}.
    """
    n = len(ranks)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0 for k in ks}
    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    metrics["n"] = n
    return metrics


def evaluate_retrievers(
    items: Sequence[EvalQuestion], retrievers: dict[str, Retriever], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, dict[str, float]]:
    """Run retrieval-only evaluation (no LLM) for several retriever variants.

    Args:
        items: Question set (unanswerable items are skipped).
        retrievers: Variant name -> retriever returning at least max(ks) passages.
        ks: Cut-offs.

    Returns:
        Variant name -> retrieval metrics.
    """
    answerable = [q for q in items if q.answerable]
    return {
        name: retrieval_metrics([hit_rank(q, r.retrieve(q.question)) for q in answerable], ks)
        for name, r in retrievers.items()
    }


def judge_answer(llm: LLMClient, model: str, item: EvalQuestion, answer_text: str) -> tuple[str, str]:
    """Ask the LLM judge (prompt: src/ragcite/prompts/judge.md) to grade one answer.

    Args:
        llm: JSON LLM client.
        model: Judge model id.
        item: Question with reference answer.
        answer_text: Candidate answer.

    Returns:
        (verdict, reasoning).
    """
    user = (
        f"<question>{item.question}</question>\n<reference_answer>{item.reference}</reference_answer>\n"
        f"<candidate_answer>{answer_text}</candidate_answer>"
    )
    result = llm.complete_json(
        model=model, system=load_prompt("judge.md"), user=user, schema=JUDGE_SCHEMA, max_tokens=4000, effort="low"
    )
    return str(result.data["verdict"]), str(result.data["reasoning"])


def evaluate_question(
    item: EvalQuestion, answer_fn: Callable[[str], Answer], judge: LLMClient | None, judge_model: str
) -> QuestionResult:
    """Answer one question and score it.

    Args:
        item: Eval question.
        answer_fn: Function question -> Answer (the full pipeline).
        judge: LLM client for the judge (None skips judging).
        judge_model: Judge model id.

    Returns:
        QuestionResult. Answerable questions the system abstained on are scored "incorrect"
        without calling the judge.
    """
    started = time.perf_counter()
    answer = answer_fn(item.question)
    latency = time.perf_counter() - started
    result = QuestionResult(
        item,
        answer.to_dict(),
        hit_rank(item, answer.passages) if item.answerable else None,
        latency_s=round(latency, 2),
    )
    if item.answerable and answer.abstained:
        result.verdict, result.judge_reasoning = "incorrect", "System abstained on an answerable question."
    elif item.answerable and judge is not None:
        result.verdict, result.judge_reasoning = judge_answer(judge, judge_model, item, answer.text)
    return result


@dataclass
class Summary:
    """Headline numbers of an answer-level eval run."""

    questions: int = 0
    answerable: int = 0
    unanswerable: int = 0
    correctness: float = 0.0
    verdicts: dict[str, int] = field(default_factory=dict)
    citations: int = 0
    citations_verified: int = 0
    citation_faithfulness: float = 0.0
    answers_fully_grounded: float = 0.0
    abstention_accuracy: float = 0.0
    false_abstention_rate: float = 0.0
    mean_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def summarize(results: Sequence[QuestionResult]) -> Summary:
    """Aggregate per-question results into headline metrics.

    Args:
        results: Output of evaluate_question for every item.

    Returns:
        Summary (rates are fractions in [0, 1]).
    """
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]
    answered = [r for r in results if not r.answer["abstained"]]
    citations = [c for r in answered for c in r.answer["citations"]]
    s = Summary(len(results), len(answerable), len(unanswerable))
    s.verdicts = {v: sum(1 for r in answerable if r.verdict == v) for v in VERDICT_SCORE}
    s.correctness = _mean([VERDICT_SCORE.get(r.verdict, 0.0) for r in answerable])
    s.citations, s.citations_verified = len(citations), sum(1 for c in citations if c["verified"])
    s.citation_faithfulness = s.citations_verified / s.citations if s.citations else 0.0
    s.answers_fully_grounded = _mean([1.0 if r.answer["all_citations_verified"] else 0.0 for r in answered])
    s.abstention_accuracy = _mean([1.0 if r.answer["abstained"] else 0.0 for r in unanswerable])
    s.false_abstention_rate = _mean([1.0 if r.answer["abstained"] else 0.0 for r in answerable])
    s.mean_latency_s = _mean([r.latency_s for r in results])
    s.input_tokens = sum(r.answer["usage"].get("input_tokens", 0) for r in results)
    s.output_tokens = sum(r.answer["usage"].get("output_tokens", 0) for r in results)
    return s


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def results_to_json(summary: Summary, results: Sequence[QuestionResult], meta: dict[str, Any]) -> dict[str, Any]:
    """Bundle a run into one JSON-serialisable object (written next to the reports)."""
    return {
        "meta": meta,
        "summary": asdict(summary),
        "results": [{**asdict(r), "item": asdict(r.item)} for r in results],
    }
