"""Tests for eval metrics and the eval loop (LLM mocked)."""

from __future__ import annotations

import json

import pytest

from ragcite.evaluation import (
    EvalQuestion,
    QuestionResult,
    evaluate_question,
    hit_rank,
    load_questions,
    retrieval_metrics,
    summarize,
)
from ragcite.models import Answer, Chunk, Citation, RetrievedChunk
from tests.conftest import FakeLLM

ANSWERABLE = EvalQuestion("q1", "How much leave?", True, "hb", ("28 days of paid",), "28 days.")
UNANSWERABLE = EvalQuestion("q2", "What is the wifi password?", False)


def _passages(*texts_and_docs):
    """Passages from (doc_id, text) pairs."""
    return [RetrievedChunk(Chunk(f"{d}:{i}", d, i, None, "S", t), d, 1.0) for i, (d, t) in enumerate(texts_and_docs)]


def _answer(abstained=False, verified=(True,), passages=()):
    """An Answer with the given citation verification flags."""
    cites = [] if abstained else [Citation(i + 1, 1, "quote text here", verified=v) for i, v in enumerate(verified)]
    return Answer("q", "text [1]", abstained, cites, list(passages), "m", {"input_tokens": 10, "output_tokens": 2})


def test_hit_rank_needs_gold_doc_and_evidence():
    """Hit rank needs gold doc and evidence."""
    # Arrange
    passages = _passages(("other", "28 days of paid leave"), ("hb", "nothing"), ("hb", "Staff: 28 days of paid leave"))
    # Act / Assert
    assert hit_rank(ANSWERABLE, passages) == 3
    assert hit_rank(ANSWERABLE, passages[:2]) is None


def test_retrieval_metrics_hit_at_k_and_mrr():
    """Retrieval metrics hit at k and mrr."""
    # Arrange
    ranks = [1, 3, None, 2]
    # Act
    metrics = retrieval_metrics(ranks, ks=(1, 3))
    # Assert
    assert metrics["hit@1"] == 0.25 and metrics["hit@3"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)


def test_abstaining_on_answerable_question_is_incorrect_without_judge_call():
    """Abstaining on answerable question is incorrect without judge call."""
    # Arrange
    judge = FakeLLM({"verdict": "correct", "reasoning": "r"})
    # Act
    result = evaluate_question(ANSWERABLE, lambda q: _answer(abstained=True), judge, "judge-model")
    # Assert
    assert result.verdict == "incorrect" and judge.calls == []


def test_answered_question_is_sent_to_judge_with_reference():
    """Answered question is sent to judge with reference."""
    # Arrange
    judge = FakeLLM({"verdict": "partially_correct", "reasoning": "missed a fact"})
    passages = _passages(("hb", "Staff get 28 days of paid leave"))
    # Act
    result = evaluate_question(ANSWERABLE, lambda q: _answer(passages=passages), judge, "judge-model")
    # Assert
    assert result.verdict == "partially_correct" and result.hit_rank == 1
    assert "<reference_answer>28 days.</reference_answer>" in judge.calls[0]["user"]
    assert judge.calls[0]["model"] == "judge-model"


def test_summarize_computes_all_headline_rates():
    """Summarize computes all headline rates."""
    # Arrange
    results = [
        QuestionResult(ANSWERABLE, _answer(verified=(True, False)).to_dict(), 1, "correct"),
        QuestionResult(ANSWERABLE, _answer(verified=(True,)).to_dict(), 2, "partially_correct"),
        QuestionResult(ANSWERABLE, _answer(abstained=True).to_dict(), None, "incorrect"),
        QuestionResult(UNANSWERABLE, _answer(abstained=True).to_dict(), None),
        QuestionResult(UNANSWERABLE, _answer().to_dict(), None),
    ]
    # Act
    s = summarize(results)
    # Assert
    assert s.correctness == pytest.approx(0.5)
    assert (s.citations, s.citations_verified) == (4, 3) and s.citation_faithfulness == 0.75
    assert s.answers_fully_grounded == pytest.approx(2 / 3)
    assert s.abstention_accuracy == 0.5 and s.false_abstention_rate == pytest.approx(1 / 3)
    assert s.input_tokens == 50


def test_load_questions_reads_jsonl(tmp_path):
    """Load questions reads jsonl."""
    # Arrange
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(
            {"id": "a", "question": "Q?", "answerable": True, "doc_id": "d", "evidence": ["x"], "reference": "r"}
        )
        + "\n\n"
    )
    # Act
    items = load_questions(path)
    # Assert
    assert items == [EvalQuestion("a", "Q?", True, "d", ("x",), "r")]
