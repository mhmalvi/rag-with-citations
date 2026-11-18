"""Tests for the report renderers, the web API and the CLI (all offline)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragcite import cli
from ragcite.answer import Answerer
from ragcite.config import Settings
from ragcite.evaluation import EvalQuestion, QuestionResult, results_to_json, summarize
from ragcite.models import Answer, Citation
from ragcite.report import render_html, render_markdown
from ragcite.retrieval import HybridRetriever, RetrievalConfig
from ragcite.web.app import create_app
from tests.conftest import FakeEmbedder, FakeLLM, FakeReranker

RETRIEVAL = {"BM25 only": {"hit@1": 0.5, "hit@3": 0.75, "hit@5": 1.0, "mrr": 0.7, "n": 4}}
META = {
    "run_name": "test <run>",
    "timestamp": "2026-11-18 10:00 UTC",
    "answer_model": "a",
    "judge_model": "j",
    "embed_model": "e",
    "rerank_model": "r",
    "final_k": 6,
    "llm_backend": "api",
}


def _run():
    """A one-question run serialised like the real eval."""
    item = EvalQuestion("q1", "Leave <b>?</b>", True, "hb", ("x",), "28 days")
    answer = Answer(
        "q",
        "28 days [1].",
        False,
        [Citation(1, 1, "28 days", "hb:0", "hb", "Handbook", "page 2", True)],
        [],
        "a",
        {"input_tokens": 5, "output_tokens": 1},
    )
    results = [QuestionResult(item, answer.to_dict(), 1, "correct", "matches")]
    return results_to_json(summarize(results), results, META)


def test_markdown_report_has_headlines_retrieval_and_answers():
    """Markdown report has headlines retrieval and answers."""
    # Arrange / Act
    text = render_markdown(_run(), RETRIEVAL)
    # Assert
    assert "| Answer correctness (LLM judge) | 100.0% |" in text
    assert "| BM25 only | 50.0% | 75.0% | 100.0% | 0.700 |" in text
    assert '- [1] Handbook, page 2 (verified): "28 days"' in text


def test_html_report_escapes_user_text():
    """Html report escapes user text."""
    # Arrange / Act
    page = render_html(_run(), RETRIEVAL)
    # Assert
    assert "Leave &lt;b&gt;?&lt;/b&gt;" in page and "<b>?</b>" not in page
    assert "test &lt;run&gt;" in page and 'class="ok">correct' in page


@pytest.fixture
def settings(tmp_path, index) -> Settings:
    """Settings pointing at the fixture index."""
    return Settings(db_path=tmp_path / "index.sqlite")


@pytest.fixture
def client(settings, index) -> TestClient:
    """Web client with a fake LLM behind the real retrieval pipeline."""
    retriever = HybridRetriever(index, FakeEmbedder(), FakeReranker(), RetrievalConfig(final_k=2))
    llm = FakeLLM({"answerable": False, "answer": "The documents do not cover this.", "citations": []})
    return TestClient(create_app(settings, Answerer(retriever, llm, "fake-model", index)))


def test_web_page_and_ask_endpoint(client):
    """Web page and ask endpoint."""
    # Arrange / Act
    page = client.get("/")
    reply = client.post("/api/ask", json={"question": "What is the wifi password?"})
    # Assert
    assert page.status_code == 200 and "Ask the documents" in page.text
    assert reply.status_code == 200
    assert reply.json()["abstained"] is True and len(reply.json()["passages"]) == 2


def test_web_rejects_too_short_or_too_long_questions(client):
    """Web rejects too short or too long questions."""
    # Arrange / Act / Assert
    assert client.post("/api/ask", json={"question": "a"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422


def test_web_serves_original_file_sandboxed_and_404s_unknown(client):
    """Web serves original file sandboxed and 404s unknown."""
    # Arrange / Act
    ok = client.get("/api/documents/security/file")
    missing = client.get("/api/documents/nope/file")
    # Assert
    assert ok.status_code == 200 and ok.headers["content-security-policy"] == "sandbox"
    assert missing.status_code == 404
    assert {d["doc_id"] for d in client.get("/api/documents").json()} == {"handbook", "security"}


def test_web_refuses_paths_outside_the_ingest_root(client, index):
    """Web refuses paths outside the ingest root."""
    # Arrange
    index.execute("UPDATE documents SET source = '../../etc/passwd' WHERE doc_id = 'handbook'")
    index.commit()
    # Act / Assert
    assert client.get("/api/documents/handbook/file").status_code == 403


def test_cli_ingest_and_eval_write_reports(tmp_path, corpus_dir, monkeypatch):
    """Cli ingest and eval write reports."""
    # Arrange
    db = tmp_path / "cli.sqlite"
    monkeypatch.setenv("RAG_DB_PATH", str(db))
    monkeypatch.setattr("ragcite.pipeline.embedder_for", lambda name: FakeEmbedder())
    monkeypatch.setattr("ragcite.pipeline.reranker_for", lambda name: FakeReranker())
    monkeypatch.setenv("RAG_EMBED_MODEL", "fake-embedder")
    judge_and_answer = FakeLLM(
        lambda req: (
            {"verdict": "correct", "reasoning": "ok"}
            if "reference" in req["user"]
            else {"answerable": False, "answer": "Not covered.", "citations": []}
        )
    )
    monkeypatch.setattr("ragcite.llm.make_llm_client", lambda **kwargs: judge_and_answer)
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id": "n1", "question": "What is the wifi password?", "answerable": false}\n')
    # Act
    assert cli.main(["ingest", str(corpus_dir)]) == 0
    code = cli.main(["eval", "--questions", str(questions), "--out", str(tmp_path / "out"), "--name", "t"])
    # Assert
    assert code == 0
    assert "Abstention on unanswerable questions | 100.0%" in (tmp_path / "out" / "report.md").read_text()
    assert Path(tmp_path / "out" / "report.html").exists() and Path(tmp_path / "out" / "results.json").exists()
