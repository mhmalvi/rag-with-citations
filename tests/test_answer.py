"""Tests for answer generation, citation resolution and verification (LLM mocked)."""

from __future__ import annotations

from ragcite.answer import Answerer, build_user_message, dangling_markers, parse_answer
from ragcite.llm import LLMRefusal
from ragcite.models import Chunk, RetrievedChunk
from ragcite.retrieval import HybridRetriever, RetrievalConfig
from tests.conftest import FakeEmbedder, FakeLLM, FakeReranker


def _passage(text: str, page: int | None = 3) -> RetrievedChunk:
    """A retrieved passage for tests."""
    return RetrievedChunk(Chunk("doc:0000", "doc", 0, page, "1. Intro", text), "Doc title", 1.0)


def test_parse_answer_verifies_exact_quote_and_sets_highlight():
    """Parse answer verifies exact quote and sets highlight."""
    # Arrange
    passages = [_passage("Staff get 28 days of paid leave. Requests go to HR.")]
    payload = {
        "answerable": True,
        "answer": "Staff get 28 days [1].",
        "citations": [{"source": 1, "quote": "Staff get 28 days of paid leave."}],
    }
    # Act
    answer = parse_answer("q", payload, passages, conn=None)
    # Assert
    citation = answer.citations[0]
    assert citation.verified and citation.location == "page 3" and citation.title == "Doc title"
    assert passages[0].chunk.text[slice(*citation.highlight)] == "Staff get 28 days of paid leave."
    assert answer.all_citations_verified


def test_parse_answer_flags_paraphrased_quote_and_bad_source():
    """Parse answer flags paraphrased quote and bad source."""
    # Arrange
    passages = [_passage("Staff get 28 days of paid leave.")]
    payload = {
        "answerable": True,
        "answer": "Thirty days [1] [2].",
        "citations": [
            {"source": 1, "quote": "Staff get 30 days of paid leave."},
            {"source": 7, "quote": "Staff get 28 days of paid leave."},
        ],
    }
    # Act
    answer = parse_answer("q", payload, passages, conn=None)
    # Assert
    assert [c.verified for c in answer.citations] == [False, False]
    assert answer.citations[1].chunk_id == ""
    assert not answer.all_citations_verified


def test_parse_answer_abstention_drops_citations():
    """Parse answer abstention drops citations."""
    # Arrange
    payload = {
        "answerable": False,
        "answer": "The documents do not cover this.",
        "citations": [{"source": 1, "quote": "anything at all here"}],
    }
    # Act
    answer = parse_answer("q", payload, [_passage("x")], conn=None)
    # Assert
    assert answer.abstained and answer.citations == []


def test_quote_must_also_exist_on_the_stored_page(index):
    """Quote must also exist on the stored page."""
    # Arrange: the passage claims a section whose stored text does not contain the quote
    passages = [
        RetrievedChunk(
            Chunk(
                "handbook:9999",
                "handbook",
                9,
                None,
                "Staff handbook > Expenses",
                "Fabricated text that is not on the page.",
            ),
            "Staff handbook",
            1.0,
        )
    ]
    payload = {
        "answerable": True,
        "answer": "x [1]",
        "citations": [{"source": 1, "quote": "Fabricated text that is not on the page."}],
    }
    # Act
    answer = parse_answer("q", payload, passages, conn=index)
    # Assert
    assert answer.citations[0].highlight is not None
    assert answer.citations[0].verified is False


def test_answerer_end_to_end_with_fake_llm(index):
    """Answerer end to end with fake llm."""
    # Arrange
    retriever = HybridRetriever(index, FakeEmbedder(), FakeReranker(), RetrievalConfig(final_k=3))

    def reply(request):
        """Helper: reply."""
        first = request["user"].split('<source id="1"', 1)[1].split("\n", 2)[1]
        return {"answerable": True, "answer": "See [1].", "citations": [{"source": 1, "quote": first[:60]}]}

    llm = FakeLLM(reply)
    answerer = Answerer(retriever, llm, "claude-test", conn=index)
    # Act
    answer = answerer.answer("How many days of paid annual leave do staff get?")
    # Assert
    assert llm.calls[0]["model"] == "claude-test"
    assert "<question>How many days" in llm.calls[0]["user"]
    assert answer.citations[0].verified
    assert answer.usage["input_tokens"] == 100


def test_answerer_turns_refusal_into_abstention(index):
    """Answerer turns refusal into abstention."""

    # Arrange
    class RefusingLLM:
        def complete_json(self, **kwargs):
            """Helper: complete json."""
            raise LLMRefusal("declined")

    retriever = HybridRetriever(index, FakeEmbedder(), None, RetrievalConfig(rerank_k=0))
    # Act
    answer = Answerer(retriever, RefusingLLM(), "m", conn=index).answer("annual leave")
    # Assert
    assert answer.abstained and "declined" in answer.text


def test_build_user_message_numbers_sources_with_location():
    """Build user message numbers sources with location."""
    # Arrange
    passages = [_passage("alpha text"), _passage("beta text", page=None)]
    # Act
    message = build_user_message("Why?", passages)
    # Assert
    assert '<source id="1" document="Doc title" location="page 3">' in message
    assert '<source id="2" document="Doc title" location="1. Intro">' in message
    assert message.endswith("<question>Why?</question>")


def test_dangling_markers_lists_unbacked_numbers():
    """Dangling markers lists unbacked numbers."""
    # Arrange
    payload = {
        "answerable": True,
        "answer": "A [1] and B [3].",
        "citations": [{"source": 1, "quote": "alpha text here"}],
    }
    # Act
    answer = parse_answer("q", payload, [_passage("alpha text here")], conn=None)
    # Assert
    assert dangling_markers(answer) == [3]
