"""Tests for the Claude client wrappers, with the SDK and the CLI mocked."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ragcite.llm import (
    FALLBACK_BETA,
    AnthropicJSONClient,
    ClaudeCodeCLIClient,
    LLMRefusal,
    make_llm_client,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}


class FakeMessages:
    """Captures create() kwargs and returns a canned response."""

    def __init__(self, response):
        """Store the canned response."""
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        """Record the request."""
        self.kwargs = kwargs
        return self.response


def _response(text='{"a": "ok"}', stop_reason="end_turn"):
    """An SDK-like Message object."""
    usage = SimpleNamespace(input_tokens=12, output_tokens=3, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=usage,
        model="served-model",
    )


def _client(response):
    """A fake anthropic.Anthropic with both the stable and beta messages namespaces."""
    stable, beta = FakeMessages(response), FakeMessages(response)
    return SimpleNamespace(messages=stable, beta=SimpleNamespace(messages=beta)), stable, beta


def test_haiku_request_uses_structured_output_on_stable_endpoint():
    """Haiku request uses structured output on stable endpoint."""
    # Arrange
    fake, stable, beta = _client(_response())
    # Act
    result = AnthropicJSONClient(fake).complete_json(
        model="claude-haiku-5-5", system="sys", user="hi", schema=SCHEMA, effort="low"
    )
    # Assert
    assert result.data == {"a": "ok"} and result.usage["input_tokens"] == 12
    assert stable.kwargs["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert stable.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert beta.kwargs is None


def test_sonnet_request_opts_into_server_side_fallbacks():
    """Sonnet request opts into server side fallbacks."""
    # Arrange
    fake, stable, beta = _client(_response())
    # Act
    AnthropicJSONClient(fake).complete_json(model="claude-sonnet-5-5", system="s", user="u", schema=SCHEMA)
    # Assert
    assert beta.kwargs["betas"] == [FALLBACK_BETA] and beta.kwargs["fallbacks"] == "default"
    assert stable.kwargs is None


def test_refusal_raises():
    """Refusal raises."""
    # Arrange
    fake, _, _ = _client(_response(stop_reason="refusal"))
    # Act / Assert
    with pytest.raises(LLMRefusal):
        AnthropicJSONClient(fake).complete_json(model="claude-haiku-5-5", system="s", user="u", schema=SCHEMA)


def test_truncated_reply_raises():
    """Truncated reply raises."""
    # Arrange
    fake, _, _ = _client(_response(text='{"a": "o', stop_reason="max_tokens"))
    # Act / Assert
    with pytest.raises(ValueError):
        AnthropicJSONClient(fake).complete_json(model="claude-haiku-5-5", system="s", user="u", schema=SCHEMA)


def test_cli_client_sends_prompt_on_stdin_without_api_key(monkeypatch):
    """Cli client sends prompt on stdin without api key."""
    # Arrange
    seen = {}

    def fake_run(argv, **kwargs):
        """Helper: fake run."""
        seen.update(argv=argv, **kwargs)
        out = {
            "structured_output": {"a": "ok"},
            "is_error": False,
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "output_tokens": 5},
        }
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Act
    result = ClaudeCodeCLIClient().complete_json(
        model="claude-sonnet-5-5", system="SYS", user="question", schema=SCHEMA, effort="low"
    )
    # Assert
    assert result.data == {"a": "ok"} and result.usage == {"input_tokens": 102, "output_tokens": 5}
    assert seen["input"] == "question" and "ANTHROPIC_API_KEY" not in seen["env"]
    argv = seen["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "SYS" and argv[argv.index("--tools") + 1] == ""
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA


def test_cli_client_raises_on_error(monkeypatch):
    """Cli client raises on error."""

    # Arrange
    def fake_run(argv, **kwargs):
        """Helper: fake run."""
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps({"is_error": True, "result": "boom"}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Act / Assert
    with pytest.raises(RuntimeError, match="boom"):
        ClaudeCodeCLIClient().complete_json(model="m", system="s", user="u", schema=SCHEMA)


def test_make_llm_client_rejects_unknown_backend():
    """Make llm client rejects unknown backend."""
    # Arrange / Act / Assert
    assert isinstance(make_llm_client("claude-cli"), ClaudeCodeCLIClient)
    with pytest.raises(ValueError):
        make_llm_client("openai")


def test_api_client_reads_base_url_and_key_from_named_env_var(monkeypatch):
    """Api client reads base url and key from named env var."""
    # Arrange
    monkeypatch.setenv("MY_GATEWAY_KEY", "placeholder-not-a-key")
    # Act
    client = make_llm_client("api", base_url="https://gateway.example/api", api_key_env="MY_GATEWAY_KEY")
    # Assert
    assert str(client.client.base_url).startswith("https://gateway.example/api")
    assert client.client.api_key == "placeholder-not-a-key"
