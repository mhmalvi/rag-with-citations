"""Thin JSON-returning wrapper around the Claude Messages API.

The answerer and the judge depend only on the LLMClient protocol, so tests run fully
offline with a fake client. AnthropicJSONClient uses structured outputs
(output_config.format with a JSON schema), so the reply is always schema-valid JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Protocol

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5"})


class LLMRefusal(RuntimeError):
    """Raised when the model (and any fallback) declined the request."""


@dataclass
class LLMResult:
    """A parsed JSON reply plus accounting.

    Attributes:
        data: Parsed JSON object.
        model: Model that served the reply.
        usage: Token counters (input_tokens, output_tokens, ...).
    """

    data: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that turns (system, user, schema) into a schema-valid JSON object."""

    def complete_json(
        self, *, model: str, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4000, effort: str = "low"
    ) -> LLMResult:
        """Return the model's JSON reply."""
        ...


def _usage_dict(usage: Any) -> dict[str, int]:
    """Copy the integer counters off an SDK Usage object."""
    names = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return {n: int(getattr(usage, n, 0) or 0) for n in names}


class AnthropicJSONClient:
    """LLMClient backed by the official anthropic SDK."""

    def __init__(self, client: Any | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        """Create the client.

        Args:
            client: An anthropic.Anthropic instance (injected in tests). When None, one is built.
            base_url: Optional Anthropic-compatible endpoint (e.g. a gateway); None means api.anthropic.com
                or ANTHROPIC_BASE_URL.
            api_key: Optional key; None lets the SDK read ANTHROPIC_API_KEY from the environment.
        """
        if client is None:
            import anthropic

            client = anthropic.Anthropic(base_url=base_url or None, api_key=api_key or None, max_retries=4)
        self.client = client

    def _create(self, model: str, **request: Any) -> Any:
        """Send the request, opting into server-side refusal fallbacks where the model supports them."""
        if model in FALLBACK_MODELS:
            return self.client.beta.messages.create(model=model, betas=[FALLBACK_BETA], fallbacks="default", **request)
        return self.client.messages.create(model=model, **request)

    def complete_json(
        self, *, model: str, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4000, effort: str = "low"
    ) -> LLMResult:
        """Call Claude with a JSON-schema output format and parse the reply.

        Args:
            model: Model id, e.g. "claude-sonnet-5-5".
            system: System prompt.
            user: User message.
            schema: JSON schema the reply must satisfy.
            max_tokens: Output cap.
            effort: output_config.effort level.

        Returns:
            LLMResult with the parsed object.

        Raises:
            LLMRefusal: On stop_reason "refusal".
            ValueError: If the reply was cut off or carried no text block.
        """
        response = self._create(
            model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        return parse_response(response, model)


def parse_response(response: Any, model: str) -> LLMResult:
    """Turn an SDK Message produced with a JSON-schema output format into an LLMResult.

    Raises:
        LLMRefusal: On stop_reason "refusal".
        ValueError: If the reply was cut off or carried no text block.
    """
    if response.stop_reason == "refusal":
        raise LLMRefusal(str(getattr(response, "stop_details", "") or "refused"))
    if response.stop_reason == "max_tokens":
        raise ValueError("reply hit max_tokens before the JSON was complete")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ValueError("reply had no text block")
    return LLMResult(json.loads(text), getattr(response, "model", model), _usage_dict(response.usage))


class ClaudeCodeCLIClient:
    """LLMClient that shells out to the Claude Code CLI (`claude -p`) instead of the API.

    For machines that have a signed-in Claude Code but no working API key. The same
    system prompt and JSON schema are sent (`--system-prompt`, `--json-schema`), with
    every built-in tool, MCP server and settings file switched off, from an empty
    working directory so no CLAUDE.md is picked up.
    """

    def __init__(self, executable: str = "claude", timeout_s: float = 300.0) -> None:
        """Configure the CLI call.

        Args:
            executable: Path or name of the `claude` binary.
            timeout_s: Per-call timeout in seconds.
        """
        self.executable, self.timeout_s = executable, timeout_s

    def command(self, model: str, system: str, schema: dict[str, Any], effort: str) -> list[str]:
        """Build the argv for one call (the user message goes to stdin)."""
        return [
            self.executable,
            "-p",
            "--model",
            model,
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(schema),
            "--effort",
            effort,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]

    def complete_json(
        self, *, model: str, system: str, user: str, schema: dict[str, Any], max_tokens: int = 4000, effort: str = "low"
    ) -> LLMResult:
        """Run `claude -p` and return its structured output (max_tokens is managed by the CLI).

        Raises:
            LLMRefusal: When the CLI reports a refusal stop reason.
            RuntimeError: When the CLI fails or returns no structured output.
        """
        payload = self.run(self.command(model, system, schema, effort), user)
        if payload.get("stop_reason") == "refusal":
            raise LLMRefusal("refused")
        if "structured_output" not in payload:
            raise RuntimeError("claude CLI returned no structured output")
        return LLMResult(payload["structured_output"], model, cli_usage(payload.get("usage", {})))

    def run(self, argv: list[str], user: str) -> dict[str, Any]:
        """Run the CLI from an empty temp dir with ANTHROPIC_API_KEY removed (so the sign-in is used).

        Args:
            argv: Command line from command().
            user: User message, sent on stdin.

        Returns:
            The CLI's JSON result object.

        Raises:
            RuntimeError: On a non-zero exit or an error result.
        """
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with tempfile.TemporaryDirectory(prefix="ragcite-cli-") as workdir:
            done = subprocess.run(
                argv,
                input=user,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd=workdir,
                env=env,
                check=False,
            )
        payload = json.loads(done.stdout or "{}")
        if done.returncode != 0 or payload.get("is_error"):
            raise RuntimeError(f"claude CLI failed (exit {done.returncode}): {str(payload.get('result', ''))[:200]}")
        return payload


def cli_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Token counters from the CLI's usage block (cache reads/writes count as input)."""
    names = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {
        "input_tokens": sum(int(usage.get(n, 0) or 0) for n in names),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def make_llm_client(backend: str, base_url: str = "", api_key_env: str = "ANTHROPIC_API_KEY") -> LLMClient:
    """Return the LLM client for a backend name.

    Args:
        backend: "api" (Anthropic SDK, direct or via an Anthropic-compatible gateway) or "claude-cli".
        base_url: Optional base URL for the SDK ("" = Anthropic default).
        api_key_env: Name of the environment variable that holds the key (the value is never logged).

    Raises:
        ValueError: For unknown backends.
    """
    if backend == "api":
        return AnthropicJSONClient(base_url=base_url or None, api_key=os.environ.get(api_key_env) or None)
    if backend == "claude-cli":
        return ClaudeCodeCLIClient()
    raise ValueError(f"unknown RAG_LLM_BACKEND {backend!r} (use 'api' or 'claude-cli')")
