"""Deterministic fakes for mocking the Anthropic client in tests.

The runner only reads attributes off the response (``.content`` blocks with
``.type``/``.text``/``.thinking``/``.name``/``.input``/``.id``, plus
``.stop_reason`` and ``.usage``), so SimpleNamespace stand-ins are enough — no
real SDK objects needed.

``ScriptedClient`` routes each ``messages.create`` call to a per-route queue of
scripted responses, matched by a substring of the first user message. This lets
one client serve a whole multi-task suite deterministically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def make_usage(input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text)


def tool_use_block(call_id: str, name: str, tool_input: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def assistant_text(text: str, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    """A final assistant message with just text."""
    return SimpleNamespace(content=[text_block(text)], stop_reason=stop_reason, usage=make_usage())


def assistant_tool(
    calls: list[tuple[str, str, Any]],
    *,
    text: str | None = None,
    thinking: str | None = None,
    stop_reason: str = "tool_use",
) -> SimpleNamespace:
    """An assistant message that issues one or more tool calls.

    ``calls`` is a list of ``(id, name, input)`` tuples.
    """
    blocks: list[SimpleNamespace] = []
    if thinking is not None:
        blocks.append(thinking_block(thinking))
    if text is not None:
        blocks.append(text_block(text))
    for call_id, name, tool_input in calls:
        blocks.append(tool_use_block(call_id, name, tool_input))
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=make_usage())


def assistant_raw(content: list[Any], *, stop_reason: str, usage: Any = None) -> SimpleNamespace:
    """Escape hatch for unusual responses (e.g. refusal with no content)."""
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage or make_usage())


class _Messages:
    def __init__(self, client: "ScriptedClient") -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return self._client._next(kwargs)


class ScriptedClient:
    """A fake Anthropic client returning scripted responses.

    ``routes`` maps a substring (matched against the first user message text) to
    an ordered list of responses; each ``create`` call pops the next response for
    the matching route. ``calls`` records every kwargs dict for assertions.
    """

    def __init__(self, routes: dict[str, list[Any]]) -> None:
        self.routes = routes
        self._counters: dict[str, int] = {key: 0 for key in routes}
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    @staticmethod
    def _first_user_text(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        user_text = self._first_user_text(kwargs.get("messages") or [])
        for key, scripts in self.routes.items():
            if key in user_text:
                idx = self._counters[key]
                if idx >= len(scripts):
                    raise AssertionError(f"ScriptedClient: no more responses for route {key!r}")
                self._counters[key] = idx + 1
                return scripts[idx]
        raise AssertionError(f"ScriptedClient: no route matched user text {user_text!r}")


def json_message(payload: Any, *, stop_reason: str = "end_turn") -> SimpleNamespace:
    """A response whose single text block is ``json.dumps(payload)`` — mimics a
    structured-output (json_schema) response from the judge."""
    return SimpleNamespace(
        content=[text_block(json.dumps(payload))],
        stop_reason=stop_reason,
        usage=make_usage(),
    )


class OneShotClient:
    """A fake client that returns the same response for every create() call and
    records the kwargs of each call (for asserting on the request)."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, client: "OneShotClient") -> None:
            self._client = client

        async def create(self, **kwargs: Any) -> Any:
            self._client.calls.append(kwargs)
            return self._client._response


class RaisingClient:
    """A fake client whose ``messages.create`` always raises, to test error handling."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("simulated API failure")
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, client: "RaisingClient") -> None:
            self._client = client

        async def create(self, **kwargs: Any) -> Any:
            raise self._client._exc
