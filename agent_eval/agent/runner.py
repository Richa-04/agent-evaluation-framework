"""The async agent loop.

``run_agent`` drives one task to completion: it calls the model, parses the
response into a TrajectoryStep (capturing reasoning, text, and every tool call
with its ARGUMENTS), executes requested tools, feeds results back, and loops
until the model finishes or a terminal condition is hit.

Hard requirements handled here:
- max-step cap (``task.max_steps``) -> TerminalState.MAX_STEPS
- terminal-state detection (end_turn -> COMPLETED, refusal -> REFUSAL,
  text-but-no-answer -> NO_ANSWER, exception -> ERROR)
- graceful malformed/failed tool calls: unknown tool or non-object input is
  marked ``malformed`` and returned to the model as an error tool_result; a tool
  that raises returns an ``is_error`` result. Neither aborts the run.

The Anthropic client is injected so tests can pass a scripted fake. The client
only needs an awaitable ``client.messages.create(**kwargs)`` returning an object
with ``.content`` (blocks), ``.stop_reason``, and optionally ``.usage``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models import (
    Task,
    TerminalState,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)
from ..models import RunConfig
from ..tools.base import Tool

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, methodical problem-solving agent. Use the provided tools "
    "when they help you reach a correct answer. Think step by step. When you have "
    "the final answer, state it directly in plain text and stop calling tools."
)

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    out: dict[str, int] = {}
    for key in _USAGE_KEYS:
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out or None


def _build_create_kwargs(
    *,
    tool_params: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    config: RunConfig,
    system_prompt: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": system_prompt,
        "messages": messages,
        "tools": tool_params,
    }
    # Only send sampling params to models that accept them (not Opus 5 / Sonnet 5).
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.thinking:
        kwargs["thinking"] = {"type": config.thinking}
    if config.effort:
        kwargs["output_config"] = {"effort": config.effort}
    return kwargs


async def run_agent(
    task: Task,
    tools: list[Tool],
    client: Any,
    config: RunConfig,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Trajectory:
    """Run ``task`` with the agent loop and return the full Trajectory."""
    selected = {t.name: t for t in tools if t.name in set(task.available_tools)}
    tool_params = [selected[name].to_param() for name in task.available_tools if name in selected]

    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    steps: list[TrajectoryStep] = []
    final_answer: str | None = None
    terminal_state = TerminalState.MAX_STEPS  # stays this only if the loop exhausts
    final_stop_reason: str | None = None
    error: str | None = None
    agg_usage: dict[str, int] = {}

    started_at = datetime.now(timezone.utc)

    for index in range(task.max_steps):
        try:
            response = await client.messages.create(
                **_build_create_kwargs(
                    tool_params=tool_params,
                    messages=messages,
                    config=config,
                    system_prompt=system_prompt,
                )
            )
        except Exception as exc:  # network/SDK failure — record and stop cleanly
            error = f"{type(exc).__name__}: {exc}"
            terminal_state = TerminalState.ERROR
            final_stop_reason = "error"
            steps.append(TrajectoryStep(index=index, stop_reason="error"))
            break

        stop_reason = getattr(response, "stop_reason", None)
        step_usage = _extract_usage(response)
        if step_usage:
            for key, value in step_usage.items():
                agg_usage[key] = agg_usage.get(key, 0) + value

        # --- parse content blocks ---
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_use_blocks: list[Any] = []
        for block in getattr(response, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "tool_use":
                tool_use_blocks.append(block)
        text = "\n".join(p for p in text_parts if p) or None
        thinking = "\n".join(p for p in thinking_parts if p) or None

        # --- execute tool calls (capturing arguments + results) ---
        tool_calls: list[ToolCall] = []
        tool_result_blocks: list[dict[str, Any]] = []
        assistant_content: list[dict[str, Any]] = []
        if text:
            assistant_content.append({"type": "text", "text": text})

        for tub in tool_use_blocks:
            call_id = getattr(tub, "id", None) or f"call_{index}_{len(tool_calls)}"
            name = getattr(tub, "name", "") or ""
            raw_input = getattr(tub, "input", None)
            args = raw_input if isinstance(raw_input, dict) else {}
            known = name in selected
            malformed = (not isinstance(raw_input, dict)) or (not known)

            assistant_content.append(
                {"type": "tool_use", "id": call_id, "name": name, "input": args}
            )

            if not known:
                result = ToolResult(
                    content=(
                        f"Error: unknown tool '{name}'. "
                        f"Available tools: {', '.join(selected) or 'none'}."
                    ),
                    is_error=True,
                )
            elif not isinstance(raw_input, dict):
                result = ToolResult(
                    content="Error: tool input was not a valid JSON object.",
                    is_error=True,
                )
            else:
                result = await selected[name].run(args)

            tool_calls.append(
                ToolCall(id=call_id, name=name, arguments=args, result=result, malformed=malformed)
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )

        steps.append(
            TrajectoryStep(
                index=index,
                thinking=thinking,
                text=text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                usage=step_usage,
            )
        )

        # --- decide whether to loop or terminate ---
        if stop_reason == "refusal":
            terminal_state = TerminalState.REFUSAL
            final_stop_reason = stop_reason
            final_answer = text
            break

        if stop_reason == "tool_use" and tool_use_blocks:
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        # end_turn, max_tokens, or anything else without pending tool calls
        final_stop_reason = stop_reason
        final_answer = text
        terminal_state = TerminalState.COMPLETED if text else TerminalState.NO_ANSWER
        break

    ended_at = datetime.now(timezone.utc)
    return Trajectory(
        task_id=task.id,
        steps=steps,
        final_answer=final_answer,
        terminal_state=terminal_state,
        stop_reason=final_stop_reason,
        error=error,
        usage=agg_usage or None,
        started_at=started_at,
        ended_at=ended_at,
    )
