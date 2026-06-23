"""Human-readable rendering of a Trajectory for the console.

Shows every step: reasoning, assistant text, and each tool call with its
arguments and result — i.e. the intermediate state, not just a transcript.
"""

from __future__ import annotations

from .models import ToolCall, Trajectory


def _format_args(arguments: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in arguments.items())


def _format_tool_call(call: ToolCall) -> str:
    head = f"{call.name}({_format_args(call.arguments)})"
    flags = " [MALFORMED]" if call.malformed else ""
    if call.result is None:
        return f"{head}{flags} -> (not executed)"
    status = "ERROR" if call.result.is_error else "ok"
    return f"{head}{flags} -> [{status}] {call.result.content!r}"


def format_trajectory(trajectory: Trajectory) -> str:
    """Return a multi-line string rendering of a full trajectory."""
    lines: list[str] = []
    lines.append(f"=== Trajectory: task '{trajectory.task_id}' ===")
    lines.append(
        f"terminal_state={trajectory.terminal_state.value} | "
        f"stop_reason={trajectory.stop_reason} | "
        f"steps={trajectory.step_count}"
    )
    if trajectory.usage:
        usage = ", ".join(f"{k}={v}" for k, v in trajectory.usage.items())
        lines.append(f"usage: {usage}")
    if trajectory.error:
        lines.append(f"error: {trajectory.error}")
    lines.append(f"final_answer: {trajectory.final_answer!r}")

    for step in trajectory.steps:
        lines.append("")
        lines.append(f"--- Step {step.index} (stop_reason={step.stop_reason}) ---")
        if step.thinking:
            lines.append(f"  thinking: {step.thinking}")
        if step.text:
            lines.append(f"  text: {step.text}")
        for call in step.tool_calls:
            lines.append(f"  tool: {_format_tool_call(call)}")
        if not step.tool_calls and not step.text and not step.thinking:
            lines.append("  (no content)")

    return "\n".join(lines)
