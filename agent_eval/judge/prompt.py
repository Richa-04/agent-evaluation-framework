"""Judge prompt construction.

The judge prompt deliberately includes (1) the task goal and expected outcome,
(2) the FULL trajectory with each tool call's arguments and result, and (3) the
explicit scoring rubric. The judge is given the trajectory only — not the
deterministic check results — so it is an independent second signal that
supplements, never replaces, the deterministic checks.
"""

from __future__ import annotations

from ..models import Task, Trajectory

JUDGE_VERSION = "v1"

JUDGE_SYSTEM = """\
You are a rigorous evaluator of AI AGENTS (multi-step, tool-using), not of single \
responses. You are given a task, the agent's full execution trajectory (including \
every tool call with its arguments and result), and a rubric. Score the agent's \
performance on each rubric dimension.

Rubric — score each dimension on an integer scale of 1 to 5:

1. goal_completion — Did the agent actually accomplish the task and produce the \
correct expected outcome?
   5 = fully correct outcome; 3 = partially correct or imprecise; 1 = wrong, \
missing, or no usable answer.

2. tool_selection — Did the agent choose the RIGHT tools and call them correctly \
(appropriate tool for each sub-step, with correct arguments)? Penalize skipping a \
tool the task required, using the wrong tool, or malformed/incorrect arguments.
   5 = ideal tools, correct arguments; 3 = mostly right with a notable misstep; \
1 = wrong tools or required tools never invoked.

3. efficiency — Was the trajectory efficient (no redundant calls, no needless \
loops, no wasted steps), reaching the answer directly?
   5 = minimal, direct path; 3 = some avoidable steps; 1 = very wasteful or \
hit the step cap.

Judge ONLY on the evidence in the trajectory. Be strict: an agent that produced a \
plausible final answer but skipped a required tool or used wrong arguments must \
score low on tool_selection even if it reached a completed state. Provide a concise \
rationale for each dimension and a brief overall rationale.\
"""


def render_trajectory(trajectory: Trajectory) -> str:
    """Render a trajectory for the judge, preserving tool arguments and results."""
    lines: list[str] = []
    lines.append(
        f"terminal_state={trajectory.terminal_state.value} | "
        f"stop_reason={trajectory.stop_reason} | steps={trajectory.step_count}"
    )
    if trajectory.error:
        lines.append(f"error: {trajectory.error}")

    for step in trajectory.steps:
        lines.append(f"\nStep {step.index} (stop_reason={step.stop_reason}):")
        if step.thinking:
            lines.append(f"  reasoning: {step.thinking}")
        if step.text:
            lines.append(f"  assistant_text: {step.text}")
        for call in step.tool_calls:
            flag = " [MALFORMED]" if call.malformed else ""
            lines.append(f"  tool_call{flag}: {call.name} arguments={call.arguments}")
            if call.result is not None:
                status = "ERROR" if call.result.is_error else "ok"
                lines.append(f"    result[{status}]: {call.result.content}")
        if not step.tool_calls and not step.text and not step.thinking:
            lines.append("  (no content)")

    lines.append(f"\nFINAL ANSWER: {trajectory.final_answer!r}")
    return "\n".join(lines)


def build_judge_prompt(task: Task, trajectory: Trajectory) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for judging this trajectory."""
    expected = task.expected_outcome if task.expected_outcome is not None else "(not specified)"
    available = ", ".join(task.available_tools) or "(none)"

    user = f"""\
TASK
====
Task id: {task.id}
Goal (prompt given to the agent):
{task.prompt}

Expected outcome (ground truth): {expected}
Tools available to the agent: {available}
Step cap (max_steps): {task.max_steps}

AGENT TRAJECTORY
================
{render_trajectory(trajectory)}

Score this agent on goal_completion, tool_selection, and efficiency (each 1-5) \
with a rationale per dimension and a brief overall rationale."""

    return JUDGE_SYSTEM, user
