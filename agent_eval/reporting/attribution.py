"""Failure attribution: classify WHY a failed task failed, at the step level.

A task "failed" when its deterministic checks did not all pass (the ground truth).
We then inspect the trajectory (tool calls with arguments + results, terminal
state) together with the failed checks to attribute a cause:

  - WRONG_TOOL            — the agent invoked a tool that wasn't provided for the task
  - MALFORMED_ARGS        — a malformed tool call, an errored tool result, or a
                            wrong-argument check failure
  - PREMATURE_TERMINATION — a required tool was never invoked before the agent
                            stopped, or it ran out of steps / produced no answer
  - INCORRECT_RESULT      — the right tools ran but the final answer was wrong
  - AGENT_ERROR           — the run ended in an error/refusal terminal state
  - OTHER                 — failed checks with no clearer attribution

The primary category is chosen by priority (most fundamental cause first).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ..models import CheckType, RunResult, Task, TerminalState


class FailureCategory(str, Enum):
    NONE = "none"
    WRONG_TOOL = "wrong_tool"
    MALFORMED_ARGS = "malformed_arguments"
    PREMATURE_TERMINATION = "premature_termination"
    INCORRECT_RESULT = "incorrect_result"
    AGENT_ERROR = "agent_error"
    OTHER = "other"


# Most fundamental cause first.
_PRIORITY = [
    FailureCategory.AGENT_ERROR,
    FailureCategory.MALFORMED_ARGS,
    FailureCategory.WRONG_TOOL,
    FailureCategory.PREMATURE_TERMINATION,
    FailureCategory.INCORRECT_RESULT,
    FailureCategory.OTHER,
]

_ANSWER_CHECKS = {CheckType.ANSWER_CONTAINS, CheckType.ANSWER_EQUALS, CheckType.ANSWER_REGEX}


class FailureFinding(BaseModel):
    category: FailureCategory
    step_index: int | None = None
    detail: str


class TaskFailure(BaseModel):
    task_id: str
    failed: bool
    primary_category: FailureCategory
    findings: list[FailureFinding] = []


def attribute_failure(run_result: RunResult, task: Task) -> TaskFailure:
    """Attribute a cause to a (possibly failed) task run."""
    traj = run_result.trajectory
    if run_result.deterministic_passed:
        return TaskFailure(task_id=run_result.task_id, failed=False, primary_category=FailureCategory.NONE)

    checks_by_id = {c.id: c for c in task.deterministic_checks}
    failed = [c for c in run_result.deterministic_results if not c.passed]
    tool_names = traj.tool_names()
    available = set(task.available_tools)
    last_step = traj.step_count - 1 if traj.steps else None

    findings: list[FailureFinding] = []

    # AGENT_ERROR — terminal error / refusal.
    if traj.terminal_state in (TerminalState.ERROR, TerminalState.REFUSAL):
        findings.append(
            FailureFinding(
                category=FailureCategory.AGENT_ERROR,
                step_index=last_step,
                detail=f"run ended in '{traj.terminal_state.value}'"
                + (f": {traj.error}" if traj.error else ""),
            )
        )

    # MALFORMED_ARGS — malformed calls or errored tool results.
    for step in traj.steps:
        for call in step.tool_calls:
            if call.malformed:
                findings.append(
                    FailureFinding(
                        category=FailureCategory.MALFORMED_ARGS,
                        step_index=step.index,
                        detail=f"malformed call to '{call.name}' with arguments={call.arguments}",
                    )
                )
            elif call.result is not None and call.result.is_error:
                findings.append(
                    FailureFinding(
                        category=FailureCategory.MALFORMED_ARGS,
                        step_index=step.index,
                        detail=f"tool '{call.name}' returned an error: {call.result.content}",
                    )
                )
    # MALFORMED_ARGS — a wrong-argument check failed.
    for c in failed:
        orig = checks_by_id.get(c.check_id)
        if orig is not None and orig.type is CheckType.TOOL_ARG_EQUALS:
            findings.append(
                FailureFinding(
                    category=FailureCategory.MALFORMED_ARGS,
                    step_index=None,
                    detail=f"wrong tool argument — {c.detail}",
                )
            )

    # WRONG_TOOL — invoked a tool that wasn't provided for this task.
    unexpected = sorted({n for n in tool_names if n not in available})
    if unexpected:
        findings.append(
            FailureFinding(
                category=FailureCategory.WRONG_TOOL,
                step_index=None,
                detail=f"invoked unexpected tool(s) {unexpected}; available={sorted(available)}",
            )
        )

    # PREMATURE_TERMINATION — a required tool was never invoked before stopping.
    missing_required = sorted(
        {
            orig.tool_name
            for c in failed
            if (orig := checks_by_id.get(c.check_id)) is not None
            and orig.type is CheckType.TOOL_INVOKED
            and orig.tool_name not in tool_names
        }
    )
    if missing_required:
        findings.append(
            FailureFinding(
                category=FailureCategory.PREMATURE_TERMINATION,
                step_index=last_step,
                detail=(
                    f"required tool(s) {missing_required} never invoked before the agent "
                    f"stopped (terminal_state={traj.terminal_state.value})"
                ),
            )
        )
    if traj.terminal_state in (TerminalState.MAX_STEPS, TerminalState.NO_ANSWER):
        findings.append(
            FailureFinding(
                category=FailureCategory.PREMATURE_TERMINATION,
                step_index=last_step,
                detail=f"agent ended in '{traj.terminal_state.value}' without a complete answer",
            )
        )

    # INCORRECT_RESULT — completed with the right tools, but the answer was wrong.
    answer_failed = [
        c for c in failed if (orig := checks_by_id.get(c.check_id)) is not None and orig.type in _ANSWER_CHECKS
    ]
    if (
        answer_failed
        and not missing_required
        and not unexpected
        and traj.terminal_state is TerminalState.COMPLETED
    ):
        detail = "; ".join(c.detail or c.description for c in answer_failed)
        findings.append(
            FailureFinding(
                category=FailureCategory.INCORRECT_RESULT,
                step_index=last_step,
                detail=f"completed with appropriate tools but the answer failed checks: {detail}",
            )
        )

    if not findings:
        names = ", ".join(c.check_id for c in failed)
        findings.append(
            FailureFinding(
                category=FailureCategory.OTHER,
                step_index=last_step,
                detail=f"failed deterministic checks ({names}) with no clearer attribution",
            )
        )

    present = {f.category for f in findings}
    primary = next(cat for cat in _PRIORITY if cat in present)
    return TaskFailure(
        task_id=run_result.task_id, failed=True, primary_category=primary, findings=findings
    )


def attribute_failures(
    run_results: list[RunResult], tasks_by_id: dict[str, Task]
) -> list[TaskFailure]:
    """Attribute causes across a list of run results."""
    return [attribute_failure(r, tasks_by_id[r.task_id]) for r in run_results]
