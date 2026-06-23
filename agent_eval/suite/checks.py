"""Deterministic check evaluation.

Each ``DeterministicCheck`` is evaluated against a completed ``Trajectory`` to
produce a ``CheckResult``. These programmatic assertions exist to reduce judge
variance — they are the ground truth the LLM judge is layered on top of.

Interpretation per CheckType:
  TOOL_INVOKED      tool_name was called at least once
  TOOL_NOT_INVOKED  tool_name was never called
  TOOL_ARG_EQUALS   some call to tool_name passed arg_name == expected
  TERMINAL_STATE    trajectory.terminal_state == expected
  ANSWER_CONTAINS   final_answer contains expected (substring)
  ANSWER_EQUALS     final_answer == expected (trimmed)
  ANSWER_REGEX      re.search(expected, final_answer)
  MAX_STEPS_UNDER   step_count <= expected
"""

from __future__ import annotations

import re

from ..models import CheckResult, CheckType, DeterministicCheck, Task, Trajectory


def _result(check: DeterministicCheck, passed: bool, detail: str) -> CheckResult:
    return CheckResult(
        check_id=check.id,
        type=check.type,
        description=check.description,
        passed=passed,
        detail=detail,
    )


def evaluate_check(check: DeterministicCheck, trajectory: Trajectory) -> CheckResult:
    """Evaluate a single deterministic check against a trajectory."""
    names = trajectory.tool_names()

    if check.type is CheckType.TOOL_INVOKED:
        ok = check.tool_name in names
        return _result(check, ok, f"tool {check.tool_name!r} invoked={ok}; calls={names}")

    if check.type is CheckType.TOOL_NOT_INVOKED:
        ok = check.tool_name not in names
        return _result(check, ok, f"tool {check.tool_name!r} invoked={not ok}; calls={names}")

    if check.type is CheckType.TOOL_ARG_EQUALS:
        matches = [tc for tc in trajectory.tool_calls() if tc.name == check.tool_name]
        ok = any(
            str(tc.arguments.get(check.arg_name)) == str(check.expected) for tc in matches
        )
        observed = [tc.arguments.get(check.arg_name) for tc in matches]
        return _result(
            check,
            ok,
            f"{check.tool_name}.{check.arg_name} == {check.expected!r}? {ok}; observed={observed}",
        )

    if check.type is CheckType.TERMINAL_STATE:
        actual = trajectory.terminal_state.value
        ok = actual == str(check.expected)
        return _result(check, ok, f"terminal_state={actual}, expected={check.expected}")

    if check.type is CheckType.MAX_STEPS_UNDER:
        limit = int(check.expected)
        ok = trajectory.step_count <= limit
        return _result(check, ok, f"steps={trajectory.step_count}, limit={limit}")

    # Answer-based checks share the no-answer guard.
    if check.type in (CheckType.ANSWER_CONTAINS, CheckType.ANSWER_EQUALS, CheckType.ANSWER_REGEX):
        answer = trajectory.final_answer
        if answer is None:
            return _result(check, False, "no final answer produced")

        if check.type is CheckType.ANSWER_CONTAINS:
            haystack = answer if check.case_sensitive else answer.lower()
            needle = str(check.expected) if check.case_sensitive else str(check.expected).lower()
            ok = needle in haystack
            return _result(check, ok, f"answer contains {check.expected!r}? {ok}")

        if check.type is CheckType.ANSWER_EQUALS:
            a = answer.strip()
            b = str(check.expected).strip()
            if not check.case_sensitive:
                a, b = a.lower(), b.lower()
            ok = a == b
            return _result(check, ok, f"answer == {check.expected!r}? {ok}")

        # ANSWER_REGEX
        flags = 0 if check.case_sensitive else re.IGNORECASE
        ok = re.search(str(check.expected), answer, flags) is not None
        return _result(check, ok, f"answer matches /{check.expected}/? {ok}")

    return _result(check, False, f"unknown check type: {check.type}")


def evaluate_checks(task: Task, trajectory: Trajectory) -> list[CheckResult]:
    """Evaluate all of a task's deterministic checks against a trajectory."""
    return [evaluate_check(check, trajectory) for check in task.deterministic_checks]
