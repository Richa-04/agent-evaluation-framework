"""Phase 2 tests: deterministic check evaluation against constructed trajectories."""

from __future__ import annotations

from agent_eval.models import (
    CheckType,
    DeterministicCheck,
    TerminalState,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)
from agent_eval.suite import evaluate_check


def _traj(*, final_answer, terminal=TerminalState.COMPLETED, calls=None, steps=None):
    tool_calls = calls or []
    step_list = steps if steps is not None else [
        TrajectoryStep(index=0, tool_calls=tool_calls, stop_reason="tool_use"),
        TrajectoryStep(index=1, text=final_answer, stop_reason="end_turn"),
    ]
    return Trajectory(
        task_id="t",
        steps=step_list,
        final_answer=final_answer,
        terminal_state=terminal,
        stop_reason="end_turn",
    )


def _call(name, args, content="ok"):
    return ToolCall(id="c", name=name, arguments=args, result=ToolResult(content=content))


def test_tool_invoked_pass_and_fail() -> None:
    traj = _traj(final_answer="done", calls=[_call("calculator", {"expression": "1+1"})])
    chk = DeterministicCheck(id="x", type=CheckType.TOOL_INVOKED, description="calc used", tool_name="calculator")
    assert evaluate_check(chk, traj).passed is True

    chk2 = DeterministicCheck(id="x", type=CheckType.TOOL_INVOKED, description="search used", tool_name="search")
    assert evaluate_check(chk2, traj).passed is False


def test_tool_not_invoked() -> None:
    traj = _traj(final_answer="done", calls=[_call("calculator", {"expression": "1+1"})])
    chk = DeterministicCheck(id="x", type=CheckType.TOOL_NOT_INVOKED, description="no search", tool_name="search")
    assert evaluate_check(chk, traj).passed is True


def test_tool_arg_equals() -> None:
    traj = _traj(final_answer="done", calls=[_call("calculator", {"expression": "47 * 19"})])
    ok = DeterministicCheck(
        id="x", type=CheckType.TOOL_ARG_EQUALS, description="expr", tool_name="calculator",
        arg_name="expression", expected="47 * 19",
    )
    assert evaluate_check(ok, traj).passed is True
    bad = DeterministicCheck(
        id="x", type=CheckType.TOOL_ARG_EQUALS, description="expr", tool_name="calculator",
        arg_name="expression", expected="1 + 1",
    )
    assert evaluate_check(bad, traj).passed is False


def test_terminal_state() -> None:
    traj = _traj(final_answer="done", terminal=TerminalState.MAX_STEPS)
    ok = DeterministicCheck(id="x", type=CheckType.TERMINAL_STATE, description="s", expected="max_steps")
    assert evaluate_check(ok, traj).passed is True
    bad = DeterministicCheck(id="x", type=CheckType.TERMINAL_STATE, description="s", expected="completed")
    assert evaluate_check(bad, traj).passed is False


def test_answer_contains_case_insensitive_by_default() -> None:
    traj = _traj(final_answer="The answer is PARIS.")
    chk = DeterministicCheck(id="x", type=CheckType.ANSWER_CONTAINS, description="paris", expected="paris")
    assert evaluate_check(chk, traj).passed is True
    cs = DeterministicCheck(
        id="x", type=CheckType.ANSWER_CONTAINS, description="paris", expected="paris", case_sensitive=True
    )
    assert evaluate_check(cs, traj).passed is False


def test_answer_equals_and_regex() -> None:
    traj = _traj(final_answer="  893  ")
    eq = DeterministicCheck(id="x", type=CheckType.ANSWER_EQUALS, description="eq", expected="893")
    assert evaluate_check(eq, traj).passed is True
    rx = DeterministicCheck(id="x", type=CheckType.ANSWER_REGEX, description="rx", expected=r"\d{3}")
    assert evaluate_check(rx, traj).passed is True


def test_answer_check_fails_when_no_answer() -> None:
    traj = _traj(final_answer=None, terminal=TerminalState.NO_ANSWER)
    chk = DeterministicCheck(id="x", type=CheckType.ANSWER_CONTAINS, description="c", expected="anything")
    res = evaluate_check(chk, traj)
    assert res.passed is False
    assert "no final answer" in res.detail


def test_max_steps_under() -> None:
    traj = _traj(final_answer="done")  # 2 steps
    ok = DeterministicCheck(id="x", type=CheckType.MAX_STEPS_UNDER, description="<=3", expected=3)
    assert evaluate_check(ok, traj).passed is True
    bad = DeterministicCheck(id="x", type=CheckType.MAX_STEPS_UNDER, description="<=1", expected=1)
    assert evaluate_check(bad, traj).passed is False
