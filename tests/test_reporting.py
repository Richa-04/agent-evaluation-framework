"""Phase 5 tests: failure attribution + pandas summary (no LLM)."""

from __future__ import annotations

from agent_eval.models import (
    CheckResult,
    CheckType,
    DeterministicCheck,
    DimensionScore,
    JudgeScore,
    RunConfig,
    RunResult,
    TerminalState,
    Task,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)
from agent_eval.reporting import (
    FailureCategory,
    attribute_failure,
    format_report,
    summarize,
)


def _task(task_id, available, checks):
    return Task(id=task_id, prompt="p", available_tools=available, deterministic_checks=checks)


def _judge(g, t, e):
    return JudgeScore(
        goal_completion=DimensionScore(score=g, rationale="g"),
        tool_selection=DimensionScore(score=t, rationale="t"),
        efficiency=DimensionScore(score=e, rationale="e"),
        overall_rationale="o",
        judge_model="claude-sonnet-4-6",
    )


def _run(task_id, *, steps, terminal, final_answer, check_results, judge=None, error=None):
    return RunResult(
        run_id="r",
        task_id=task_id,
        trajectory=Trajectory(
            task_id=task_id,
            steps=steps,
            final_answer=final_answer,
            terminal_state=terminal,
            stop_reason="end_turn",
            error=error,
        ),
        deterministic_results=check_results,
        judge_score=judge,
        config=RunConfig(model="claude-opus-4-8"),
    )


def test_premature_termination_required_tool_skipped() -> None:
    # Task needs search AND calculator; agent called search only, then stopped.
    task = _task(
        "convert",
        ["search", "calculator"],
        [
            DeterministicCheck(id="s", type=CheckType.TOOL_INVOKED, description="search", tool_name="search"),
            DeterministicCheck(id="c", type=CheckType.TOOL_INVOKED, description="calc", tool_name="calculator"),
        ],
    )
    run = _run(
        "convert",
        steps=[
            TrajectoryStep(index=0, tool_calls=[ToolCall(id="1", name="search", arguments={"query": "x"}, result=ToolResult(content="ok"))], stop_reason="tool_use"),
            TrajectoryStep(index=1, text="done", stop_reason="end_turn"),
        ],
        terminal=TerminalState.COMPLETED,
        final_answer="done",
        check_results=[
            CheckResult(check_id="s", type=CheckType.TOOL_INVOKED, description="search", passed=True),
            CheckResult(check_id="c", type=CheckType.TOOL_INVOKED, description="calc", passed=False),
        ],
    )
    f = attribute_failure(run, task)
    assert f.failed is True
    assert f.primary_category is FailureCategory.PREMATURE_TERMINATION
    assert any("calculator" in finding.detail for finding in f.findings)


def test_malformed_arguments() -> None:
    task = _task("t", ["calculator"], [
        DeterministicCheck(id="a", type=CheckType.ANSWER_CONTAINS, description="ans", expected="4"),
    ])
    run = _run(
        "t",
        steps=[
            TrajectoryStep(
                index=0,
                tool_calls=[ToolCall(id="1", name="calculator", arguments={}, malformed=True,
                                     result=ToolResult(content="Error: bad input", is_error=True))],
                stop_reason="tool_use",
            ),
            TrajectoryStep(index=1, text="x", stop_reason="end_turn"),
        ],
        terminal=TerminalState.COMPLETED,
        final_answer="x",
        check_results=[CheckResult(check_id="a", type=CheckType.ANSWER_CONTAINS, description="ans", passed=False)],
    )
    f = attribute_failure(run, task)
    assert f.primary_category is FailureCategory.MALFORMED_ARGS


def test_wrong_tool_not_in_available() -> None:
    task = _task("t", ["calculator"], [
        DeterministicCheck(id="c", type=CheckType.TOOL_INVOKED, description="calc", tool_name="calculator"),
    ])
    run = _run(
        "t",
        steps=[
            TrajectoryStep(index=0, tool_calls=[ToolCall(id="1", name="search", arguments={"query": "x"}, result=ToolResult(content="ok"))], stop_reason="tool_use"),
            TrajectoryStep(index=1, text="x", stop_reason="end_turn"),
        ],
        terminal=TerminalState.COMPLETED,
        final_answer="x",
        check_results=[CheckResult(check_id="c", type=CheckType.TOOL_INVOKED, description="calc", passed=False)],
    )
    f = attribute_failure(run, task)
    # 'search' isn't in available_tools -> wrong tool (ranks above premature termination).
    assert f.primary_category is FailureCategory.WRONG_TOOL


def test_incorrect_result_tools_fine_answer_wrong() -> None:
    task = _task("t", ["calculator"], [
        DeterministicCheck(id="c", type=CheckType.TOOL_INVOKED, description="calc", tool_name="calculator"),
        DeterministicCheck(id="a", type=CheckType.ANSWER_CONTAINS, description="ans", expected="893"),
    ])
    run = _run(
        "t",
        steps=[
            TrajectoryStep(index=0, tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "47*19"}, result=ToolResult(content="893"))], stop_reason="tool_use"),
            TrajectoryStep(index=1, text="The answer is 8003.", stop_reason="end_turn"),
        ],
        terminal=TerminalState.COMPLETED,
        final_answer="The answer is 8003.",
        check_results=[
            CheckResult(check_id="c", type=CheckType.TOOL_INVOKED, description="calc", passed=True),
            CheckResult(check_id="a", type=CheckType.ANSWER_CONTAINS, description="ans", passed=False),
        ],
    )
    f = attribute_failure(run, task)
    assert f.primary_category is FailureCategory.INCORRECT_RESULT


def test_agent_error_terminal() -> None:
    task = _task("t", ["calculator"], [
        DeterministicCheck(id="d", type=CheckType.TERMINAL_STATE, description="done", expected="completed"),
    ])
    run = _run(
        "t",
        steps=[TrajectoryStep(index=0, stop_reason="error")],
        terminal=TerminalState.ERROR,
        final_answer=None,
        error="ValueError: boom",
        check_results=[CheckResult(check_id="d", type=CheckType.TERMINAL_STATE, description="done", passed=False)],
    )
    f = attribute_failure(run, task)
    assert f.primary_category is FailureCategory.AGENT_ERROR


def test_passing_task_has_no_failure() -> None:
    task = _task("t", ["calculator"], [
        DeterministicCheck(id="a", type=CheckType.ANSWER_CONTAINS, description="ans", expected="4"),
    ])
    run = _run(
        "t",
        steps=[TrajectoryStep(index=0, text="4", stop_reason="end_turn")],
        terminal=TerminalState.COMPLETED,
        final_answer="4",
        check_results=[CheckResult(check_id="a", type=CheckType.ANSWER_CONTAINS, description="ans", passed=True)],
    )
    f = attribute_failure(run, task)
    assert f.failed is False
    assert f.primary_category is FailureCategory.NONE


def test_summarize_pass_rate_dims_and_breakdown() -> None:
    pass_task = _task("ok", ["calculator"], [
        DeterministicCheck(id="a", type=CheckType.ANSWER_CONTAINS, description="ans", expected="4"),
    ])
    fail_task = _task("bad", ["search", "calculator"], [
        DeterministicCheck(id="c", type=CheckType.TOOL_INVOKED, description="calc", tool_name="calculator"),
    ])
    tasks_by_id = {"ok": pass_task, "bad": fail_task}

    ok_run = _run("ok", steps=[TrajectoryStep(index=0, text="4", stop_reason="end_turn")],
                  terminal=TerminalState.COMPLETED, final_answer="4",
                  check_results=[CheckResult(check_id="a", type=CheckType.ANSWER_CONTAINS, description="ans", passed=True)],
                  judge=_judge(5, 5, 5))
    bad_run = _run("bad",
                   steps=[
                       TrajectoryStep(index=0, tool_calls=[ToolCall(id="1", name="search", arguments={"query": "x"}, result=ToolResult(content="ok"))], stop_reason="tool_use"),
                       TrajectoryStep(index=1, text="x", stop_reason="end_turn"),
                   ],
                   terminal=TerminalState.COMPLETED, final_answer="x",
                   check_results=[CheckResult(check_id="c", type=CheckType.TOOL_INVOKED, description="calc", passed=False)],
                   judge=_judge(2, 2, 3))

    results = [ok_run, bad_run]
    rep = summarize(results, tasks_by_id)
    assert rep.total_tasks == 2
    assert rep.passed == 1
    assert rep.pass_rate == 0.5
    assert rep.avg_goal_completion == 3.5  # (5 + 2) / 2
    assert rep.failure_breakdown == {FailureCategory.PREMATURE_TERMINATION.value: 1}

    text = format_report(results, tasks_by_id)
    assert "pass_rate: 50%" in text
    assert "premature_termination: 1" in text
    assert "ok" in text and "bad" in text
