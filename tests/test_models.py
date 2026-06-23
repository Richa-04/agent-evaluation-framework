"""Scaffold tests: validate the core models, their invariants, and round-tripping.

No LLM calls here — these exercise the data layer only. Later phases add mocked-LLM
tests for the runner, judge, storage, and reporting.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.models import (
    CheckResult,
    CheckType,
    DeterministicCheck,
    DimensionScore,
    JudgeScore,
    RunConfig,
    RunResult,
    Task,
    TerminalState,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)


def _sample_trajectory() -> Trajectory:
    step = TrajectoryStep(
        index=0,
        text="I'll compute this.",
        tool_calls=[
            ToolCall(
                id="toolu_1",
                name="calculator",
                arguments={"expression": "2 + 2"},
                result=ToolResult(content="4"),
            )
        ],
        stop_reason="tool_use",
    )
    final = TrajectoryStep(index=1, text="The answer is 4.", stop_reason="end_turn")
    return Trajectory(
        task_id="t1",
        steps=[step, final],
        final_answer="The answer is 4.",
        terminal_state=TerminalState.COMPLETED,
        stop_reason="end_turn",
    )


def test_task_with_checks_builds() -> None:
    task = Task(
        id="add",
        prompt="What is 2 + 2?",
        available_tools=["calculator"],
        expected_outcome="4",
        max_steps=5,
        deterministic_checks=[
            DeterministicCheck(
                id="c1",
                type=CheckType.TOOL_INVOKED,
                description="calculator is used",
                tool_name="calculator",
            ),
            DeterministicCheck(
                id="c2",
                type=CheckType.ANSWER_CONTAINS,
                description="answer mentions 4",
                expected="4",
            ),
        ],
    )
    assert task.max_steps == 5
    assert len(task.deterministic_checks) == 2


def test_trajectory_helpers_capture_tool_args() -> None:
    traj = _sample_trajectory()
    assert traj.step_count == 2
    assert traj.tool_names() == ["calculator"]
    calls = traj.tool_calls()
    assert len(calls) == 1
    # The load-bearing property: arguments are captured, not just a transcript.
    assert calls[0].arguments == {"expression": "2 + 2"}
    assert calls[0].result is not None and calls[0].result.content == "4"


def test_task_max_steps_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Task(id="bad", prompt="x", max_steps=0)


def test_dimension_score_bounds() -> None:
    DimensionScore(score=1, rationale="floor")
    DimensionScore(score=5, rationale="ceiling")
    with pytest.raises(ValidationError):
        DimensionScore(score=0, rationale="too low")
    with pytest.raises(ValidationError):
        DimensionScore(score=6, rationale="too high")


def test_judge_score_average() -> None:
    js = JudgeScore(
        goal_completion=DimensionScore(score=5, rationale="done"),
        tool_selection=DimensionScore(score=4, rationale="ok"),
        efficiency=DimensionScore(score=3, rationale="meh"),
        overall_rationale="solid",
        judge_model="claude-opus-4-8",
    )
    assert js.average == pytest.approx(4.0)


def test_run_result_roundtrip_and_deterministic_passed() -> None:
    result = RunResult(
        run_id="r1",
        task_id="t1",
        trajectory=_sample_trajectory(),
        deterministic_results=[
            CheckResult(
                check_id="c1",
                type=CheckType.TOOL_INVOKED,
                description="calculator used",
                passed=True,
            ),
            CheckResult(
                check_id="c2",
                type=CheckType.ANSWER_CONTAINS,
                description="answer mentions 4",
                passed=True,
                detail="found '4'",
            ),
        ],
        config=RunConfig(model="claude-opus-4-8", effort="high", thinking="adaptive"),
    )
    assert result.deterministic_passed is True

    # JSON round-trip preserves nested trajectory + tool args.
    dumped = result.model_dump_json()
    restored = RunResult.model_validate_json(dumped)
    assert restored.trajectory.tool_calls()[0].arguments == {"expression": "2 + 2"}
    assert restored.deterministic_passed is True


def test_deterministic_passed_false_when_any_check_fails() -> None:
    result = RunResult(
        run_id="r2",
        task_id="t1",
        trajectory=_sample_trajectory(),
        deterministic_results=[
            CheckResult(check_id="c1", type=CheckType.TOOL_INVOKED, description="x", passed=True),
            CheckResult(check_id="c2", type=CheckType.ANSWER_CONTAINS, description="y", passed=False),
        ],
        config=RunConfig(model="claude-opus-4-8"),
    )
    assert result.deterministic_passed is False


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RunConfig(model="claude-opus-4-8", bogus_field=123)  # type: ignore[call-arg]
