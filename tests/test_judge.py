"""Phase 3 tests: the LLM judge, with the judge model MOCKED for determinism.

Covers: structured-output request shape (model, temperature=0, json_schema),
prompt contents (goal, tool arguments, rubric), parsing into a JudgeScore,
attaching the score to a RunResult without disturbing deterministic results,
low scores for a broken trajectory, and malformed-response handling.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_eval.config import DEFAULT_JUDGE_MODEL
from agent_eval.judge import (
    JudgeError,
    build_judge_prompt,
    judge_run_result,
    judge_trajectory,
)
from agent_eval.judge.judge import JUDGE_OUTPUT_SCHEMA
from agent_eval.models import (
    CheckResult,
    CheckType,
    RunConfig,
    RunResult,
    TerminalState,
    Task,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)

from .fakes import OneShotClient, json_message

TASK = Task(
    id="calc-multiply",
    prompt="What is 47 * 19? Use the calculator tool.",
    available_tools=["calculator"],
    expected_outcome="893",
    max_steps=5,
)


def _good_trajectory() -> Trajectory:
    return Trajectory(
        task_id="calc-multiply",
        steps=[
            TrajectoryStep(
                index=0,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="calculator",
                        arguments={"expression": "47 * 19"},
                        result=ToolResult(content="893"),
                    )
                ],
                stop_reason="tool_use",
            ),
            TrajectoryStep(index=1, text="47 * 19 = 893.", stop_reason="end_turn"),
        ],
        final_answer="47 * 19 = 893.",
        terminal_state=TerminalState.COMPLETED,
        stop_reason="end_turn",
    )


_HIGH_VERDICT = {
    "goal_completion": {"score": 5, "rationale": "Correct answer 893."},
    "tool_selection": {"score": 5, "rationale": "Used calculator with the right expression."},
    "efficiency": {"score": 4, "rationale": "Two steps, direct."},
    "overall_rationale": "Solid: correct, well-tooled, efficient.",
}


def test_judge_request_shape_is_structured_and_temperature_zero() -> None:
    client = OneShotClient(json_message(_HIGH_VERDICT))
    asyncio.run(judge_trajectory(TASK, _good_trajectory(), client))

    assert len(client.calls) == 1
    kwargs = client.calls[0]
    assert kwargs["model"] == DEFAULT_JUDGE_MODEL == "claude-sonnet-4-6"
    assert kwargs["temperature"] == 0.0  # pinned for repeatability
    # Structured output to the judge schema.
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] is JUDGE_OUTPUT_SCHEMA


def test_judge_prompt_includes_goal_tool_args_and_rubric() -> None:
    system, user = build_judge_prompt(TASK, _good_trajectory())
    # Goal + expected outcome present.
    assert "What is 47 * 19?" in user
    assert "893" in user
    # Tool ARGUMENTS captured in the trajectory rendering (the hard part).
    assert "expression" in user and "47 * 19" in user
    # Rubric dimensions present in the system prompt.
    assert "goal_completion" in system
    assert "tool_selection" in system
    assert "efficiency" in system


def test_judge_parses_structured_output_into_judgescore() -> None:
    client = OneShotClient(json_message(_HIGH_VERDICT))
    score = asyncio.run(judge_trajectory(TASK, _good_trajectory(), client))

    assert score.goal_completion.score == 5
    assert score.tool_selection.score == 5
    assert score.efficiency.score == 4
    assert score.overall_rationale.startswith("Solid")
    assert score.judge_model == "claude-sonnet-4-6"
    assert score.judge_version == "v1"
    assert score.temperature == 0.0
    assert score.average == pytest.approx((5 + 5 + 4) / 3)
    # Raw response logged for reproducibility.
    assert score.raw_response == _HIGH_VERDICT


def test_judge_attaches_score_without_replacing_deterministic_results() -> None:
    deterministic = [
        CheckResult(check_id="tool", type=CheckType.TOOL_INVOKED, description="calc used", passed=True),
        CheckResult(check_id="ans", type=CheckType.ANSWER_CONTAINS, description="has 893", passed=True),
    ]
    run = RunResult(
        run_id="r1",
        task_id="calc-multiply",
        trajectory=_good_trajectory(),
        deterministic_results=deterministic,
        config=RunConfig(model="claude-opus-4-8"),
    )
    client = OneShotClient(json_message(_HIGH_VERDICT))
    judged = asyncio.run(judge_run_result(run, TASK, client))

    # Judge supplements — deterministic results are preserved unchanged.
    assert judged.deterministic_results == deterministic
    assert judged.deterministic_passed is True
    assert judged.judge_score is not None
    assert judged.judge_score.goal_completion.score == 5
    # Original run object is not mutated (model_copy).
    assert run.judge_score is None


def test_judge_gives_low_scores_for_broken_trajectory() -> None:
    # Agent skipped the calculator and gave an imprecise answer.
    broken = Trajectory(
        task_id="convert",
        steps=[
            TrajectoryStep(
                index=0,
                tool_calls=[
                    ToolCall(id="c1", name="search", arguments={"query": "speed of light"},
                             result=ToolResult(content="299792458 m/s"))
                ],
                stop_reason="tool_use",
            ),
            TrajectoryStep(index=1, text="About 300,000 km/s.", stop_reason="end_turn"),
        ],
        final_answer="About 300,000 km/s.",
        terminal_state=TerminalState.COMPLETED,
        stop_reason="end_turn",
    )
    low_verdict = {
        "goal_completion": {"score": 2, "rationale": "Imprecise; expected 299792.458."},
        "tool_selection": {"score": 2, "rationale": "Skipped the required calculator tool."},
        "efficiency": {"score": 3, "rationale": "Few steps but incomplete."},
        "overall_rationale": "Reached a completed state but skipped a required tool and was imprecise.",
    }
    client = OneShotClient(json_message(low_verdict))
    score = asyncio.run(judge_trajectory(TASK, broken, client))
    assert score.goal_completion.score <= 2
    assert score.tool_selection.score <= 2


def test_malformed_judge_response_raises_judgeerror() -> None:
    from .fakes import text_block
    from types import SimpleNamespace

    bad = SimpleNamespace(content=[text_block("not json")], stop_reason="end_turn", usage=None)
    client = OneShotClient(bad)
    with pytest.raises(JudgeError):
        asyncio.run(judge_trajectory(TASK, _good_trajectory(), client))


def test_judgescore_roundtrips_with_raw_response() -> None:
    client = OneShotClient(json_message(_HIGH_VERDICT))
    score = asyncio.run(judge_trajectory(TASK, _good_trajectory(), client))
    restored = type(score).model_validate_json(score.model_dump_json())
    assert restored.raw_response == _HIGH_VERDICT
    assert restored.temperature == 0.0
