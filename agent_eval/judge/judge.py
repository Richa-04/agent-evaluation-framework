"""The Claude-based LLM judge.

Calls Sonnet 4.6 at ``temperature=0`` with **structured output** (json_schema)
so the verdict is low-variance and always schema-valid. The 1-5 score range is
enforced in-schema via ``enum`` (json_schema structured output does not allow
numeric min/max constraints). The raw structured payload is logged on the
JudgeScore for reproducibility.

The judge supplements the deterministic checks — ``judge_run_result`` attaches a
JudgeScore to a RunResult without touching its ``deterministic_results``.
"""

from __future__ import annotations

import json
from typing import Any

from .. import config as _config
from ..models import DimensionScore, JudgeScore, RunResult, Task, Trajectory
from .prompt import JUDGE_VERSION, build_judge_prompt

_DIMENSION_SCHEMA = {
    "type": "object",
    "properties": {
        # Range enforced via enum — json_schema structured output forbids min/max.
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_completion": _DIMENSION_SCHEMA,
        "tool_selection": _DIMENSION_SCHEMA,
        "efficiency": _DIMENSION_SCHEMA,
        "overall_rationale": {"type": "string"},
    },
    "required": ["goal_completion", "tool_selection", "efficiency", "overall_rationale"],
    "additionalProperties": False,
}


class JudgeError(Exception):
    """Raised when the judge response cannot be parsed into a JudgeScore."""


def _first_text(response: Any) -> str:
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


def build_judge_kwargs(
    task: Task,
    trajectory: Trajectory,
    *,
    model: str,
    temperature: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Construct the messages.create kwargs for a judge call (exposed for tests)."""
    system, user = build_judge_prompt(task, trajectory)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": JUDGE_OUTPUT_SCHEMA}},
    }
    # Sonnet 4.6 accepts temperature (unlike Opus 4.8) — our determinism lever.
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


async def judge_trajectory(
    task: Task,
    trajectory: Trajectory,
    client: Any,
    *,
    model: str = _config.DEFAULT_JUDGE_MODEL,
    temperature: float | None = _config.DEFAULT_JUDGE_TEMPERATURE,
    max_tokens: int = _config.DEFAULT_JUDGE_MAX_TOKENS,
    judge_version: str = JUDGE_VERSION,
) -> JudgeScore:
    """Score one trajectory against the rubric and return a JudgeScore."""
    kwargs = build_judge_kwargs(
        task, trajectory, model=model, temperature=temperature, max_tokens=max_tokens
    )
    response = await client.messages.create(**kwargs)

    text = _first_text(response)
    try:
        data = json.loads(text)
        return JudgeScore(
            goal_completion=DimensionScore(**data["goal_completion"]),
            tool_selection=DimensionScore(**data["tool_selection"]),
            efficiency=DimensionScore(**data["efficiency"]),
            overall_rationale=data["overall_rationale"],
            judge_model=model,
            judge_version=judge_version,
            temperature=temperature,
            raw_response=data,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise JudgeError(
            f"could not parse judge response into a JudgeScore: {exc}; raw text={text!r}"
        ) from exc


async def judge_run_result(
    run_result: RunResult,
    task: Task,
    client: Any,
    **kwargs: Any,
) -> RunResult:
    """Return a copy of ``run_result`` with a JudgeScore attached.

    Deterministic results are untouched — the judge supplements them.
    """
    score = await judge_trajectory(task, run_result.trajectory, client, **kwargs)
    return run_result.model_copy(update={"judge_score": score})


async def judge_run_results(
    run_results: list[RunResult],
    tasks_by_id: dict[str, Task],
    client: Any,
    **kwargs: Any,
) -> list[RunResult]:
    """Judge a list of RunResults, looking each task up by id."""
    judged: list[RunResult] = []
    for result in run_results:
        task = tasks_by_id[result.task_id]
        judged.append(await judge_run_result(result, task, client, **kwargs))
    return judged
