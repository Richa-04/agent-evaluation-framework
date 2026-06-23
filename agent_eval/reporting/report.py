"""pandas-based summary reporting over a suite run.

Aggregates RunResults into: pass rate, average judge score per dimension, and a
breakdown of failure types (from failure attribution). Also exposes the per-task
detail as a pandas DataFrame for ad-hoc analysis.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from ..models import RunResult, Task
from .attribution import FailureCategory, attribute_failure


class SuiteReport(BaseModel):
    total_tasks: int
    passed: int
    pass_rate: float
    avg_goal_completion: float | None
    avg_tool_selection: float | None
    avg_efficiency: float | None
    avg_overall: float | None
    failure_breakdown: dict[str, int]


def results_dataframe(run_results: list[RunResult], tasks_by_id: dict[str, Task]) -> pd.DataFrame:
    """One row per task: outcome, judge dimensions, and failure attribution."""
    rows: list[dict] = []
    for r in run_results:
        js = r.judge_score
        failure = attribute_failure(r, tasks_by_id[r.task_id])
        rows.append(
            {
                "task_id": r.task_id,
                "passed": r.deterministic_passed,
                "terminal_state": r.trajectory.terminal_state.value,
                "goal_completion": js.goal_completion.score if js else float("nan"),
                "tool_selection": js.tool_selection.score if js else float("nan"),
                "efficiency": js.efficiency.score if js else float("nan"),
                "judge_avg": js.average if js else float("nan"),
                "failure": "" if not failure.failed else failure.primary_category.value,
            }
        )
    columns = [
        "task_id", "passed", "terminal_state",
        "goal_completion", "tool_selection", "efficiency", "judge_avg", "failure",
    ]
    return pd.DataFrame(rows, columns=columns)


def _mean_or_none(series: pd.Series) -> float | None:
    value = series.mean(skipna=True)
    return None if pd.isna(value) else round(float(value), 3)


def summarize(run_results: list[RunResult], tasks_by_id: dict[str, Task]) -> SuiteReport:
    """Aggregate a suite run into a SuiteReport (uses pandas for the aggregates)."""
    df = results_dataframe(run_results, tasks_by_id)
    total = len(df)
    passed = int(df["passed"].sum()) if total else 0
    pass_rate = round(float(df["passed"].mean()), 3) if total else 0.0

    failed = df.loc[~df["passed"], "failure"]
    breakdown = {k: int(v) for k, v in failed.value_counts().items()}

    return SuiteReport(
        total_tasks=total,
        passed=passed,
        pass_rate=pass_rate,
        avg_goal_completion=_mean_or_none(df["goal_completion"]),
        avg_tool_selection=_mean_or_none(df["tool_selection"]),
        avg_efficiency=_mean_or_none(df["efficiency"]),
        avg_overall=_mean_or_none(df["judge_avg"]),
        failure_breakdown=breakdown,
    )


def format_report(run_results: list[RunResult], tasks_by_id: dict[str, Task]) -> str:
    """Render a full summary report: aggregates + failure breakdown + per-task table."""
    df = results_dataframe(run_results, tasks_by_id)
    rep = summarize(run_results, tasks_by_id)

    lines: list[str] = []
    lines.append("=== Suite summary report ===")
    lines.append(
        f"tasks: {rep.total_tasks} | passed: {rep.passed} | "
        f"pass_rate: {rep.pass_rate:.0%}"
    )
    lines.append("")
    lines.append("Average judge scores (1-5):")
    lines.append(f"  goal_completion: {_fmt(rep.avg_goal_completion)}")
    lines.append(f"  tool_selection:  {_fmt(rep.avg_tool_selection)}")
    lines.append(f"  efficiency:      {_fmt(rep.avg_efficiency)}")
    lines.append(f"  overall:         {_fmt(rep.avg_overall)}")
    lines.append("")
    lines.append("Failure type breakdown:")
    if rep.failure_breakdown:
        for category, count in sorted(rep.failure_breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {category}: {count}")
    else:
        lines.append("  (no failures)")
    lines.append("")
    lines.append("Per-task:")
    display = df.copy()
    for col in ("goal_completion", "tool_selection", "efficiency", "judge_avg"):
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    lines.append(display.to_string(index=False))
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
