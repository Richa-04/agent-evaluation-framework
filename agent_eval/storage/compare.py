"""Version-to-version regression comparison.

Given two version labels (A = before, B = after), compute per-task changes in
both the deterministic pass/fail outcome AND the three judge dimensions
(goal completion, tool selection, efficiency), then classify each task as
REGRESSED / IMPROVED / UNCHANGED / ADDED / REMOVED.

Deterministic outcome is the ground truth and decides first; judge scores break
ties when the deterministic result is unchanged.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ..models import RunResult
from .db import RunStore

_EPS = 1e-9


class ComparisonStatus(str, Enum):
    REGRESSED = "regressed"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    ADDED = "added"      # present only in B
    REMOVED = "removed"  # present only in A


class TaskComparison(BaseModel):
    task_id: str
    status: ComparisonStatus

    deterministic_before: bool | None = None
    deterministic_after: bool | None = None

    avg_before: float | None = None
    avg_after: float | None = None
    average_delta: float | None = None

    goal_completion_delta: int | None = None
    tool_selection_delta: int | None = None
    efficiency_delta: int | None = None

    detail: str = ""


class VersionComparison(BaseModel):
    label_a: str
    label_b: str
    tasks: list[TaskComparison]

    def by_status(self, status: ComparisonStatus) -> list[TaskComparison]:
        return [t for t in self.tasks if t.status is status]

    @property
    def regressed(self) -> list[TaskComparison]:
        return self.by_status(ComparisonStatus.REGRESSED)

    @property
    def improved(self) -> list[TaskComparison]:
        return self.by_status(ComparisonStatus.IMPROVED)


def _classify(
    det_before: bool, det_after: bool, avg_before: float | None, avg_after: float | None
) -> ComparisonStatus:
    # Deterministic outcome is ground truth and decides first.
    if det_before and not det_after:
        return ComparisonStatus.REGRESSED
    if (not det_before) and det_after:
        return ComparisonStatus.IMPROVED
    # Deterministic unchanged — break the tie on judge average.
    if avg_before is not None and avg_after is not None:
        if avg_after < avg_before - _EPS:
            return ComparisonStatus.REGRESSED
        if avg_after > avg_before + _EPS:
            return ComparisonStatus.IMPROVED
    return ComparisonStatus.UNCHANGED


def _compare_task(task_id: str, a: RunResult | None, b: RunResult | None) -> TaskComparison:
    if a is None:
        return TaskComparison(
            task_id=task_id,
            status=ComparisonStatus.ADDED,
            deterministic_after=b.deterministic_passed if b else None,
            avg_after=b.judge_score.average if b and b.judge_score else None,
            detail="present only in B",
        )
    if b is None:
        return TaskComparison(
            task_id=task_id,
            status=ComparisonStatus.REMOVED,
            deterministic_before=a.deterministic_passed,
            avg_before=a.judge_score.average if a.judge_score else None,
            detail="present only in A",
        )

    det_before = a.deterministic_passed
    det_after = b.deterministic_passed
    ja, jb = a.judge_score, b.judge_score
    avg_before = ja.average if ja else None
    avg_after = jb.average if jb else None

    average_delta = (avg_after - avg_before) if (avg_before is not None and avg_after is not None) else None
    if ja and jb:
        goal_d = jb.goal_completion.score - ja.goal_completion.score
        tool_d = jb.tool_selection.score - ja.tool_selection.score
        eff_d = jb.efficiency.score - ja.efficiency.score
    else:
        goal_d = tool_d = eff_d = None

    status = _classify(det_before, det_after, avg_before, avg_after)

    bits: list[str] = []
    if det_before != det_after:
        bits.append(f"deterministic {_pf(det_before)}->{_pf(det_after)}")
    else:
        bits.append(f"deterministic {_pf(det_before)} (unchanged)")
    if average_delta is not None:
        bits.append(f"judge avg {avg_before:.2f}->{avg_after:.2f} (Δ{average_delta:+.2f})")

    return TaskComparison(
        task_id=task_id,
        status=status,
        deterministic_before=det_before,
        deterministic_after=det_after,
        avg_before=avg_before,
        avg_after=avg_after,
        average_delta=average_delta,
        goal_completion_delta=goal_d,
        tool_selection_delta=tool_d,
        efficiency_delta=eff_d,
        detail="; ".join(bits),
    )


def _pf(passed: bool | None) -> str:
    return "PASS" if passed else "FAIL"


def compare_versions(store: RunStore, label_a: str, label_b: str) -> VersionComparison:
    """Compare two stored version labels (A = before, B = after)."""
    runs_a = {r.task_id: r for r in store.get_runs(label_a)}
    runs_b = {r.task_id: r for r in store.get_runs(label_b)}
    task_ids = sorted(set(runs_a) | set(runs_b))
    tasks = [_compare_task(tid, runs_a.get(tid), runs_b.get(tid)) for tid in task_ids]
    return VersionComparison(label_a=label_a, label_b=label_b, tasks=tasks)


def _dim_deltas(t: TaskComparison) -> str:
    if t.goal_completion_delta is None:
        return ""
    return (
        f"  [goal Δ{t.goal_completion_delta:+d}, "
        f"tool Δ{t.tool_selection_delta:+d}, eff Δ{t.efficiency_delta:+d}]"
    )


def format_comparison(comparison: VersionComparison) -> str:
    """Render the comparison, leading with regressions so they aren't buried."""
    lines: list[str] = []
    lines.append("=== Version comparison ===")
    lines.append(f"A (before): {comparison.label_a}")
    lines.append(f"B (after):  {comparison.label_b}")

    counts = {s: len(comparison.by_status(s)) for s in ComparisonStatus}
    lines.append(
        f"tasks: {len(comparison.tasks)} | "
        f"regressed: {counts[ComparisonStatus.REGRESSED]} | "
        f"improved: {counts[ComparisonStatus.IMPROVED]} | "
        f"unchanged: {counts[ComparisonStatus.UNCHANGED]} | "
        f"added: {counts[ComparisonStatus.ADDED]} | "
        f"removed: {counts[ComparisonStatus.REMOVED]}"
    )

    regressed = comparison.regressed
    lines.append("")
    if regressed:
        lines.append(f"!!! REGRESSED ({len(regressed)}):")
        for t in regressed:
            lines.append(f"  - {t.task_id}: {t.detail}{_dim_deltas(t)}")
    else:
        lines.append("No regressions. ✓")

    if comparison.improved:
        lines.append("")
        lines.append(f"IMPROVED ({len(comparison.improved)}):")
        for t in comparison.improved:
            lines.append(f"  - {t.task_id}: {t.detail}{_dim_deltas(t)}")

    lines.append("")
    lines.append("Full per-task detail:")
    for t in comparison.tasks:
        lines.append(f"  [{t.status.value.upper()}] {t.task_id}: {t.detail}{_dim_deltas(t)}")

    return "\n".join(lines)
