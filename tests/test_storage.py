"""Phase 4 tests: SQLite persistence + version regression comparison.

No LLM involved — RunResults are constructed directly so the tests are fast and
deterministic.
"""

from __future__ import annotations

from pathlib import Path

from agent_eval.models import (
    CheckResult,
    CheckType,
    DimensionScore,
    JudgeScore,
    RunConfig,
    RunResult,
    TerminalState,
    Trajectory,
    TrajectoryStep,
)
from agent_eval.storage import (
    ComparisonStatus,
    RunStore,
    compare_versions,
    format_comparison,
    make_version_label,
)


def _result(
    task_id: str,
    *,
    det_passed: bool,
    dims: tuple[int, int, int] | None,
    run_id: str = "r",
    suite_version: str = "v1",
    model: str = "claude-opus-4-8",
) -> RunResult:
    judge = None
    if dims is not None:
        g, t, e = dims
        judge = JudgeScore(
            goal_completion=DimensionScore(score=g, rationale="g"),
            tool_selection=DimensionScore(score=t, rationale="t"),
            efficiency=DimensionScore(score=e, rationale="e"),
            overall_rationale="o",
            judge_model="claude-sonnet-4-6",
            temperature=0.0,
        )
    return RunResult(
        run_id=run_id,
        task_id=task_id,
        trajectory=Trajectory(
            task_id=task_id,
            steps=[TrajectoryStep(index=0, text="done", stop_reason="end_turn")],
            final_answer="done",
            terminal_state=TerminalState.COMPLETED,
            stop_reason="end_turn",
        ),
        deterministic_results=[
            CheckResult(check_id="c", type=CheckType.ANSWER_CONTAINS, description="d", passed=det_passed)
        ],
        judge_score=judge,
        config=RunConfig(model=model, suite_version=suite_version),
        suite_version=suite_version,
    )


# --- persistence ------------------------------------------------------------
def test_save_and_get_roundtrip_preserves_judge_and_raw() -> None:
    with RunStore() as store:
        r = _result("t1", det_passed=True, dims=(5, 4, 3))
        store.save_run(r, "v1/baseline")
        back = store.get_runs("v1/baseline")
        assert len(back) == 1
        got = back[0]
        assert got.task_id == "t1"
        assert got.deterministic_passed is True
        assert got.judge_score is not None
        assert got.judge_score.goal_completion.score == 5
        assert got.judge_score.average == r.judge_score.average


def test_unique_per_version_task_replaces() -> None:
    with RunStore() as store:
        store.save_run(_result("t1", det_passed=False, dims=(2, 2, 2)), "label")
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "label")  # replace
        runs = store.get_runs("label")
        assert len(runs) == 1
        assert runs[0].deterministic_passed is True
        assert runs[0].judge_score.goal_completion.score == 5


def test_list_versions_summary() -> None:
    with RunStore() as store:
        store.save_runs(
            [_result("t1", det_passed=True, dims=(5, 5, 5)), _result("t2", det_passed=True, dims=(4, 4, 4))],
            "A",
        )
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "B")
        versions = {v.version_label: v for v in store.list_versions()}
        assert versions["A"].task_count == 2
        assert versions["B"].task_count == 1
        assert versions["A"].suite_version == "v1"


def test_file_persistence_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    store = RunStore(db)
    store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "A")
    store.close()

    reopened = RunStore(db)
    assert len(reopened.get_runs("A")) == 1
    reopened.close()


def test_make_version_label_captures_model_suite_timestamp() -> None:
    label = make_version_label("claude-opus-4-8", "v1", tag="baseline")
    assert label.startswith("claude-opus-4-8/v1/")
    assert label.endswith("baseline")


# --- comparison -------------------------------------------------------------
def test_compare_detects_regression() -> None:
    with RunStore() as store:
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "A")
        store.save_run(_result("t1", det_passed=False, dims=(2, 1, 3)), "B")
        cmp = compare_versions(store, "A", "B")
        t = cmp.tasks[0]
        assert t.status is ComparisonStatus.REGRESSED
        assert t.deterministic_before is True and t.deterministic_after is False
        assert t.goal_completion_delta == -3
        assert t.tool_selection_delta == -4
        assert t.efficiency_delta == -2
        assert [x.task_id for x in cmp.regressed] == ["t1"]


def test_compare_detects_improvement() -> None:
    with RunStore() as store:
        store.save_run(_result("t1", det_passed=False, dims=(2, 2, 2)), "A")
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "B")
        cmp = compare_versions(store, "A", "B")
        assert cmp.tasks[0].status is ComparisonStatus.IMPROVED
        assert [x.task_id for x in cmp.improved] == ["t1"]


def test_compare_judge_breaks_tie_when_deterministic_unchanged() -> None:
    # Both pass deterministically, but judge average dropped -> regression.
    with RunStore() as store:
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "A")
        store.save_run(_result("t1", det_passed=True, dims=(5, 4, 3)), "B")
        cmp = compare_versions(store, "A", "B")
        assert cmp.tasks[0].status is ComparisonStatus.REGRESSED


def test_compare_unchanged_and_added_removed() -> None:
    with RunStore() as store:
        store.save_run(_result("same", det_passed=True, dims=(5, 5, 5)), "A")
        store.save_run(_result("same", det_passed=True, dims=(5, 5, 5)), "B")
        store.save_run(_result("only_a", det_passed=True, dims=(5, 5, 5)), "A")
        store.save_run(_result("only_b", det_passed=True, dims=(5, 5, 5)), "B")
        cmp = compare_versions(store, "A", "B")
        by_id = {t.task_id: t.status for t in cmp.tasks}
        assert by_id["same"] is ComparisonStatus.UNCHANGED
        assert by_id["only_a"] is ComparisonStatus.REMOVED
        assert by_id["only_b"] is ComparisonStatus.ADDED


def test_format_comparison_calls_out_regression() -> None:
    with RunStore() as store:
        store.save_run(_result("t1", det_passed=True, dims=(5, 5, 5)), "A")
        store.save_run(_result("t1", det_passed=False, dims=(1, 1, 2)), "B")
        report = format_comparison(compare_versions(store, "A", "B"))
        assert "REGRESSED (1)" in report
        assert "t1" in report
        assert "PASS->FAIL" in report
