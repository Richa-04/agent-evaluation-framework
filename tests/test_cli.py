"""Phase 5 tests: the typer CLI (versions / report / compare) against a SQLite db
populated directly with constructed RunResults — no LLM, no network."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agent_eval.cli import app
from agent_eval.models import (
    CheckResult,
    CheckType,
    DimensionScore,
    JudgeScore,
    RunConfig,
    RunResult,
    TerminalState,
    ToolCall,
    ToolResult,
    Trajectory,
    TrajectoryStep,
)
from agent_eval.storage import RunStore

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE = str(REPO_ROOT / "tasks" / "suite_v1.json")


def _judge(g, t, e):
    return JudgeScore(
        goal_completion=DimensionScore(score=g, rationale="g"),
        tool_selection=DimensionScore(score=t, rationale="t"),
        efficiency=DimensionScore(score=e, rationale="e"),
        overall_rationale="o",
        judge_model="claude-sonnet-5",
    )


def _result(task_id, *, det_passed, dims, tool_calls=None, terminal=TerminalState.COMPLETED):
    g, t, e = dims
    steps = [
        TrajectoryStep(index=0, tool_calls=tool_calls or [], stop_reason="tool_use" if tool_calls else "end_turn"),
        TrajectoryStep(index=1, text="ans", stop_reason="end_turn"),
    ]
    return RunResult(
        run_id="r",
        task_id=task_id,
        trajectory=Trajectory(task_id=task_id, steps=steps, final_answer="ans",
                              terminal_state=terminal, stop_reason="end_turn"),
        deterministic_results=[
            CheckResult(check_id="x", type=CheckType.ANSWER_CONTAINS, description="d", passed=det_passed)
        ],
        judge_score=_judge(g, t, e),
        config=RunConfig(model="claude-opus-5", suite_version="v1"),
        suite_version="v1",
    )


def _convert_fail() -> RunResult:
    """A realistic regression of 'convert-speed-of-light': calls search, skips the
    required calculator. Deterministic check ids match tasks/suite_v1.json so
    failure attribution resolves the missing required tool."""
    search_call = ToolCall(id="1", name="search", arguments={"query": "speed of light"},
                           result=ToolResult(content="299792458 m/s"))
    return RunResult(
        run_id="r",
        task_id="convert-speed-of-light",
        trajectory=Trajectory(
            task_id="convert-speed-of-light",
            steps=[
                TrajectoryStep(index=0, tool_calls=[search_call], stop_reason="tool_use"),
                TrajectoryStep(index=1, text="about 300000 km/s", stop_reason="end_turn"),
            ],
            final_answer="about 300000 km/s",
            terminal_state=TerminalState.COMPLETED,
            stop_reason="end_turn",
        ),
        deterministic_results=[
            CheckResult(check_id="search", type=CheckType.TOOL_INVOKED, description="search used", passed=True),
            CheckResult(check_id="calc", type=CheckType.TOOL_INVOKED, description="calculator used", passed=False),
            CheckResult(check_id="answer", type=CheckType.ANSWER_CONTAINS, description="precise value", passed=False),
            CheckResult(check_id="done", type=CheckType.TERMINAL_STATE, description="completed", passed=True),
        ],
        judge_score=_judge(2, 2, 3),
        config=RunConfig(model="claude-opus-5", suite_version="v1"),
        suite_version="v1",
    )


def _populate(db: Path) -> None:
    store = RunStore(db)
    # baseline: all four suite tasks pass
    baseline = [
        _result("calc-multiply", det_passed=True, dims=(5, 5, 5)),
        _result("search-capital", det_passed=True, dims=(5, 5, 5)),
        _result("read-codename", det_passed=True, dims=(5, 5, 4)),
        _result("convert-speed-of-light", det_passed=True, dims=(5, 5, 4)),
    ]
    # candidate: convert-speed-of-light regresses (skips the calculator)
    candidate = [
        _result("calc-multiply", det_passed=True, dims=(5, 5, 5)),
        _result("search-capital", det_passed=True, dims=(5, 5, 5)),
        _result("read-codename", det_passed=True, dims=(5, 5, 4)),
        _convert_fail(),
    ]
    store.save_runs(baseline, "A")
    store.save_runs(candidate, "B")
    store.close()


def test_versions_lists_labels(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _populate(db)
    result = runner.invoke(app, ["versions", "--db", str(db)])
    assert result.exit_code == 0
    assert "A" in result.stdout and "B" in result.stdout
    assert "tasks=4" in result.stdout


def test_report_shows_summary_and_attribution(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _populate(db)
    result = runner.invoke(app, ["report", "--db", str(db), "--label", "B", "--suite", SUITE])
    assert result.exit_code == 0
    assert "Suite summary report" in result.stdout
    assert "pass_rate: 75%" in result.stdout
    assert "Failure attribution" in result.stdout
    # The regressed task is attributed (required calculator never invoked).
    assert "convert-speed-of-light" in result.stdout
    assert "premature_termination" in result.stdout


def test_compare_reports_regression(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _populate(db)
    result = runner.invoke(app, ["compare", "--db", str(db), "--label-a", "A", "--label-b", "B"])
    assert result.exit_code == 0
    assert "REGRESSED (1)" in result.stdout
    assert "convert-speed-of-light" in result.stdout
    assert "PASS->FAIL" in result.stdout


def test_report_missing_label_errors(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite"
    _populate(db)
    result = runner.invoke(app, ["report", "--db", str(db), "--label", "nope", "--suite", SUITE])
    assert result.exit_code == 1
