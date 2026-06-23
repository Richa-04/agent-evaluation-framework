"""Run a whole task suite and evaluate deterministic checks per task.

Produces a ``RunResult`` per task (trajectory + deterministic check results +
pinned config). The LLM judge (Phase 3) will attach to these same RunResults.
"""

from __future__ import annotations

import uuid

from ..agent.runner import run_agent
from ..models import RunConfig, RunResult, Task
from ..tools.base import Tool
from .checks import evaluate_checks


async def run_task(task: Task, tools: list[Tool], client, config: RunConfig) -> RunResult:
    """Run one task end-to-end and evaluate its deterministic checks."""
    # Pin per-task fields onto the recorded config for reproducibility.
    task_config = config.model_copy(
        update={"max_steps": task.max_steps, "suite_version": task.suite_version}
    )
    trajectory = await run_agent(task, tools, client, task_config)
    check_results = evaluate_checks(task, trajectory)
    return RunResult(
        run_id=str(uuid.uuid4()),
        task_id=task.id,
        trajectory=trajectory,
        deterministic_results=check_results,
        config=task_config,
        suite_version=task.suite_version,
    )


async def run_suite(
    tasks: list[Task], tools: list[Tool], client, config: RunConfig
) -> list[RunResult]:
    """Run every task in the suite sequentially and return their RunResults."""
    results: list[RunResult] = []
    for task in tasks:
        results.append(await run_task(task, tools, client, config))
    return results


def format_suite_report(results: list[RunResult]) -> str:
    """Render a per-task pass/fail report over the deterministic checks."""
    lines: list[str] = []
    passed = sum(1 for r in results if r.deterministic_passed)
    lines.append(f"=== Deterministic suite report: {passed}/{len(results)} tasks passed ===")
    for result in results:
        status = "PASS" if result.deterministic_passed else "FAIL"
        term = result.trajectory.terminal_state.value
        lines.append("")
        lines.append(f"[{status}] {result.task_id}  (terminal_state={term})")
        for check in result.deterministic_results:
            mark = "✓" if check.passed else "✗"
            lines.append(f"    {mark} {check.type.value}: {check.description}")
            if not check.passed and check.detail:
                lines.append(f"        -> {check.detail}")
    return "\n".join(lines)
