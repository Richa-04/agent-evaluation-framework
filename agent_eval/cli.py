"""Typer CLI for the Agent Evaluation Framework.

Commands:
  run       run a task suite, judge each trajectory, and persist to SQLite
  versions  list stored version labels in a SQLite db
  report    failure attribution + pandas summary for one stored version
  compare   per-dimension + deterministic regression comparison of two versions

`run` calls the real Anthropic API (agent on Opus 4.8, judge on Sonnet 4.6) and
needs ANTHROPIC_API_KEY. `versions` / `report` / `compare` operate purely on the
stored SQLite data and need no API key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from . import __version__, config
from .judge import judge_run_results
from .models import RunConfig
from .reporting import attribute_failures, format_report
from .storage import RunStore, compare_versions, format_comparison, make_version_label
from .suite import load_suite, run_suite
from .tools import build_default_tools

app = typer.Typer(
    help="Agent Evaluation Framework — run agent task suites and view reports.",
    no_args_is_help=True,
)

_DEFAULT_SUITE = Path("tasks/suite_v1.json")
_DEFAULT_READ_DIR = Path("tasks/files")


@app.callback()
def main() -> None:
    """Agent Evaluation Framework CLI."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def run(
    db: Path = typer.Option(..., help="SQLite database file to write results to."),
    suite: Path = typer.Option(_DEFAULT_SUITE, help="Task suite JSON file."),
    read_dir: Path = typer.Option(_DEFAULT_READ_DIR, help="Sandbox dir for the read_file tool."),
    label: str = typer.Option(None, help="Version label (auto-generated if omitted)."),
    tag: str = typer.Option(None, help="Optional human tag appended to an auto label."),
    judge: bool = typer.Option(True, help="Run the LLM judge after the suite."),
) -> None:
    """Run a suite against the real API, judge it, and persist the results."""
    version_str, tasks = load_suite(suite)
    tasks_by_id = {t.id: t for t in tasks}
    tools = build_default_tools(read_base_dir=read_dir)

    async def _go():
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=config.get_api_key())
        agent_config = RunConfig(
            model=config.DEFAULT_MODEL,
            thinking="adaptive",
            max_tokens=config.DEFAULT_AGENT_MAX_TOKENS,
            suite_version=version_str,
        )
        results = await run_suite(tasks, tools, client, agent_config)
        if judge:
            results = await judge_run_results(results, tasks_by_id, client)
        return results

    try:
        results = asyncio.run(_go())
    except RuntimeError as exc:  # e.g. missing API key
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    final_label = label or make_version_label(config.DEFAULT_MODEL, version_str, tag=tag)
    with RunStore(db) as store:
        store.save_runs(results, final_label)
    typer.echo(f"Saved {len(results)} task runs under label: {final_label}\n")
    typer.echo(format_report(results, tasks_by_id))


@app.command()
def versions(db: Path = typer.Option(..., help="SQLite database file.")) -> None:
    """List stored version labels."""
    with RunStore(db) as store:
        infos = store.list_versions()
    if not infos:
        typer.echo("(no versions stored)")
        return
    for info in infos:
        typer.echo(
            f"{info.version_label}  "
            f"(tasks={info.task_count}, model={info.agent_model}, suite={info.suite_version})"
        )


@app.command()
def report(
    db: Path = typer.Option(..., help="SQLite database file."),
    label: str = typer.Option(..., help="Version label to report on."),
    suite: Path = typer.Option(_DEFAULT_SUITE, help="Task suite JSON (for failure attribution)."),
) -> None:
    """Show failure attribution + a pandas summary for one stored version."""
    _version, tasks = load_suite(suite)
    tasks_by_id = {t.id: t for t in tasks}
    with RunStore(db) as store:
        results = store.get_runs(label)
    if not results:
        typer.secho(f"No runs found for label {label!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"=== Failure attribution: {label} ===")
    any_failed = False
    for failure in attribute_failures(results, tasks_by_id):
        if not failure.failed:
            continue
        any_failed = True
        typer.echo(f"\n[{failure.primary_category.value}] {failure.task_id}")
        for finding in failure.findings:
            where = "" if finding.step_index is None else f" (step {finding.step_index})"
            typer.echo(f"    - {finding.category.value}{where}: {finding.detail}")
    if not any_failed:
        typer.echo("All tasks passed — no failures to attribute.")

    typer.echo("")
    typer.echo(format_report(results, tasks_by_id))


@app.command()
def compare(
    db: Path = typer.Option(..., help="SQLite database file."),
    label_a: str = typer.Option(..., "--label-a", help="Baseline (before) version label."),
    label_b: str = typer.Option(..., "--label-b", help="Candidate (after) version label."),
) -> None:
    """Compare two stored versions: per-dimension deltas + regressions."""
    with RunStore(db) as store:
        comparison = compare_versions(store, label_a, label_b)
    typer.echo(format_comparison(comparison))


if __name__ == "__main__":
    app()
