"""Typer CLI entry point. Commands are added in Phase 5.

Kept minimal now so `agent-eval --help` works after `pip install -e .`.
"""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    help="Agent Evaluation Framework — run agent task suites and view reports.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Agent Evaluation Framework CLI. Subcommands are added in Phase 5."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
