"""Agent tools: calculator, mock search, read_file.

``build_default_tools`` constructs the standard set; ``select_tools`` filters a
tool list down to a set of names (e.g. a task's ``available_tools``).
"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError
from .calculator import CalculatorTool
from .read_file import ReadFileTool
from .search import MockSearchTool

__all__ = [
    "Tool",
    "ToolError",
    "CalculatorTool",
    "MockSearchTool",
    "ReadFileTool",
    "build_default_tools",
    "select_tools",
]


def build_default_tools(
    *,
    search_corpus: dict[str, str] | None = None,
    read_base_dir: str | Path | None = None,
) -> list[Tool]:
    """Build the standard toolset.

    ``read_base_dir`` defaults to the current working directory; pass an explicit
    sandbox directory for the read_file tool when running suites or demos.
    """
    return [
        CalculatorTool(),
        MockSearchTool(corpus=search_corpus),
        ReadFileTool(base_dir=read_base_dir or Path.cwd()),
    ]


def select_tools(tools: list[Tool], names: list[str]) -> list[Tool]:
    """Return the subset of ``tools`` whose names are in ``names`` (preserving order)."""
    by_name = {t.name: t for t in tools}
    return [by_name[n] for n in names if n in by_name]
