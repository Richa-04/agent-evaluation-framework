"""Phase 1 tests: the individual tools (no LLM involved)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.tools import CalculatorTool, MockSearchTool, ReadFileTool


def _run(coro):
    return asyncio.run(coro)


def test_calculator_basic_and_formatting() -> None:
    tool = CalculatorTool()
    assert _run(tool.run({"expression": "47 * 19"})).content == "893"
    assert _run(tool.run({"expression": "(12 + 8) * 5"})).content == "100"
    assert _run(tool.run({"expression": "299792458 / 1000"})).content == "299792.458"


def test_calculator_rejects_bad_input() -> None:
    tool = CalculatorTool()
    assert _run(tool.run({"expression": "2 +"})).is_error is True       # syntax
    assert _run(tool.run({"expression": "1/0"})).is_error is True        # div by zero
    assert _run(tool.run({"expression": "__import__('os')"})).is_error is True  # no calls/names
    assert _run(tool.run({"expression": ""})).is_error is True           # empty


def test_search_matches_and_misses() -> None:
    tool = MockSearchTool()
    hit = _run(tool.run({"query": "what is the capital of France?"}))
    assert hit.is_error is False
    assert "Paris" in hit.content
    miss = _run(tool.run({"query": "price of tea in antarctica"}))
    assert "No results" in miss.content


def test_read_file_sandbox(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("codename: Bluebird", encoding="utf-8")
    tool = ReadFileTool(base_dir=tmp_path)

    ok = _run(tool.run({"path": "notes.txt"}))
    assert ok.is_error is False and "Bluebird" in ok.content

    missing = _run(tool.run({"path": "nope.txt"}))
    assert missing.is_error is True

    escape = _run(tool.run({"path": "../../etc/passwd"}))
    assert escape.is_error is True
    assert "sandbox" in escape.content.lower()
