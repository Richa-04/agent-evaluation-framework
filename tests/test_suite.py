"""Phase 2 end-to-end test: load the versioned suite, run it with a mocked LLM,
and confirm the deterministic checks pass the good tasks and FAIL the bad one.

The scripted agent deliberately mishandles the multi-step 'convert-speed-of-light'
task (answers from memory, skips the calculator) so we can verify the checks
actually catch a real agent failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_eval.models import RunConfig
from agent_eval.suite import format_suite_report, load_suite, run_suite
from agent_eval.tools import build_default_tools

from .fakes import ScriptedClient, assistant_text, assistant_tool

SUITE_PATH = Path(__file__).resolve().parents[1] / "tasks" / "suite_v1.json"
FILES_DIR = Path(__file__).resolve().parents[1] / "tasks" / "files"

CONFIG = RunConfig(model="claude-opus-5", thinking="adaptive", max_tokens=2048)


def _scripted_client() -> ScriptedClient:
    return ScriptedClient(
        {
            # calc-multiply: correct two-turn solution -> PASS
            "47 * 19": [
                assistant_tool([("c1", "calculator", {"expression": "47 * 19"})]),
                assistant_text("47 * 19 = 893."),
            ],
            # search-capital: correct -> PASS
            "capital of France": [
                assistant_tool([("c1", "search", {"query": "capital of France"})]),
                assistant_text("The capital of France is Paris."),
            ],
            # read-codename: correct -> PASS
            "notes.txt": [
                assistant_tool([("c1", "read_file", {"path": "notes.txt"})]),
                assistant_text("The project codename is Bluebird."),
            ],
            # convert-speed-of-light: skips the calculator, answers a rounded value
            # from memory -> the calc + precise-value checks must FAIL.
            "speed of light": [
                assistant_tool([("c1", "search", {"query": "speed of light"})]),
                assistant_text("The speed of light is about 300,000 km/s."),
            ],
        }
    )


def test_loader_reads_version_and_tasks() -> None:
    version, tasks = load_suite(SUITE_PATH)
    assert version == "v1"
    ids = [t.id for t in tasks]
    assert ids == ["calc-multiply", "search-capital", "read-codename", "convert-speed-of-light"]
    assert all(t.suite_version == "v1" for t in tasks)


def test_suite_run_passes_good_tasks_and_fails_the_bad_one() -> None:
    _version, tasks = load_suite(SUITE_PATH)
    tools = build_default_tools(read_base_dir=FILES_DIR)
    results = asyncio.run(run_suite(tasks, tools, _scripted_client(), CONFIG))

    by_id = {r.task_id: r for r in results}

    # Three well-formed tasks pass every deterministic check.
    assert by_id["calc-multiply"].deterministic_passed is True
    assert by_id["search-capital"].deterministic_passed is True
    assert by_id["read-codename"].deterministic_passed is True

    # The multi-step task fails — and we assert WHICH checks caught it.
    failing = by_id["convert-speed-of-light"]
    assert failing.deterministic_passed is False
    failed_ids = {c.check_id for c in failing.deterministic_results if not c.passed}
    assert "calc" in failed_ids       # calculator was never invoked
    assert "answer" in failed_ids     # precise value 299792.458 absent
    passed_ids = {c.check_id for c in failing.deterministic_results if c.passed}
    assert "search" in passed_ids     # search WAS used
    assert "done" in passed_ids       # it did complete


def test_calc_task_arg_check_passes() -> None:
    _version, tasks = load_suite(SUITE_PATH)
    tools = build_default_tools(read_base_dir=FILES_DIR)
    results = asyncio.run(run_suite(tasks, tools, _scripted_client(), CONFIG))
    calc = next(r for r in results if r.task_id == "calc-multiply")
    arg_check = next(c for c in calc.deterministic_results if c.check_id == "arg")
    assert arg_check.passed is True


def test_report_renders_pass_and_fail() -> None:
    _version, tasks = load_suite(SUITE_PATH)
    tools = build_default_tools(read_base_dir=FILES_DIR)
    results = asyncio.run(run_suite(tasks, tools, _scripted_client(), CONFIG))
    report = format_suite_report(results)
    assert "3/4 tasks passed" in report
    assert "[FAIL] convert-speed-of-light" in report
    assert "[PASS] calc-multiply" in report
