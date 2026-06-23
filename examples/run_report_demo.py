"""Phase 5 acceptance demo: failure attribution + pandas summary report.

Offline by default (scripted agent + scripted judge), deterministic, no API key.

Runs the v1 suite where the expected-fail 'convert-speed-of-light' task skips the
required calculator. Prints (1) failure attribution — the failed task classified
with a specific reason — and (2) the pandas summary report (pass rate, average
judge scores per dimension, failure-type breakdown).

It also persists a healthy 'baseline' run and this 'candidate' run to a SQLite
file (default ./demo_runs.sqlite, or argv[1]) so you can then drive the CLI:

    agent-eval versions --db demo_runs.sqlite
    agent-eval report   --db demo_runs.sqlite --label <candidate-label>
    agent-eval compare  --db demo_runs.sqlite --label-a <baseline> --label-b <candidate>
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from agent_eval.judge import judge_run_results
from agent_eval.models import RunConfig
from agent_eval.reporting import attribute_failures, format_report
from agent_eval.storage import RunStore, make_version_label
from agent_eval.suite import load_suite, run_suite
from agent_eval.tools import build_default_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tasks" / "suite_v1.json"
FILES_DIR = ROOT / "tasks" / "files"


def _text(t):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=20, output_tokens=10))


def _tool(call_id, name, args):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)],
                           stop_reason="tool_use", usage=SimpleNamespace(input_tokens=20, output_tokens=10))


class _Router:
    def __init__(self, routes):
        self._routes = routes
        self._counters = {k: 0 for k in routes}
        self.messages = self

    async def create(self, **kwargs):
        text = ""
        for m in kwargs.get("messages", []):
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, str):
                text = c
                break
        for key, scripts in self._routes.items():
            if key in text:
                i = self._counters[key]
                self._counters[key] += 1
                return scripts[i]
        raise AssertionError(f"no route matched: {text[:80]!r}")


def _base_routes():
    return {
        "47 * 19": [_tool("c1", "calculator", {"expression": "47 * 19"}), _text("47 * 19 = 893.")],
        "capital of France": [_tool("c1", "search", {"query": "capital of France"}), _text("The capital of France is Paris.")],
        "notes.txt": [_tool("c1", "read_file", {"path": "notes.txt"}), _text("The project codename is Bluebird.")],
    }


def _agent_candidate():
    routes = _base_routes()
    # convert-speed-of-light: calls search, SKIPS the required calculator.
    routes["speed of light"] = [_tool("c1", "search", {"query": "speed of light"}),
                                _text("The speed of light is about 300,000 km/s.")]
    return _Router(routes)


def _agent_baseline():
    routes = _base_routes()
    # convert done RIGHT: search, then calculator, precise value.
    routes["speed of light"] = [
        _tool("c1", "search", {"query": "speed of light"}),
        _tool("c2", "calculator", {"expression": "299792458 / 1000"}),
        _text("The speed of light is 299792.458 km/s."),
    ]
    return _Router(routes)


def _verdict(gc, ts, eff, overall):
    payload = {
        "goal_completion": {"score": gc, "rationale": f"goal={gc}"},
        "tool_selection": {"score": ts, "rationale": f"tool={ts}"},
        "efficiency": {"score": eff, "rationale": f"eff={eff}"},
        "overall_rationale": overall,
    }
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))],
                           stop_reason="end_turn", usage=SimpleNamespace(input_tokens=300, output_tokens=80))


def _judge_candidate():
    return _Router({
        "Task id: calc-multiply": [_verdict(5, 5, 5, "Correct and well-tooled.")],
        "Task id: search-capital": [_verdict(5, 5, 5, "Correct via search.")],
        "Task id: read-codename": [_verdict(5, 5, 4, "Read the file, reported codename.")],
        "Task id: convert-speed-of-light": [_verdict(2, 2, 3, "Skipped the required calculator; imprecise.")],
    })


def _judge_baseline():
    return _Router({
        "Task id: calc-multiply": [_verdict(5, 5, 5, "ok")],
        "Task id: search-capital": [_verdict(5, 5, 5, "ok")],
        "Task id: read-codename": [_verdict(5, 5, 4, "ok")],
        "Task id: convert-speed-of-light": [_verdict(5, 5, 4, "Used both tools, precise value.")],
    })


async def _amain(db_path: str) -> None:
    version, tasks = load_suite(SUITE_PATH)
    tasks_by_id = {t.id: t for t in tasks}
    tools = build_default_tools(read_base_dir=FILES_DIR)
    config = RunConfig(model="claude-opus-4-8 (scripted offline)", max_tokens=2048)

    candidate = await judge_run_results(
        await run_suite(tasks, tools, _agent_candidate(), config), tasks_by_id, _judge_candidate()
    )
    baseline = await judge_run_results(
        await run_suite(tasks, tools, _agent_baseline(), config), tasks_by_id, _judge_baseline()
    )

    # 1) Failure attribution for the candidate run.
    print("=== Failure attribution (candidate run) ===")
    for failure in attribute_failures(candidate, tasks_by_id):
        if not failure.failed:
            continue
        print(f"\n[{failure.primary_category.value}] {failure.task_id}")
        for finding in failure.findings:
            where = "" if finding.step_index is None else f" (step {finding.step_index})"
            print(f"    - {finding.category.value}{where}: {finding.detail}")

    # 2) pandas summary report.
    print("\n")
    print(format_report(candidate, tasks_by_id))

    # Persist both versions for the CLI.
    base_label = make_version_label("claude-opus-4-8", version,
                                    timestamp=datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc), tag="baseline")
    cand_label = make_version_label("claude-opus-4-8", version,
                                    timestamp=datetime(2026, 6, 23, 13, 0, 0, tzinfo=timezone.utc), tag="candidate")
    with RunStore(db_path) as store:
        store.save_runs(baseline, base_label)
        store.save_runs(candidate, cand_label)

    print(f"\nPersisted baseline + candidate to {db_path}")
    print("Try the CLI:")
    print(f"  agent-eval versions --db {db_path}")
    print(f"  agent-eval report   --db {db_path} --label '{cand_label}'")
    print(f"  agent-eval compare  --db {db_path} --label-a '{base_label}' --label-b '{cand_label}'")


if __name__ == "__main__":
    asyncio.run(_amain(sys.argv[1] if len(sys.argv) > 1 else "demo_runs.sqlite"))
