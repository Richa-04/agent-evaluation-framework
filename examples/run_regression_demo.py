"""Phase 4 acceptance demo: run the v1 suite TWICE under different configs,
persist both to SQLite, and print a version comparison that calls out a regression.

Offline by default (scripted agent + scripted judge), deterministic, no API key.

  - Run A ("baseline"): a healthy agent — all four tasks pass.
  - Run B ("candidate"): the same agent EXCEPT it regresses on 'calc-multiply'
    (skips the calculator and returns a wrong answer).

Both runs are written to a real on-disk SQLite file, then re-opened to prove the
data persisted, then compared. The report must single out 'calc-multiply' as the
sole regression.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from agent_eval.judge import judge_run_results
from agent_eval.models import RunConfig
from agent_eval.storage import RunStore, compare_versions, format_comparison, make_version_label
from agent_eval.suite import format_suite_report, load_suite, run_suite
from agent_eval.tools import build_default_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tasks" / "suite_v1.json"
FILES_DIR = ROOT / "tasks" / "files"


def _text(t, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], stop_reason=stop,
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


# Healthy behavior for every task (used as-is in run A, and for 3/4 tasks in run B).
def _healthy_routes():
    return {
        "47 * 19": [_tool("c1", "calculator", {"expression": "47 * 19"}), _text("47 * 19 = 893.")],
        "capital of France": [_tool("c1", "search", {"query": "capital of France"}), _text("The capital of France is Paris.")],
        "notes.txt": [_tool("c1", "read_file", {"path": "notes.txt"}), _text("The project codename is Bluebird.")],
        # convert-speed-of-light done RIGHT: search, then calculator, precise value.
        "speed of light": [
            _tool("c1", "search", {"query": "speed of light"}),
            _tool("c2", "calculator", {"expression": "299792458 / 1000"}),
            _text("The speed of light is 299792.458 km/s."),
        ],
    }


def _agent_a():
    return _Router(_healthy_routes())


def _agent_b():
    routes = _healthy_routes()
    # REGRESSION: skip the calculator and give a wrong answer.
    routes["47 * 19"] = [_text("47 * 19 = 8003.")]
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


def _judge_all_good():
    return _Router({
        "Task id: calc-multiply": [_verdict(5, 5, 5, "Correct and well-tooled.")],
        "Task id: search-capital": [_verdict(5, 5, 5, "Correct via search.")],
        "Task id: read-codename": [_verdict(5, 5, 4, "Read the file, reported codename.")],
        "Task id: convert-speed-of-light": [_verdict(5, 5, 4, "Used both tools, precise value.")],
    })


def _judge_b():
    routes = _judge_all_good()._routes.copy()
    # calc-multiply now scores low (skipped tool, wrong answer).
    routes["Task id: calc-multiply"] = [_verdict(1, 1, 2, "Wrong answer; skipped the calculator.")]
    return _Router(routes)


async def _run(agent_client, judge_client, tasks, tasks_by_id, tools):
    config = RunConfig(model="claude-opus-4-8 (scripted offline)", max_tokens=2048)
    results = await run_suite(tasks, tools, agent_client, config)
    return await judge_run_results(results, tasks_by_id, judge_client)


async def _amain() -> None:
    version, tasks = load_suite(SUITE_PATH)
    tasks_by_id = {t.id: t for t in tasks}
    tools = build_default_tools(read_base_dir=FILES_DIR)

    label_a = make_version_label("claude-opus-4-8", version,
                                 timestamp=datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc), tag="baseline")
    label_b = make_version_label("claude-opus-4-8", version,
                                 timestamp=datetime(2026, 6, 23, 13, 0, 0, tzinfo=timezone.utc), tag="candidate")

    judged_a = await _run(_agent_a(), _judge_all_good(), tasks, tasks_by_id, tools)
    judged_b = await _run(_agent_b(), _judge_b(), tasks, tasks_by_id, tools)

    # Persist both runs to a real on-disk SQLite file.
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        store = RunStore(path)
        store.save_runs(judged_a, label_a)
        store.save_runs(judged_b, label_b)
        store.close()
        print(f"Persisted 2 versions x {len(tasks)} tasks to SQLite at {path}\n")

        # Re-open to prove persistence, then compare.
        store = RunStore(path)
        print("Stored versions:")
        for v in store.list_versions():
            print(f"  {v.version_label}  ({v.task_count} tasks, model={v.agent_model}, suite={v.suite_version})")
        print()
        print("Run B per-task deterministic report:")
        print(format_suite_report(store.get_runs(label_b)))
        print()
        report = format_comparison(compare_versions(store, label_a, label_b))
        store.close()
        print(report)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    asyncio.run(_amain())
