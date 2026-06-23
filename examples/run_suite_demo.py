"""Phase 2 acceptance demo: run the versioned suite and print the pass/fail report.

Offline by default (scripted model) so it is deterministic and needs no API key.
The 'convert-speed-of-light' task is scripted to fail (the agent skips the
calculator), demonstrating that the deterministic checks catch real failures.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent_eval.models import RunConfig
from agent_eval.suite import format_suite_report, load_suite, run_suite
from agent_eval.tools import build_default_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tasks" / "suite_v1.json"
FILES_DIR = ROOT / "tasks" / "files"


def _text(t, stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=t)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=20, output_tokens=10),
    )


def _tool(call_id, name, args):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=20, output_tokens=10),
    )


class _ScriptedModel:
    """Routes by a substring of the first user message (see tests/fakes.py)."""

    def __init__(self):
        self._routes = {
            "47 * 19": [_tool("c1", "calculator", {"expression": "47 * 19"}), _text("47 * 19 = 893.")],
            "capital of France": [_tool("c1", "search", {"query": "capital of France"}), _text("The capital of France is Paris.")],
            "notes.txt": [_tool("c1", "read_file", {"path": "notes.txt"}), _text("The project codename is Bluebird.")],
            "speed of light": [_tool("c1", "search", {"query": "speed of light"}), _text("The speed of light is about 300,000 km/s.")],
        }
        self._counters = {k: 0 for k in self._routes}
        self.messages = self

    async def create(self, **kwargs):
        text = ""
        for m in kwargs.get("messages", []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                text = m["content"]
                break
        for key, scripts in self._routes.items():
            if key in text:
                i = self._counters[key]
                self._counters[key] += 1
                return scripts[i]
        raise AssertionError(f"no route for: {text!r}")


async def _amain():
    version, tasks = load_suite(SUITE_PATH)
    tools = build_default_tools(read_base_dir=FILES_DIR)
    config = RunConfig(model="claude-opus-4-8 (scripted offline)", max_tokens=2048)
    print(f"Running suite '{version}' with {len(tasks)} tasks (scripted offline)\n")
    results = await run_suite(tasks, tools, _ScriptedModel(), config)
    print(format_suite_report(results))


if __name__ == "__main__":
    asyncio.run(_amain())
