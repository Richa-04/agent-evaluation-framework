"""Phase 3 acceptance demo: run the v1 suite, judge every trajectory, and show
per-task deterministic results + judge scores side by side.

Offline by default (scripted agent AND scripted judge) so it is deterministic and
needs no API key. The scripted judge returns realistic verdicts: high for the three
well-formed tasks, LOW on goal_completion and tool_selection for the expected-fail
'convert-speed-of-light' task (which skips the calculator). The demo asserts that
the broken task scores low and FLAGS loudly if a judge ever rates it well — exactly
the regression the brief asks us to guard against.

The same `judge_trajectory` / `judge_run_result` work against a real
`anthropic.AsyncAnthropic` client (Sonnet 4.6, temperature=0, structured output).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agent_eval.judge import judge_run_results
from agent_eval.models import RunConfig
from agent_eval.suite import load_suite, run_suite
from agent_eval.tools import build_default_tools

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tasks" / "suite_v1.json"
FILES_DIR = ROOT / "tasks" / "files"


# --- scripted AGENT (same behavior as the Phase 2 suite demo) ---------------
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


class _Router:
    """Routes create() to per-key scripted responses by a substring of the first
    user message (agent) or by task id embedded in the prompt (judge)."""

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


def _agent_client():
    return _Router(
        {
            "47 * 19": [_tool("c1", "calculator", {"expression": "47 * 19"}), _text("47 * 19 = 893.")],
            "capital of France": [_tool("c1", "search", {"query": "capital of France"}), _text("The capital of France is Paris.")],
            "notes.txt": [_tool("c1", "read_file", {"path": "notes.txt"}), _text("The project codename is Bluebird.")],
            "speed of light": [_tool("c1", "search", {"query": "speed of light"}), _text("The speed of light is about 300,000 km/s.")],
        }
    )


# --- scripted JUDGE (realistic verdicts; routed by task id) ------------------
def _verdict(gc, ts, eff, overall):
    payload = {
        "goal_completion": {"score": gc, "rationale": f"goal_completion={gc}"},
        "tool_selection": {"score": ts, "rationale": f"tool_selection={ts}"},
        "efficiency": {"score": eff, "rationale": f"efficiency={eff}"},
        "overall_rationale": overall,
    }
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=300, output_tokens=80),
    )


def _judge_client():
    return _Router(
        {
            "Task id: calc-multiply": [_verdict(5, 5, 5, "Correct, well-tooled, efficient.")],
            "Task id: search-capital": [_verdict(5, 5, 5, "Correct fact via search.")],
            "Task id: read-codename": [_verdict(5, 5, 4, "Read the file and reported the codename.")],
            "Task id: convert-speed-of-light": [
                _verdict(2, 2, 3, "Completed but skipped the required calculator and gave an imprecise value.")
            ],
        }
    )


def _print_report(results) -> bool:
    print("=== Per-task: deterministic checks + judge scores ===\n")
    flagged = False
    for r in results:
        det = "PASS" if r.deterministic_passed else "FAIL"
        js = r.judge_score
        print(f"[{det}] {r.task_id}  (terminal_state={r.trajectory.terminal_state.value})")
        print(
            f"    judge: goal_completion={js.goal_completion.score}  "
            f"tool_selection={js.tool_selection.score}  "
            f"efficiency={js.efficiency.score}  (avg={js.average:.2f})"
        )
        print(f"    rationale: {js.overall_rationale}")

        # Guard: the known-broken task must score low on goal + tool selection.
        if r.task_id == "convert-speed-of-light":
            if js.goal_completion.score >= 4 or js.tool_selection.score >= 4:
                flagged = True
                print(
                    "    !!! FLAG: judge rated the BROKEN task well — "
                    "goal_completion/tool_selection should be LOW. Investigate the judge."
                )
        print()
    return flagged


async def _amain() -> None:
    version, tasks = load_suite(SUITE_PATH)
    tasks_by_id = {t.id: t for t in tasks}
    tools = build_default_tools(read_base_dir=FILES_DIR)
    agent_config = RunConfig(model="claude-opus-4-8 (scripted offline)", max_tokens=2048)

    print(f"Suite '{version}': running {len(tasks)} tasks (scripted agent), then judging each "
          f"(scripted Sonnet 4.6 @ temperature=0, structured output)\n")

    results = await run_suite(tasks, tools, _agent_client(), agent_config)
    judged = await judge_run_results(results, tasks_by_id, _judge_client())

    flagged = _print_report(judged)
    if flagged:
        raise SystemExit("Judge rated a broken task well — stopping and flagging (see above).")
    print("OK: the three well-formed tasks score high; the expected-fail task scores "
          "low on goal_completion and tool_selection, as required.")


if __name__ == "__main__":
    asyncio.run(_amain())
