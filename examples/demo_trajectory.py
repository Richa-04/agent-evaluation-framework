"""Phase 1 acceptance demo: run one task end-to-end and print the full trajectory.

Runs OFFLINE by default using a small scripted model (deterministic, no API key
needed) so the trajectory is reproducible. The exact same `run_agent` works with
a real `anthropic.AsyncAnthropic` client — set ANTHROPIC_API_KEY and pass
`--live` to use it.

    python examples/demo_trajectory.py          # scripted (offline)
    python examples/demo_trajectory.py --live    # real Opus 4.8 agent
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from agent_eval.agent import run_agent
from agent_eval.display import format_trajectory
from agent_eval.models import RunConfig, Task
from agent_eval.tools import build_default_tools

TASK = Task(
    id="demo-speed-of-light",
    prompt=(
        "Use the search tool to find the speed of light, then use the calculator "
        "to divide 299792458 by 1000, and report the value in km/s."
    ),
    available_tools=["search", "calculator"],
    expected_outcome="299792.458 km/s",
    max_steps=6,
)


class _ScriptedModel:
    """Minimal offline stand-in for the Anthropic client (see tests/fakes.py for
    the fuller version used in the test suite)."""

    def __init__(self) -> None:
        self._turns = iter(
            [
                SimpleNamespace(
                    stop_reason="tool_use",
                    usage=SimpleNamespace(input_tokens=180, output_tokens=40),
                    content=[
                        SimpleNamespace(type="thinking", thinking="First I'll look up the speed of light."),
                        SimpleNamespace(type="tool_use", id="t1", name="search", input={"query": "speed of light"}),
                    ],
                ),
                SimpleNamespace(
                    stop_reason="tool_use",
                    usage=SimpleNamespace(input_tokens=210, output_tokens=35),
                    content=[
                        SimpleNamespace(type="thinking", thinking="Now convert m/s to km/s by dividing by 1000."),
                        SimpleNamespace(type="tool_use", id="t2", name="calculator", input={"expression": "299792458 / 1000"}),
                    ],
                ),
                SimpleNamespace(
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=230, output_tokens=20),
                    content=[SimpleNamespace(type="text", text="The speed of light is 299792.458 km/s.")],
                ),
            ]
        )
        self.messages = self

    async def create(self, **_kwargs: object) -> object:
        return next(self._turns)


async def _amain(live: bool) -> None:
    tools = build_default_tools()
    if live:
        from anthropic import AsyncAnthropic  # imported lazily; only needed for --live

        from agent_eval.config import get_api_key

        client = AsyncAnthropic(api_key=get_api_key())
        config = RunConfig(model="claude-opus-4-8", thinking="adaptive", max_tokens=2048, max_steps=TASK.max_steps)
        print("(running against the live Opus 4.8 agent)\n")
    else:
        client = _ScriptedModel()
        config = RunConfig(model="claude-opus-4-8 (scripted offline)", max_tokens=2048, max_steps=TASK.max_steps)
        print("(running offline with a scripted model — set --live for the real API)\n")

    trajectory = await run_agent(TASK, tools, client, config)
    print(format_trajectory(trajectory))


if __name__ == "__main__":
    asyncio.run(_amain(live="--live" in sys.argv))
