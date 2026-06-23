"""Phase 1 tests: the async agent loop, with the LLM mocked via ScriptedClient.

Covers: single tool call -> answer, argument capture, max-step cap, terminal-state
detection (completed / refusal / error / no_answer), and graceful handling of
malformed (unknown tool / bad input) and failing tool calls.
"""

from __future__ import annotations

import asyncio

from agent_eval.agent import run_agent
from agent_eval.models import RunConfig, TerminalState, Task
from agent_eval.tools import CalculatorTool

from .fakes import (
    RaisingClient,
    ScriptedClient,
    assistant_raw,
    assistant_text,
    assistant_tool,
    make_usage,
)

CONFIG = RunConfig(model="claude-opus-4-8", thinking="adaptive", max_steps=10)


def _run(task, tools, client):
    return asyncio.run(run_agent(task, tools, client, CONFIG))


def test_single_tool_then_answer_captures_arguments() -> None:
    task = Task(
        id="calc",
        prompt="What is 47 * 19? Use the calculator tool.",
        available_tools=["calculator"],
        max_steps=5,
    )
    client = ScriptedClient(
        {
            "47 * 19": [
                assistant_tool(
                    [("c1", "calculator", {"expression": "47 * 19"})],
                    thinking="I'll compute this with the calculator.",
                ),
                assistant_text("47 * 19 = 893."),
            ]
        }
    )

    traj = _run(task, [CalculatorTool()], client)

    assert traj.terminal_state is TerminalState.COMPLETED
    assert traj.stop_reason == "end_turn"
    assert traj.step_count == 2
    assert traj.final_answer == "47 * 19 = 893."

    calls = traj.tool_calls()
    assert len(calls) == 1
    # The load-bearing assertion: arguments are captured, plus the executed result.
    assert calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "47 * 19"}
    assert calls[0].malformed is False
    assert calls[0].result is not None
    assert calls[0].result.content == "893"
    assert calls[0].result.is_error is False
    # Reasoning captured on the step.
    assert traj.steps[0].thinking == "I'll compute this with the calculator."
    # Usage aggregated across both turns.
    assert traj.usage == {"input_tokens": 20, "output_tokens": 10}


def test_max_steps_cap_terminates() -> None:
    # The model keeps calling tools and never finishes; the cap must stop it.
    task = Task(
        id="loop",
        prompt="Keep computing 1 + 1 forever.",
        available_tools=["calculator"],
        max_steps=3,
    )
    tool_turn = assistant_tool([("c", "calculator", {"expression": "1 + 1"})])
    client = ScriptedClient({"Keep computing": [tool_turn, tool_turn, tool_turn]})

    traj = _run(task, [CalculatorTool()], client)

    assert traj.terminal_state is TerminalState.MAX_STEPS
    assert traj.step_count == 3
    assert traj.final_answer is None


def test_unknown_tool_is_malformed_but_recovers() -> None:
    task = Task(
        id="badtool",
        prompt="Do the thing.",
        available_tools=["calculator"],
        max_steps=5,
    )
    client = ScriptedClient(
        {
            "Do the thing": [
                assistant_tool([("c1", "nonexistent_tool", {"x": 1})]),
                assistant_text("Okay, I'll answer directly: 42."),
            ]
        }
    )

    traj = _run(task, [CalculatorTool()], client)

    assert traj.terminal_state is TerminalState.COMPLETED
    bad_call = traj.steps[0].tool_calls[0]
    assert bad_call.malformed is True
    assert bad_call.result is not None and bad_call.result.is_error is True
    assert "unknown tool" in bad_call.result.content.lower()
    assert traj.final_answer == "Okay, I'll answer directly: 42."


def test_failing_tool_call_returns_error_result_not_malformed() -> None:
    # Tool exists and input is a valid object, but the expression is invalid.
    task = Task(
        id="badexpr",
        prompt="Compute the broken expression.",
        available_tools=["calculator"],
        max_steps=5,
    )
    client = ScriptedClient(
        {
            "broken expression": [
                assistant_tool([("c1", "calculator", {"expression": "2 +"})]),
                assistant_text("That expression was invalid."),
            ]
        }
    )

    traj = _run(task, [CalculatorTool()], client)

    call = traj.steps[0].tool_calls[0]
    assert call.malformed is False  # tool & input shape were fine
    assert call.result is not None and call.result.is_error is True
    assert "error" in call.result.content.lower()
    assert traj.terminal_state is TerminalState.COMPLETED


def test_non_object_tool_input_is_malformed() -> None:
    task = Task(id="badinput", prompt="weird input case", available_tools=["calculator"], max_steps=3)
    client = ScriptedClient(
        {
            "weird input": [
                assistant_tool([("c1", "calculator", "47 * 19")]),  # input is a str, not dict
                assistant_text("Recovered."),
            ]
        }
    )

    traj = _run(task, [CalculatorTool()], client)
    call = traj.steps[0].tool_calls[0]
    assert call.malformed is True
    assert call.arguments == {}
    assert call.result is not None and call.result.is_error is True


def test_refusal_terminal_state() -> None:
    task = Task(id="refuse", prompt="please refuse this", available_tools=["calculator"], max_steps=3)
    client = ScriptedClient(
        {
            "refuse this": [
                assistant_raw(
                    [type("B", (), {"type": "text", "text": "I can't help with that."})()],
                    stop_reason="refusal",
                    usage=make_usage(),
                )
            ]
        }
    )

    traj = _run(task, [CalculatorTool()], client)
    assert traj.terminal_state is TerminalState.REFUSAL
    assert traj.stop_reason == "refusal"


def test_no_answer_terminal_state() -> None:
    # end_turn but no text content -> NO_ANSWER
    task = Task(id="empty", prompt="empty turn case", available_tools=["calculator"], max_steps=3)
    client = ScriptedClient({"empty turn": [assistant_raw([], stop_reason="end_turn")]})

    traj = _run(task, [CalculatorTool()], client)
    assert traj.terminal_state is TerminalState.NO_ANSWER
    assert traj.final_answer is None


def test_client_exception_yields_error_state() -> None:
    task = Task(id="boom", prompt="anything", available_tools=["calculator"], max_steps=3)
    traj = _run(task, [CalculatorTool()], RaisingClient(ValueError("kaboom")))

    assert traj.terminal_state is TerminalState.ERROR
    assert traj.error is not None and "kaboom" in traj.error
    assert traj.stop_reason == "error"


def test_temperature_only_sent_when_set() -> None:
    # Opus-style config: no temperature -> not in kwargs. Sonnet-style: present.
    task = Task(id="cfg", prompt="config check", available_tools=["calculator"], max_steps=2)

    opus = ScriptedClient({"config check": [assistant_text("done")]})
    asyncio.run(run_agent(task, [CalculatorTool()], opus, RunConfig(model="claude-opus-4-8")))
    assert "temperature" not in opus.calls[0]

    sonnet = ScriptedClient({"config check": [assistant_text("done")]})
    asyncio.run(
        run_agent(task, [CalculatorTool()], sonnet, RunConfig(model="claude-sonnet-4-6", temperature=0.0))
    )
    assert sonnet.calls[0]["temperature"] == 0.0
