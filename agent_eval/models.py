"""Core Pydantic v2 data models for the Agent Evaluation Framework.

Design priorities (the hard parts called out in the project brief):

* The Trajectory captures tool **arguments** and intermediate state — not just a
  flat transcript. Every ToolCall records the args the model passed and the
  result it got back.
* Deterministic checks are first-class structured objects so they can be
  evaluated programmatically (Phase 2) and reduce judge variance.
* RunConfig pins model + sampling + versions for reproducibility, and is stored
  on every RunResult.

These models are pure data. Execution logic (the agent loop, the judge, storage)
lives in later phases and consumes/produces these types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TerminalState(str, Enum):
    """Why a trajectory ended."""

    COMPLETED = "completed"        # model ended its turn with a final answer
    MAX_STEPS = "max_steps"        # hit the step cap before finishing
    ERROR = "error"                # unrecoverable error during the run
    REFUSAL = "refusal"            # model declined (stop_reason == "refusal")
    NO_ANSWER = "no_answer"        # ended turn without producing a final answer


class CheckType(str, Enum):
    """Kinds of deterministic assertion. Evaluated in Phase 2."""

    TOOL_INVOKED = "tool_invoked"          # a named tool was called at least once
    TOOL_NOT_INVOKED = "tool_not_invoked"  # a named tool was never called
    TOOL_ARG_EQUALS = "tool_arg_equals"    # a tool was called with arg == expected
    TERMINAL_STATE = "terminal_state"      # trajectory ended in the expected state
    ANSWER_CONTAINS = "answer_contains"    # final answer contains a substring
    ANSWER_EQUALS = "answer_equals"        # final answer equals expected (normalized)
    ANSWER_REGEX = "answer_regex"          # final answer matches a regex
    MAX_STEPS_UNDER = "max_steps_under"    # step count <= expected (efficiency)


class JudgeDimension(str, Enum):
    """The rubric dimensions the LLM judge scores."""

    GOAL_COMPLETION = "goal_completion"
    TOOL_SELECTION = "tool_selection"
    EFFICIENCY = "efficiency"


# --------------------------------------------------------------------------- #
# Task + deterministic checks
# --------------------------------------------------------------------------- #
class DeterministicCheck(BaseModel):
    """A single programmatic assertion against a trajectory.

    Fields are interpreted per ``type`` (see CheckType). Unused fields stay None.
    Examples:
      * TOOL_INVOKED        -> tool_name="calculator"
      * TOOL_ARG_EQUALS     -> tool_name="calculator", arg_name="expression", expected="2+2"
      * ANSWER_CONTAINS     -> expected="4"
      * TERMINAL_STATE      -> expected="completed"
      * MAX_STEPS_UNDER     -> expected=3
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable id, unique within a task.")
    type: CheckType
    description: str = Field(..., description="Human-readable intent of the check.")

    # Targets (interpreted per type)
    tool_name: str | None = None
    arg_name: str | None = None
    expected: Any | None = Field(
        default=None, description="Expected value/substring/regex/state/count."
    )
    case_sensitive: bool = False


class Task(BaseModel):
    """One evaluation task: a prompt, the tools the agent may use, the expected
    outcome (ground truth), and the deterministic checks that gate it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    available_tools: list[str] = Field(
        default_factory=list, description="Names of tools exposed to the agent."
    )
    expected_outcome: str | None = Field(
        default=None, description="Ground-truth / expected answer, for the judge and checks."
    )
    deterministic_checks: list[DeterministicCheck] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, description="Per-task agent-loop cap.")

    # Suite/versioning metadata
    suite_version: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Trajectory (tool args + intermediate state, not just a transcript)
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """The result of executing one tool call."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="Stringified tool output fed back to the model.")
    is_error: bool = Field(default=False, description="True if the tool failed.")
    raw: dict[str, Any] | None = Field(
        default=None, description="Optional structured tool output for analysis."
    )


class ToolCall(BaseModel):
    """A single tool invocation: the name, the ARGUMENTS the model passed, and the
    result. This is the load-bearing record for evaluating tool-using agents."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Provider tool_use id, for matching result to call.")
    name: str
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments the model passed to the tool."
    )
    result: ToolResult | None = Field(
        default=None, description="None until the tool has been executed."
    )
    malformed: bool = Field(
        default=False, description="True if the call could not be parsed/validated."
    )


class TrajectoryStep(BaseModel):
    """One iteration of the agent loop: the model's reasoning/message for this turn
    plus the tool calls it issued (each with args and result)."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="0-based position in the trajectory.")
    thinking: str | None = Field(default=None, description="Model reasoning, if surfaced.")
    text: str | None = Field(default=None, description="Assistant message text for this turn.")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = Field(default=None, description="API stop_reason for this turn.")
    usage: dict[str, int] | None = Field(
        default=None, description="Token usage for this turn, if captured."
    )


class Trajectory(BaseModel):
    """The full ordered execution of an agent on a task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str | None = None
    terminal_state: TerminalState
    stop_reason: str | None = Field(default=None, description="Final API stop_reason.")
    error: str | None = Field(default=None, description="Error detail if terminal_state == error.")
    usage: dict[str, int] | None = Field(
        default=None, description="Aggregate token usage across the run."
    )
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def tool_calls(self) -> list[ToolCall]:
        """Flattened list of every tool call across all steps."""
        return [tc for step in self.steps for tc in step.tool_calls]

    def tool_names(self) -> list[str]:
        """Names of every tool invoked, in order (with repeats)."""
        return [tc.name for tc in self.tool_calls()]


# --------------------------------------------------------------------------- #
# Judge scoring
# --------------------------------------------------------------------------- #
class DimensionScore(BaseModel):
    """Score + written rationale for one rubric dimension."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=5, description="1 (worst) .. 5 (best).")
    rationale: str


class JudgeScore(BaseModel):
    """The LLM judge's structured verdict across the rubric dimensions."""

    model_config = ConfigDict(extra="forbid")

    goal_completion: DimensionScore
    tool_selection: DimensionScore
    efficiency: DimensionScore
    overall_rationale: str

    judge_model: str = Field(..., description="Model id used for the judge.")
    judge_version: str = Field(default="v1", description="Judge prompt/rubric version.")
    temperature: float | None = Field(
        default=None, description="Sampling temperature used for the judge (pinned for repeatability)."
    )
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw structured payload returned by the judge, logged for reproducibility."
    )

    @property
    def average(self) -> float:
        return (
            self.goal_completion.score
            + self.tool_selection.score
            + self.efficiency.score
        ) / 3.0


# --------------------------------------------------------------------------- #
# Deterministic results + run config + run result
# --------------------------------------------------------------------------- #
class CheckResult(BaseModel):
    """Outcome of evaluating one DeterministicCheck against a trajectory."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    type: CheckType
    description: str
    passed: bool
    detail: str | None = Field(default=None, description="Why it passed/failed.")


class RunConfig(BaseModel):
    """Everything pinned for reproducibility. Stored on every RunResult."""

    model_config = ConfigDict(extra="forbid")

    model: str
    # Optional: only sent to models that accept sampling params (not Opus 4.8).
    temperature: float | None = None
    effort: str | None = None          # output_config.effort, e.g. "low"
    thinking: str | None = None        # e.g. "adaptive" or None
    max_tokens: int = 4096
    max_steps: int = 10

    # Version tags for regression tracking (Phase 4).
    agent_version: str = "v1"
    suite_version: str | None = None


class RunResult(BaseModel):
    """The complete record of evaluating one agent on one task."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    trajectory: Trajectory
    deterministic_results: list[CheckResult] = Field(default_factory=list)
    judge_score: JudgeScore | None = None
    config: RunConfig
    suite_version: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def deterministic_passed(self) -> bool:
        """True only if every deterministic check passed (vacuously true if none)."""
        return all(r.passed for r in self.deterministic_results)
