# CLAUDE.md — Agent Evaluation Framework

## Goal

A Python tool that runs LLM **agents** (multi-step, tool-using) against versioned
task suites, captures their full execution **trajectories**, and scores them using
**both** an LLM judge **and** deterministic assertions. The point is to evaluate
agents — not single LLM responses — with reproducible, regression-trackable results.

Deterministic assertions exist to reduce judge variance. We never rely on the judge alone.

## Locked tech stack (do NOT add dependencies without asking)

- Python 3.11+, AsyncIO
- Anthropic Python SDK (`anthropic`) — for both the agent under test and the judge
- Pydantic v2 — all data models
- SQLite (stdlib `sqlite3`) — run storage
- pandas — analysis/reporting
- pytest — tests (LLM calls mocked so tests are deterministic)
- typer — CLI

## Model

Confirmed against Anthropic's current docs.

- **Agent under test:** **Claude Opus 4.8**, `claude-opus-4-8` (1M context, 128K max output) — most capable widely released model.
- **Judge:** **Claude Sonnet 4.6**, `claude-sonnet-4-6`, pinned at **`temperature=0`** for repeatability.

> ⚠️ **Opus 4.8 API surface:** `temperature` / `top_p` / `top_k` are removed (400 if sent),
> and extended-thinking `budget_tokens` is removed — thinking is adaptive only
> (`thinking={"type": "adaptive"}`). That's why the **judge runs on Sonnet 4.6** instead:
> it accepts `temperature=0`, giving literal low-temp determinism as the brief requires.
> The judge also returns **structured output** (`output_config.format`, the `JudgeScore`
> schema) as a second repeatability lever. `RunConfig.temperature` is **optional and only
> sent for models that accept it** (set for the judge, left `None` for the Opus 4.8 agent).

## Folder structure

```
agent_eval/
  __init__.py
  config.py            # settings, model string, .env loading (no extra deps)
  models.py            # core Pydantic v2 models (see below)
  tools/               # Phase 1 — calculator, mock search, read_file
  agent/               # Phase 1 — agent loop / runner
  suite/               # Phase 2 — task suite loading + deterministic checks
  judge/               # Phase 3 — Claude-based judge
  storage/             # Phase 4 — SQLite persistence + regression compare
  reporting/           # Phase 5 — pandas reports + failure attribution
  cli.py               # Phase 5 — typer CLI
tasks/                 # versioned task suite definitions
tests/                 # pytest, mocked LLM
pyproject.toml
.env.example
README.md
CLAUDE.md
```

## Core models (`agent_eval/models.py`)

- **Task** — `id`, `prompt`, `available_tools`, `expected_outcome`, `deterministic_checks`, `max_steps`, version/metadata.
- **DeterministicCheck** — a single programmatic assertion (type + targets). Evaluated in Phase 2.
- **ToolCall / ToolResult** — a tool invocation with its **arguments** and result. Captures intermediate state, not just a transcript.
- **TrajectoryStep** — one agent-loop iteration: model reasoning/text + the tool calls it made (with args + results) + the API stop reason.
- **Trajectory** — ordered steps, final answer, terminal state, stop reason, usage.
- **DimensionScore / JudgeScore** — per-dimension score + rationale (goal completion, tool-selection accuracy, trajectory efficiency) + overall rationale + judge model/version.
- **CheckResult** — outcome of one DeterministicCheck (passed + detail).
- **RunConfig** — pinned model, temperature (optional), max_steps, effort, agent/suite versions — logged for reproducibility.
- **RunResult** — task id, trajectory, deterministic results, judge score, version tags, config, timestamp.

## Conventions

- Async throughout (`async def`, `AsyncAnthropic`).
- All data crossing a boundary is a Pydantic v2 model; serialize via `.model_dump()` / `.model_dump_json()`.
- Pin model + config in every `RunResult`. Log everything for reproducibility.
- Tests mock the Anthropic client — no network, fully deterministic.
- The agent loop must: enforce a max-step cap, detect terminal state, and handle malformed/failed tool calls gracefully.
- The judge prompt must include the task goal, the full trajectory (with tool args), and the explicit rubric, and must return structured JSON.

## How to run

```bash
# install (editable)
pip install -e .

# set the API key (real runs only; tests are mocked)
cp .env.example .env   # then edit ANTHROPIC_API_KEY

# tests
pytest

# CLI (Phase 5)
agent-eval --help
```

## Build plan (phased; stop for review at the end of each)

- **Scaffold** ✅ — CLAUDE.md, repo skeleton, core Pydantic models.
- **Phase 1** ✅ — Agent runner: async loop with calculator/search/read_file, full trajectory capture.
- **Phase 2** ✅ — Versioned task suite + deterministic assertions (incl. an expected-fail task).
- **Phase 3** ✅ — LLM judge: Sonnet 4.6 @ temp 0, structured output, 3-dimension rubric; attaches to RunResult alongside deterministic results.
- **Phase 4** ✅ — SQLite persistence (RunStore, version labels) + version regression comparison (per-dimension deltas, deterministic pass/fail changes, regressed/improved classification).
- **Phase 5** ✅ — Failure attribution (wrong tool / malformed args / premature termination / incorrect result / agent error) + pandas summary report + typer CLI (`run` / `versions` / `report` / `compare`).

## Resolved decisions

1. **Judge determinism vs. model choice.** Opus 4.8 can't take `temperature`.
   **Resolved:** the judge runs on **Sonnet 4.6 at `temperature=0`** (the agent stays on
   Opus 4.8). This gives the literal low-temperature determinism the brief asks for, with
   structured output as a second repeatability lever.
