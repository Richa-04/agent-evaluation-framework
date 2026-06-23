# Agent Evaluation Framework

Run LLM **agents** (multi-step, tool-using) against versioned task suites, capture
their full execution **trajectories**, and score them with **both** a Claude-based
judge **and** deterministic assertions.

Deterministic checks reduce judge variance — the judge is never trusted alone.

## Status

Scaffold complete: project layout, core Pydantic v2 models, config/`.env` handling.
Feature phases (agent runner → task suite → judge → persistence → reporting) follow.

## Stack

Python 3.11+ · AsyncIO · `anthropic` · Pydantic v2 · SQLite · pandas · pytest · typer.
Model: `claude-opus-4-8` (Claude Opus 4.8) for both the agent under test and the judge.

## Quick start

```bash
pip install -e .
cp .env.example .env     # add your ANTHROPIC_API_KEY (real runs only)
pytest                   # tests mock the LLM — no network, deterministic
agent-eval --help        # CLI (lands in Phase 5)
```

See [CLAUDE.md](CLAUDE.md) for the full design, conventions, and build plan.
