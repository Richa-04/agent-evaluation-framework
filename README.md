# Agent Evaluation Framework

Run LLM **agents** (multi-step, tool-using) against versioned task suites, capture
their full execution **trajectories**, and score them with **both** a Claude-based
judge **and** deterministic assertions.

Deterministic checks reduce judge variance — the judge is never trusted alone.

## Status

All phases complete: agent runner + trajectory capture → versioned task suite +
deterministic assertions → LLM judge → SQLite persistence + regression tracking →
failure attribution + pandas reporting + CLI.

## CLI

```bash
agent-eval run     --db runs.sqlite --suite tasks/suite_v1.json --tag baseline  # real API
agent-eval versions --db runs.sqlite
agent-eval report  --db runs.sqlite --label <version-label>
agent-eval compare --db runs.sqlite --label-a <before> --label-b <after>
```

`run` calls the real API (needs `ANTHROPIC_API_KEY`); `versions`/`report`/`compare`
work offline on the stored SQLite data. See `examples/` for offline, mocked demos
of each phase.

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
