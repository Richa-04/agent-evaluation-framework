"""Configuration: model string, defaults, and lightweight .env loading.

We avoid adding a dependency for .env handling (e.g. python-dotenv) and parse a
minimal KEY=VALUE file ourselves. The Anthropic SDK reads ANTHROPIC_API_KEY from
the environment automatically once it is set.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Models ----------------------------------------------------------------
# Agent under test: current most-capable Opus generation (confirmed against docs).
# claude-opus-5 runs adaptive thinking and REJECTS temperature/top_p/top_k (400).
DEFAULT_MODEL = "claude-opus-5"

# Judge: Sonnet 5 (cheaper than Opus, ample for scoring). NOTE: unlike the older
# Sonnet 4.6, claude-sonnet-5 also REJECTS temperature/top_p/top_k (400) — the
# whole current generation removed sampling params. So the judge's determinism no
# longer comes from temperature=0; it comes from (a) structured output
# (output_config.format json_schema) and (b) low effort (output_config.effort).
# DEFAULT_JUDGE_TEMPERATURE is therefore None (not sent). Decision recorded in CLAUDE.md.
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_JUDGE_TEMPERATURE = None

# Default token ceilings. Judge output is small; agent turns are bounded.
DEFAULT_AGENT_MAX_TOKENS = 4096
DEFAULT_JUDGE_MAX_TOKENS = 2048

# Default agent-loop cap. Tasks may override via Task.max_steps.
DEFAULT_MAX_STEPS = 10

# Effort levels (output_config.effort). Low effort favors repeatability for the judge.
DEFAULT_JUDGE_EFFORT = "low"

ENV_API_KEY = "ANTHROPIC_API_KEY"


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Minimal parser (no external dependency): ignores blank lines and `#` comments,
    strips surrounding quotes. Existing env vars are not overwritten unless
    ``override=True``. Returns the parsed mapping. Missing file is a no-op.
    """
    env_path = Path(path)
    parsed: dict[str, str] = {}
    if not env_path.is_file():
        return parsed

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def get_api_key(*, load_env: bool = True) -> str:
    """Return the Anthropic API key, loading .env first if requested.

    Raises RuntimeError if the key is absent — callers doing real runs need it;
    the test suite mocks the client and never calls this.
    """
    if load_env:
        load_dotenv()
    key = os.environ.get(ENV_API_KEY)
    if not key:
        raise RuntimeError(
            f"{ENV_API_KEY} is not set. Copy .env.example to .env and add your key, "
            f"or export {ENV_API_KEY} in your environment."
        )
    return key
