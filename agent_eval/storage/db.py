"""SQLite persistence for RunResults.

Each RunResult is stored as a row tagged with a **version label** (capturing the
agent model + suite version + run timestamp) plus indexed columns for the
deterministic outcome and the three judge dimension scores, so runs are queryable
and comparable across versions. The full RunResult is preserved as JSON in
``result_json`` for lossless reconstruction.

``(version_label, task_id)`` is unique — re-saving a task under the same label
replaces the prior row, so a version label always names one coherent suite run.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from ..models import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL,
    version_label        TEXT NOT NULL,
    task_id              TEXT NOT NULL,
    agent_model          TEXT,
    suite_version        TEXT,
    created_at           TEXT,
    deterministic_passed INTEGER NOT NULL,
    goal_completion      INTEGER,
    tool_selection       INTEGER,
    efficiency           INTEGER,
    judge_average        REAL,
    result_json          TEXT NOT NULL,
    UNIQUE(version_label, task_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_version ON runs(version_label);
"""


def make_version_label(
    agent_model: str,
    suite_version: str | None,
    *,
    timestamp: datetime | None = None,
    tag: str | None = None,
) -> str:
    """Build a version label capturing agent model + suite version + timestamp.

    Optional ``tag`` adds a human-readable suffix (e.g. "baseline"/"candidate").
    """
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    parts = [agent_model, suite_version or "none", ts]
    if tag:
        parts.append(tag)
    return "/".join(parts)


class VersionInfo(BaseModel):
    """Summary of one stored version label."""

    version_label: str
    agent_model: str | None
    suite_version: str | None
    task_count: int
    first_created_at: str | None


class RunStore:
    """A SQLite-backed store for RunResults.

    Holds a single connection for its lifetime (so ``:memory:`` works across
    calls). Usable as a context manager.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes --------------------------------------------------------------
    def save_run(self, result: RunResult, version_label: str) -> None:
        """Insert (or replace) one RunResult under ``version_label``."""
        js = result.judge_score
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, version_label, task_id, agent_model, suite_version,
                created_at, deterministic_passed, goal_completion, tool_selection,
                efficiency, judge_average, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version_label, task_id) DO UPDATE SET
                run_id=excluded.run_id,
                agent_model=excluded.agent_model,
                suite_version=excluded.suite_version,
                created_at=excluded.created_at,
                deterministic_passed=excluded.deterministic_passed,
                goal_completion=excluded.goal_completion,
                tool_selection=excluded.tool_selection,
                efficiency=excluded.efficiency,
                judge_average=excluded.judge_average,
                result_json=excluded.result_json
            """,
            (
                result.run_id,
                version_label,
                result.task_id,
                result.config.model,
                result.suite_version or result.config.suite_version,
                result.created_at.isoformat(),
                int(result.deterministic_passed),
                js.goal_completion.score if js else None,
                js.tool_selection.score if js else None,
                js.efficiency.score if js else None,
                js.average if js else None,
                result.model_dump_json(),
            ),
        )
        self._conn.commit()

    def save_runs(self, results: list[RunResult], version_label: str) -> None:
        for result in results:
            self.save_run(result, version_label)

    # -- reads ---------------------------------------------------------------
    def get_runs(self, version_label: str) -> list[RunResult]:
        """Return all RunResults stored under ``version_label`` (ordered by task)."""
        rows = self._conn.execute(
            "SELECT result_json FROM runs WHERE version_label = ? ORDER BY task_id",
            (version_label,),
        ).fetchall()
        return [RunResult.model_validate_json(row["result_json"]) for row in rows]

    def get_run(self, version_label: str, task_id: str) -> RunResult | None:
        row = self._conn.execute(
            "SELECT result_json FROM runs WHERE version_label = ? AND task_id = ?",
            (version_label, task_id),
        ).fetchone()
        return RunResult.model_validate_json(row["result_json"]) if row else None

    def list_versions(self) -> list[VersionInfo]:
        """Summarize every stored version label."""
        rows = self._conn.execute(
            """
            SELECT version_label,
                   MIN(agent_model)  AS agent_model,
                   MIN(suite_version) AS suite_version,
                   COUNT(*)          AS task_count,
                   MIN(created_at)   AS first_created_at
            FROM runs
            GROUP BY version_label
            ORDER BY first_created_at
            """
        ).fetchall()
        return [
            VersionInfo(
                version_label=r["version_label"],
                agent_model=r["agent_model"],
                suite_version=r["suite_version"],
                task_count=r["task_count"],
                first_created_at=r["first_created_at"],
            )
            for r in rows
        ]
