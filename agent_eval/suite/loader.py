"""Load a versioned task suite from a JSON file.

File shape:

    {
      "suite_version": "v1",
      "tasks": [ { ...Task fields... }, ... ]
    }

Each task inherits the top-level ``suite_version`` unless it sets its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Task


def load_suite(path: str | Path) -> tuple[str | None, list[Task]]:
    """Return ``(suite_version, tasks)`` parsed and validated from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    suite_version = data.get("suite_version")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("suite file must contain a 'tasks' list")

    tasks: list[Task] = []
    for raw in raw_tasks:
        payload = dict(raw)
        payload.setdefault("suite_version", suite_version)
        tasks.append(Task.model_validate(payload))
    return suite_version, tasks
