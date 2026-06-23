"""Read-file tool: read a UTF-8 text file from a sandboxed base directory.

All paths are resolved and confined to ``base_dir`` — traversal (``..``,
absolute paths, symlinks pointing outside) is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolError

# Cap returned content so a huge file can't blow up the context window.
_MAX_BYTES = 20_000


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a UTF-8 text file by path, relative to the sandbox "
        "directory. Returns the file's text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the sandbox, e.g. 'notes.txt'.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).resolve()

    async def _execute(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        rel = arguments.get("path")
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("'path' must be a non-empty string")

        target = (self.base_dir / rel).resolve()
        if target != self.base_dir and self.base_dir not in target.parents:
            raise ToolError("path escapes the sandbox directory")
        if not target.is_file():
            raise ToolError(f"no such file: {rel}")

        text = target.read_text(encoding="utf-8")
        truncated = len(text) > _MAX_BYTES
        if truncated:
            text = text[:_MAX_BYTES] + "\n...[truncated]"
        return text, {"path": rel, "bytes": len(text), "truncated": truncated}
