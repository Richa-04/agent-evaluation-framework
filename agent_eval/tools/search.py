"""Mock search tool: looks up facts in a small in-memory knowledge base.

Deterministic by design (no network) so runs and tests are reproducible. A query
matches a corpus entry when all of the entry-key's words appear in the query.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Tool, ToolError

_WORD_RE = re.compile(r"[a-z0-9]+")

DEFAULT_CORPUS: dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "speed of light": (
        "The speed of light in a vacuum is 299792458 meters per second."
    ),
    "tallest mountain": (
        "Mount Everest is Earth's tallest mountain above sea level, at 8849 meters."
    ),
    "largest planet": "Jupiter is the largest planet in the Solar System.",
}


class MockSearchTool(Tool):
    name = "search"
    description = (
        "Search a small knowledge base for a fact and return the best matching "
        "snippet, or a 'no results' message if nothing matches."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up, e.g. 'capital of France'.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, corpus: dict[str, str] | None = None) -> None:
        self.corpus = corpus if corpus is not None else DEFAULT_CORPUS

    async def _execute(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("'query' must be a non-empty string")
        words = set(_WORD_RE.findall(query.lower()))
        for key, value in self.corpus.items():
            if all(token in words for token in key.split()):
                return value, {"query": query, "matched": key}
        return f"No results found for '{query}'.", {"query": query, "matched": None}
