"""Tool abstraction shared by all agent tools.

A Tool exposes an Anthropic-compatible schema (`to_param`) and an async `run`
that never raises — it converts argument/execution errors into an error
``ToolResult`` so a misbehaving tool can never crash the agent loop. Subclasses
implement `_execute`, returning ``(content_str, raw_dict_or_None)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import ToolResult


class ToolError(Exception):
    """Raised by a tool for invalid arguments or a recoverable execution failure.

    The runner surfaces this back to the model as an error tool_result so it can
    adjust and retry, rather than aborting the run.
    """


class Tool(ABC):
    """Base class for agent tools.

    Subclasses set ``name``, ``description``, ``input_schema`` and implement
    ``_execute``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_param(self) -> dict[str, Any]:
        """Anthropic ``tools`` entry for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool, converting any failure into an error ToolResult."""
        try:
            content, raw = await self._execute(arguments)
            return ToolResult(content=content, is_error=False, raw=raw)
        except ToolError as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)
        except Exception as exc:  # defensive: a tool must never crash the loop
            return ToolResult(
                content=f"Error: unexpected failure in tool '{self.name}': {exc}",
                is_error=True,
            )

    @abstractmethod
    async def _execute(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Do the work. Return (content, raw). Raise ToolError on bad input."""
        raise NotImplementedError
