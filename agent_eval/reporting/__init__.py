"""Failure attribution + pandas reporting (Phase 5)."""

from __future__ import annotations

from .attribution import (
    FailureCategory,
    FailureFinding,
    TaskFailure,
    attribute_failure,
    attribute_failures,
)
from .report import SuiteReport, format_report, results_dataframe, summarize

__all__ = [
    "FailureCategory",
    "FailureFinding",
    "TaskFailure",
    "attribute_failure",
    "attribute_failures",
    "SuiteReport",
    "summarize",
    "results_dataframe",
    "format_report",
]
