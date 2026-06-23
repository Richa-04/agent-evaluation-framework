"""SQLite persistence + version regression comparison (Phase 4)."""

from __future__ import annotations

from .compare import (
    ComparisonStatus,
    TaskComparison,
    VersionComparison,
    compare_versions,
    format_comparison,
)
from .db import RunStore, VersionInfo, make_version_label

__all__ = [
    "RunStore",
    "VersionInfo",
    "make_version_label",
    "compare_versions",
    "format_comparison",
    "ComparisonStatus",
    "TaskComparison",
    "VersionComparison",
]
