"""Atomic compute job runners (peeled from the legacy monolith)."""

from science.compute.runners.base import JobRunContext, JobRunResult
from science.compute.runners.source_leak_detection import run_source_leak_detection

__all__ = [
    "JobRunContext",
    "JobRunResult",
    "run_source_leak_detection",
]
