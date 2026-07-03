"""Shared types for atomic compute job runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from science.compute.registry import DEFAULT_PATHWAY


def json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]

    module_name = type(value).__module__
    type_name = type(value).__name__
    if module_name == "numpy" and type_name == "ndarray":
        return value.tolist()
    if module_name == "numpy" and hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


@dataclass(frozen=True)
class JobRunContext:
    structure_id: str
    job_id: str
    pathway: str = DEFAULT_PATHWAY
    computation_run_id: str | None = None
    parent_run_id: str | None = None
    code_version: str | None = None
    device: str = "cpu"
    checkpoint_path: str | None = None
    job_params: dict[str, Any] = field(default_factory=dict)
    learned_curvature: float | None = None
    pipeline_job_id: str | None = None


@dataclass
class JobRunResult:
    job_id: str
    run_id: str
    structure_id: str
    success: bool
    artifacts_produced: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "structure_id": self.structure_id,
            "success": self.success,
            "artifacts_produced": self.artifacts_produced,
            "outputs": json_safe(self.outputs),
            "warnings": self.warnings,
        }
