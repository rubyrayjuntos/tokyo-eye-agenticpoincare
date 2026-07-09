"""V6 GNN training lifecycle package.

Keep this module import-light: the agent container mounts ``science/training``
but does not install torch. Heavy exports (checkpoint, monitor) are available
via direct submodule imports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CheckpointManager",
    "ConvergenceMonitor",
    "LossCoeffs",
    "PhaseConfig",
    "TrainingConfig",
]


def __getattr__(name: str) -> Any:
    if name in {"LossCoeffs", "PhaseConfig", "TrainingConfig"}:
        from science.training.config import LossCoeffs, PhaseConfig, TrainingConfig

        return {
            "LossCoeffs": LossCoeffs,
            "PhaseConfig": PhaseConfig,
            "TrainingConfig": TrainingConfig,
        }[name]
    if name == "CheckpointManager":
        from science.training.checkpoint import CheckpointManager

        return CheckpointManager
    if name == "ConvergenceMonitor":
        from science.training.monitor import ConvergenceMonitor

        return ConvergenceMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
