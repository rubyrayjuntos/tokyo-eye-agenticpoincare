"""Backward-compatible alias for assess_checkpoint (plan name: evaluate)."""

from experiments.training.v6.assess_checkpoint import assess_checkpoint, main

__all__ = ["assess_checkpoint", "main"]
