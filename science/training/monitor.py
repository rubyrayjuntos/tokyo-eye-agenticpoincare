"""Training convergence monitor with fail-fast abort conditions."""

from __future__ import annotations

import math
from typing import Any


class ConvergenceMonitor:
    """Monitors training convergence and implements fail-fast conditions."""

    REQUIRED_KEYS = (
        "total",
        "cone_consistency",
        "neighborhood_consistency",
        "angular_diversity",
        "domain_separation_2d",
        "domain_separation_3d",
    )

    def __init__(self, collapse_window: int = 3) -> None:
        self.history: list[dict[str, Any]] = []
        self.collapse_window = collapse_window
        self._nan_streak = 0
        self._zero_depth_streak = 0

    def record_epoch(self, metrics: dict[str, Any], health: dict[str, Any] | None = None) -> None:
        """Record epoch metrics and update abort streak counters."""
        entry = {"metrics": metrics, "health": health or {}}
        self.history.append(entry)

        nonfinite = False
        for key in ("total", "cone_consistency", "routing_entropy"):
            val = metrics.get(key)
            if val is not None and (math.isnan(float(val)) or math.isinf(float(val))):
                nonfinite = True
                break
        if not nonfinite and health:
            for key in ("cone_range_mean", "radial_std_mean", "proj_frac_mean"):
                val = health.get(key)
                if val is not None and (math.isnan(float(val)) or math.isinf(float(val))):
                    nonfinite = True
                    break
        if nonfinite:
            self._nan_streak += 1
        else:
            self._nan_streak = 0

        radial_std = (health or {}).get("radial_std_mean")
        cone_range = (health or {}).get("cone_range_mean")
        if cone_range is None and radial_std is None:
            depth_collapsed = False
        else:
            depth_collapsed = (
                radial_std is not None and float(radial_std) == 0.0
            ) or (cone_range is not None and float(cone_range) == 0.0)
        if depth_collapsed:
            self._zero_depth_streak += 1
        else:
            self._zero_depth_streak = 0

    def should_abort(self) -> tuple[bool, str]:
        """Check if training should abort due to NaN or depth collapse."""
        if self._nan_streak >= self.collapse_window:
            return True, f"nonfinite metrics for {self.collapse_window} consecutive epochs"
        if self._zero_depth_streak >= self.collapse_window:
            return (
                True,
                f"depth collapsed (radial_std=0 or cone_range=0) for "
                f"{self.collapse_window} consecutive epochs",
            )
        return False, ""

    def stage_summary(self) -> dict[str, Any]:
        """Return convergence summary for the completed stage."""
        if not self.history:
            return {"epochs": 0}
        totals = [float(h["metrics"].get("total", 0.0)) for h in self.history]
        return {
            "epochs": len(self.history),
            "loss_first": totals[0],
            "loss_last": totals[-1],
            "loss_min": min(totals),
        }

    @staticmethod
    def validate_epoch_metrics(metrics: dict[str, Any]) -> list[str]:
        """Return missing required metric keys."""
        missing = [k for k in ConvergenceMonitor.REQUIRED_KEYS if k not in metrics]
        grad_keys = ("grad_radial", "grad_angular", "grad_backbone")
        missing.extend(k for k in grad_keys if k not in metrics)
        return missing
