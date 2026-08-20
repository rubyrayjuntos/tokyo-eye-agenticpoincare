"""Evaluate gates before alias promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class PromoteBlocked(RuntimeError):
    """Raised when metrics fail the promotion threshold pack."""


@dataclass(frozen=True)
class ThresholdSpec:
    """Minimum acceptable value for a logged MLflow metric (greater-or-equal)."""

    metric: str
    minimum: float
    mode: Literal["min"] = "min"


@dataclass(frozen=True)
class EvaluateResult:
    passed: bool
    checked: dict[str, float]
    failures: list[str]

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise PromoteBlocked(
                "Evaluate gate failed: " + "; ".join(self.failures)
            )


def evaluate_metrics(
    metrics: dict[str, float],
    thresholds: list[ThresholdSpec],
) -> EvaluateResult:
    checked: dict[str, float] = {}
    failures: list[str] = []
    for spec in thresholds:
        if spec.metric not in metrics:
            failures.append(f"missing metric {spec.metric!r}")
            continue
        value = float(metrics[spec.metric])
        checked[spec.metric] = value
        if spec.mode == "min" and value < spec.minimum:
            failures.append(
                f"{spec.metric}={value} < minimum {spec.minimum}"
            )
    return EvaluateResult(
        passed=len(failures) == 0,
        checked=checked,
        failures=failures,
    )


def evaluate_run_metrics(
    *,
    run_id: str,
    thresholds: list[ThresholdSpec],
    tracking_uri: str | None = None,
) -> EvaluateResult:
    from science.tokyo_eye.governance.registry import ensure_tracking
    from mlflow.tracking import MlflowClient

    ensure_tracking(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    raw = {k: float(v) for k, v in (run.data.metrics or {}).items()}
    return evaluate_metrics(raw, thresholds)


def assert_promotable(
    *,
    run_id: str,
    thresholds: list[ThresholdSpec],
    tracking_uri: str | None = None,
) -> EvaluateResult:
    result = evaluate_run_metrics(
        run_id=run_id, thresholds=thresholds, tracking_uri=tracking_uri
    )
    result.raise_if_failed()
    return result


def thresholds_from_mapping(raw: dict[str, Any]) -> list[ThresholdSpec]:
    """Build threshold list from ``{metric: minimum}`` or ``{metric: {min: …}}``."""
    out: list[ThresholdSpec] = []
    for key, val in raw.items():
        if isinstance(val, dict):
            minimum = float(val["min"] if "min" in val else val["minimum"])
        else:
            minimum = float(val)
        out.append(ThresholdSpec(metric=str(key), minimum=minimum))
    return out
