"""τ-rim / structural-SSOT outer-shell signal gate for post-ingest GNN outputs.

Validates that per-residue uncertainty has usable spread and that cone_depth
tracks the hyperbolic rim (disc radius, τ dehydron flag) — not SASA correlation.

Use after ``gnn_inference`` before trusting interactive HTML outer-shell reads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNNodeOutput
from science.dtie.v6.visualization.interactive_viewer import (
    _node_aleatoric,
    investigation_scores,
)

# Floors aligned with dehydron-rim cone gate + shell diagnostics intent.
MIN_EPI_STD = 0.015
MIN_ALE_STD = 0.015
MIN_EPI_RANGE = 0.05
MIN_ALE_RANGE = 0.05
MIN_INVESTIGATION_STD = 0.06
MIN_R_DISC_R_DEPTH = 0.25
MIN_R_DEPTH_TAU = 0.12
MIN_RIM_EPI_MEAN_DELTA = 0.02  # mean(epi | top disc quartile) - mean(epi | bottom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _node_arrays(nodes: list[GNNNodeOutput]) -> dict[str, np.ndarray]:
    if not nodes:
        return {
            "epistemic": np.array([], dtype=np.float64),
            "aleatoric": np.array([], dtype=np.float64),
            "cone_depth": np.array([], dtype=np.float64),
            "disc_r": np.array([], dtype=np.float64),
            "tau": np.array([], dtype=np.float64),
            "investigation": np.array([], dtype=np.float64),
        }
    epistemic = np.array([n.epistemic_uncertainty for n in nodes], dtype=np.float64)
    aleatoric = np.array([_node_aleatoric(n) for n in nodes], dtype=np.float64)
    depth = np.array([n.cone_depth for n in nodes], dtype=np.float64)
    disc_r = np.array(
        [
            float(np.hypot(n.hyp_projections[0], n.hyp_projections[1]))
            if n.hyp_projections is not None and len(n.hyp_projections) >= 2
            else 0.0
            for n in nodes
        ],
        dtype=np.float64,
    )
    tau = np.array(
        [
            float(n.input_features[1])
            if n.input_features is not None and len(n.input_features) > 1
            else 0.0
            for n in nodes
        ],
        dtype=np.float64,
    )
    investigation = investigation_scores(epistemic, aleatoric)
    return {
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "cone_depth": depth,
        "disc_r": disc_r,
        "tau": tau,
        "investigation": investigation,
    }


@dataclass
class ShellSignalGateVerdict:
    """Result of τ-rim outer-shell + uncertainty spread gate."""

    passed: bool
    structure_id: str
    n_residues: int
    structural_disc_frozen: bool
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            return (
                f"shell signal OK ({self.n_residues} residues, "
                f"r(disc_r,depth)={self.metrics.get('r_disc_r_depth', float('nan')):.3f}, "
                f"σ_epi={self.metrics.get('epistemic_std', float('nan')):.4f})"
            )
        return "; ".join(self.failures) if self.failures else "shell signal gate failed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_shell_signal_ssot(
    nodes: list[GNNNodeOutput],
    *,
    structure_id: str = "",
    structural_disc_frozen: bool = True,
) -> ShellSignalGateVerdict:
    """Return pass/fail for outer-shell uncertainty + τ-rim depth alignment."""
    failures: list[str] = []
    warnings: list[str] = []

    if not nodes:
        return ShellSignalGateVerdict(
            passed=False,
            structure_id=structure_id,
            n_residues=0,
            structural_disc_frozen=structural_disc_frozen,
            failures=["no GNN nodes"],
        )

    arr = _node_arrays(nodes)
    epi, ale = arr["epistemic"], arr["aleatoric"]
    depth, disc_r, tau = arr["cone_depth"], arr["disc_r"], arr["tau"]
    inv = arr["investigation"]

    epi_std = float(epi.std())
    ale_std = float(ale.std())
    epi_range = float(epi.max() - epi.min())
    ale_range = float(ale.max() - ale.min())
    inv_std = float(inv.std())
    r_disc_depth = _pearson(disc_r, depth)
    r_depth_tau = _pearson(depth, tau)

    # Rim vs core epistemic contrast (top vs bottom disc-r quartile)
    q25, q75 = np.percentile(disc_r, [25, 75])
    rim_mask = disc_r >= q75
    core_mask = disc_r <= q25
    rim_epi_delta = float("nan")
    if rim_mask.any() and core_mask.any():
        rim_epi_delta = float(epi[rim_mask].mean() - epi[core_mask].mean())

    metrics: dict[str, float] = {
        "epistemic_std": epi_std,
        "aleatoric_std": ale_std,
        "epistemic_range": epi_range,
        "aleatoric_range": ale_range,
        "investigation_std": inv_std,
        "r_disc_r_depth": r_disc_depth,
        "r_depth_tau": r_depth_tau,
        "rim_epistemic_delta": rim_epi_delta,
        "disc_r_std": float(disc_r.std()),
        "cone_depth_std": float(depth.std()),
    }

    if epi_std < MIN_EPI_STD:
        failures.append(f"flat epistemic (σ={epi_std:.4f} < {MIN_EPI_STD})")
    if ale_std < MIN_ALE_STD:
        failures.append(f"flat aleatoric (σ={ale_std:.4f} < {MIN_ALE_STD})")
    if epi_range < MIN_EPI_RANGE:
        failures.append(f"epistemic range {epi_range:.4f} < {MIN_EPI_RANGE}")
    if ale_range < MIN_ALE_RANGE:
        failures.append(f"aleatoric range {ale_range:.4f} < {MIN_ALE_RANGE}")
    if inv_std < MIN_INVESTIGATION_STD:
        failures.append(f"flat investigation score (σ={inv_std:.4f} < {MIN_INVESTIGATION_STD})")

    if not np.isfinite(r_disc_depth) or r_disc_depth < MIN_R_DISC_R_DEPTH:
        failures.append(
            f"weak disc rim gradient r(disc_r,depth)={r_disc_depth:.3f} < {MIN_R_DISC_R_DEPTH}"
        )
    if structural_disc_frozen and (
        not np.isfinite(r_depth_tau) or r_depth_tau < MIN_R_DEPTH_TAU
    ):
        failures.append(
            f"τ-rim depth misaligned r(depth,τ)={r_depth_tau:.3f} < {MIN_R_DEPTH_TAU}"
        )

    if np.isfinite(rim_epi_delta) and abs(rim_epi_delta) < MIN_RIM_EPI_MEAN_DELTA:
        warnings.append(
            f"rim vs core epistemic contrast weak (Δ={rim_epi_delta:.4f})"
        )

    return ShellSignalGateVerdict(
        passed=len(failures) == 0,
        structure_id=structure_id,
        n_residues=len(nodes),
        structural_disc_frozen=structural_disc_frozen,
        metrics=metrics,
        failures=failures,
        warnings=warnings,
    )


def write_shell_gate_report(
    verdict: ShellSignalGateVerdict,
    output_path: Path,
) -> Path:
    """Persist gate JSON next to viewer artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = verdict.to_dict()
    payload["gate"] = "shell_signal_ssot_v1"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
