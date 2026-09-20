"""Tokyo Eye EQU lift_radius — gates, learnable α lift, L_rad, final_only τ clamp (CPU-safe)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from experiments.training.v8.equ_cold_boot import (
    FORBIDDEN_CKPT_SUBSTRINGS as _BOOT_FORBIDDEN,
    PINNED_FRONTEND_SHA256,
    assert_frontend_bank,
    assert_not_forbidden_ckpt as _assert_not_forbidden_boot,
    load_boot_split,
    sha256_path,
)
from experiments.training.v8.equ_correct_start import (
    QuadraticRadiusController,
    assert_pure_hyp_strict,
    epoch_is_best_eligible,
    hyp_radius,
    normalized_routing_entropy,
    static_pure_hyp_pass,
)
from science.tokyo_eye.v8.engine import CurriculumRadiusController  # noqa: F401

GATE_ID = "tokyo_eye_equ_lift_radius"
DISPLAY_LINEAGE = "Tokyo Eye EQU"
MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine"
MLFLOW_RUN_NAME = "tokyo_eye_equ_lift_radius"
DEFAULT_MANIFEST = Path("manifests/equ_cold_boot_v1.json")
DEFAULT_PINS = Path("data/gates/tokyo_eye_equ_lift_radius_pins.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_equ_lift_radius.json")
DEFAULT_FRONTEND_CKPT = Path(
    "checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt"
)

RV_EPOCHS = 20
RV_PROBE_EVERY = 5
RV_TAU_START = 0.60
RV_TAU_END = 0.95
RV_WRAP_MAX = 1
RV_BOUNDARY_RADIUS = 0.80
RV_LR_HYP = 3.0e-4
RV_LR_BACKBONE = 1.0e-5
RV_WEIGHT_DECAY = 1.0e-4
RV_SAT_CEILING = 0.50  # mean Euclidean ball radius
RV_SPREAD_FLOOR = 0.15  # std of hyp radii
RV_H_NORM_FLOOR = 0.60
RV_EQUIV_CEILING = 1.0e-5
RV_MIN_TRAIN = 6
RV_BEST_MOE_MIN = 0.02
RV_NUM_EXPERTS = 4
RV_VOLUME_COEFF = 1.0
RV_VOLUME_SIGMA = 0.20
RV_VOLUME_LAM = 1.0
RV_VOLUME_MU = 0.35
RV_VOLUME_LAM_MEAN = 3.0
RV_VOLUME_LAM_BARRIER = 0.5
RV_TAU_CLAMP_MODE = "final_only"

FORBIDDEN_CKPT_SUBSTRINGS = tuple(
    dict.fromkeys(
        list(_BOOT_FORBIDDEN)
        + [
            "eqf_equ_correct_start",
            "tokyo_eye_equ_correct_start",
            "correct_start",
            "eqf_equ_rim_volume",
            "tokyo_eye_equ_rim_volume",
            "rim_volume",
            "eqf_equ_shell_unpack",
            "tokyo_eye_equ_shell_unpack",
            "shell_unpack",
        ]
    )
)


class LiftRadiusGuardError(RuntimeError):
    """Refuses start when pure-hyp / pin / forbidden init checks fail."""


def assert_not_forbidden_ckpt(path: Path | str) -> None:
    p = Path(path)
    text = str(p)
    for token in FORBIDDEN_CKPT_SUBSTRINGS:
        if token in text:
            raise LiftRadiusGuardError(f"forbidden checkpoint path token {token!r}: {p}")
    _assert_not_forbidden_boot(p)


def load_pins(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_PINS
    if not p.is_file():
        return {
            "frontend_ckpt": str(DEFAULT_FRONTEND_CKPT),
            "frontend_bank_sha256": PINNED_FRONTEND_SHA256,
            "source_manifest": str(DEFAULT_MANIFEST),
        }
    return json.loads(p.read_text())


def enrich_rim_probe_row(
    row: Mapping[str, Any],
    *,
    z_hyp: torch.Tensor | None = None,
    c: float = 1.0,
) -> dict[str, Any]:
    """Attach sealed sat/spread fields: mean ‖z‖₂ and std(r_H)."""
    out = dict(row)
    if z_hyp is not None and torch.is_tensor(z_hyp) and z_hyp.numel():
        z = z_hyp.detach().float()
        r = torch.linalg.vector_norm(z, dim=-1)
        out["mean_ball_radius"] = float(r.mean().item())
        r_h = hyp_radius(z, c=float(c))
        out["std_hyp_radius"] = float(r_h.std(unbiased=False).item()) if r_h.numel() > 1 else 0.0
    else:
        # Fall back to observe_one mean_radius; spread unknown → 0
        if out.get("mean_radius") is not None:
            out["mean_ball_radius"] = float(out["mean_radius"])
        out.setdefault("std_hyp_radius", 0.0)
    return out


def evaluate_probe_hygiene(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    sat_ceiling: float = RV_SAT_CEILING,
    spread_floor: float = RV_SPREAD_FLOOR,
    h_norm_floor: float = RV_H_NORM_FLOOR,
) -> dict[str, Any]:
    loaded = [r for r in probe_rows if r.get("loaded")]
    sats = [
        float(r.get("mean_ball_radius", r.get("mean_radius")))
        for r in loaded
        if r.get("mean_ball_radius") is not None or r.get("mean_radius") is not None
    ]
    spreads = [
        float(r["std_hyp_radius"])
        for r in loaded
        if r.get("std_hyp_radius") is not None and math.isfinite(float(r["std_hyp_radius"]))
    ]
    h_norms = [
        float(r["h_norm"])
        for r in loaded
        if r.get("h_norm") is not None and math.isfinite(float(r["h_norm"]))
    ]
    finite = all(
        bool(r.get("z_hyp_finite")) and bool(r.get("curvature_finite")) for r in loaded
    )
    mean_sat = float(sum(sats) / len(sats)) if sats else 1.0
    mean_spread = float(sum(spreads) / len(spreads)) if spreads else 0.0
    mean_h = float(sum(h_norms) / len(h_norms)) if h_norms else 0.0
    return {
        "n_probe_loaded": len(loaded),
        "mean_sat": mean_sat,
        "mean_spread": mean_spread,
        "mean_h_norm": mean_h,
        "finite_h2": finite and len(loaded) == 6,
        "probe_sat_gate": mean_sat < float(sat_ceiling),
        "radius_spread_gate": mean_spread > float(spread_floor),
        "moe_liveness_gate": mean_h >= float(h_norm_floor),
        "finite_h2_gate": finite and len(loaded) == 6,
        "sat_formula": "mean_ball_radius",
        "spread_formula": "std_hyp_radius",
    }


@torch.no_grad()
def evaluate_lift_spine_equivariance(
    system: nn.Module,
    batch: Mapping[str, Any],
    *,
    tau_ceiling: float,
    seed: int = 0,
) -> dict[str, float]:
    """Sealed Δ_equiv: hold s, rotate v (3-vector channels), compare z_hyp.

    Also logs full-system R·x residual as diagnostic only (not the sealed gate).
    """
    system.eval()
    device = batch["x"].device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    r, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(r) < 0:
        r = r.clone()
        r[:, 0] = -r[:, 0]
    r = r.to(device=device, dtype=batch["x"].dtype)

    # Full-system diagnostic (frontend not e(3)-true → expected nonzero)
    out0 = system(
        batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=tau_ceiling
    )
    out_full = system(
        batch["x"] @ r.T,
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceiling,
    )
    full_diff = torch.linalg.vector_norm(out0["z_hyp"] - out_full["z_hyp"], dim=-1)
    full_residual = float(full_diff.max().item())

    # Lift+spine: freeze frontend features, rotate vector channels only
    s, v = system.frontend(
        batch["x"], edge_index=batch["edge_index"], edge_type=batch["edge_type"]
    )
    v_rot = _rotate_vector_channels(v, r)
    spine = system.spine
    z0 = spine(s, v, batch["edge_index"], batch["edge_type"], tau_ceiling=tau_ceiling)
    z1 = spine(
        s, v_rot, batch["edge_index"], batch["edge_type"], tau_ceiling=tau_ceiling
    )
    diff = torch.linalg.vector_norm(z0["z_hyp"] - z1["z_hyp"], dim=-1)
    lift_residual = float(diff.max().item())
    return {
        "lift_spine_residual": lift_residual,
        "full_system_residual": full_residual,
    }


def _rotate_vector_channels(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Apply SO(3) rotation to each 3-vector channel in ``v`` [N, 3K] or [N, 3]."""
    if v.shape[-1] == 3:
        return v @ r.T
    if v.shape[-1] % 3 == 0:
        n, d = v.shape
        vv = v.view(n, d // 3, 3)
        vv = torch.einsum("nki,ji->nkj", vv, r)  # v @ R.T per channel
        return vv.reshape(n, d)
    # Non-geometric flat vector: leave unchanged (no sealed SO(3) action)
    return v


def build_lift_radius_stamp(
    *,
    pure_hyp: Mapping[str, Any],
    hygiene: Mapping[str, Any],
    equiv_residual: float | None,
    full_system_residual: float | None,
    frontend_sha256: str,
    mlflow_run_id: str | None,
    extra: Mapping[str, Any] | None = None,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    equiv_ok = (
        equiv_residual is not None
        and math.isfinite(float(equiv_residual))
        and float(equiv_residual) < RV_EQUIV_CEILING
    )
    gates = {
        "pure_hyp_pass": {
            "pass": bool(pure_hyp.get("pure_hyp_pass")),
            "value": 1.0 if pure_hyp.get("pure_hyp_pass") else 0.0,
            "threshold": 1.0,
        },
        "probe_sat_gate": {
            "pass": bool(hygiene.get("probe_sat_gate")),
            "value": float(hygiene.get("mean_sat", 1.0)),
            "threshold": RV_SAT_CEILING,
        },
        "radius_spread_gate": {
            "pass": bool(hygiene.get("radius_spread_gate")),
            "value": float(hygiene.get("mean_spread", 0.0)),
            "threshold": RV_SPREAD_FLOOR,
        },
        "finite_h2_gate": {
            "pass": bool(hygiene.get("finite_h2_gate")),
            "value": 1.0 if hygiene.get("finite_h2_gate") else 0.0,
            "threshold": 1.0,
        },
        "moe_liveness_gate": {
            "pass": bool(hygiene.get("moe_liveness_gate")),
            "value": float(hygiene.get("mean_h_norm", 0.0)),
            "threshold": RV_H_NORM_FLOOR,
        },
        "equiv_residual_gate": {
            "pass": bool(equiv_ok),
            "value": float(equiv_residual) if equiv_residual is not None else float("nan"),
            "threshold": RV_EQUIV_CEILING,
            "scope": "lift_spine_rotate_v",
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    stamp: dict[str, Any] = {}
    if prior:
        stamp.update(dict(prior))
    stamp.update(
        {
            "gate_id": GATE_ID,
            "display_lineage": DISPLAY_LINEAGE,
            "status": "QUALIFIED" if all_pass else "FAILED",
            "execution_state": (
                "TRAIN_COMPLETE_QUALIFIED" if all_pass else "TRAIN_COMPLETE_FAILED"
            ),
            "mlflow_experiment": MLFLOW_EXPERIMENT,
            "mlflow_run_id": mlflow_run_id,
            "frontend_sha256": frontend_sha256,
            "gates": gates,
            "hygiene": dict(hygiene),
            "equiv_diagnostics": {
                "lift_spine_residual": equiv_residual,
                "full_system_residual": full_system_residual,
            },
            "pure_hyp": {
                "pass": bool(pure_hyp.get("pure_hyp_pass")),
                "findings": pure_hyp.get("findings"),
            },
            "artifacts": [
                "docs/superpowers/specs/2026-09-15-tokyoeye-equ-lift-radius-design.md",
                "docs/superpowers/plans/2026-09-15-tokyoeye-equ-lift-radius.md",
                "experiments/training/v8/run_tokyo_eye_equ_lift_radius.py",
            ],
        }
    )
    if extra:
        stamp["run_extra"] = dict(extra)
    if prior:
        for k in ("approved_at", "approver", "design", "plan", "governance", "amendments"):
            if k in prior:
                stamp[k] = prior[k]
        if prior.get("status") in {"APPROVED_LOCKED", "APPROVED"}:
            stamp["approval_status"] = "APPROVED_LOCKED"
    return stamp


__all__ = [
    "DISPLAY_LINEAGE",
    "FORBIDDEN_CKPT_SUBSTRINGS",
    "GATE_ID",
    "MLFLOW_EXPERIMENT",
    "MLFLOW_RUN_NAME",
    "PINNED_FRONTEND_SHA256",
    "QuadraticRadiusController",
    "RV_EPOCHS",
    "RV_LR_HYP",
    "RV_TAU_END",
    "RV_TAU_START",
    "RV_VOLUME_COEFF",
    "RV_VOLUME_MU",
    "RV_VOLUME_LAM_MEAN",
    "RV_VOLUME_LAM_BARRIER",
    "RV_TAU_CLAMP_MODE",
    "LiftRadiusGuardError",
    "assert_frontend_bank",
    "assert_not_forbidden_ckpt",
    "assert_pure_hyp_strict",
    "build_lift_radius_stamp",
    "enrich_rim_probe_row",
    "epoch_is_best_eligible",
    "evaluate_lift_spine_equivariance",
    "evaluate_probe_hygiene",
    "load_boot_split",
    "load_pins",
    "normalized_routing_entropy",
    "sha256_path",
    "static_pure_hyp_pass",
]
