"""Tokyo Eye EQU geoopt_restore — Equiformer-to-pool + geoopt lift gates (CPU-safe)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from experiments.training.v8.equ_cold_boot import (
    FORBIDDEN_CKPT_SUBSTRINGS as _BOOT_FORBIDDEN,
    PINNED_FRONTEND_SHA256,
    assert_frontend_bank,
    assert_not_forbidden_ckpt as _assert_not_forbidden_boot,
    load_boot_split,
)
from experiments.training.v8.equ_correct_start import (
    QuadraticRadiusController,
    assert_pure_hyp_strict,
    epoch_is_best_eligible,
    hyp_radius,
    normalized_routing_entropy,
)
from experiments.training.v8.equ_lift_radius import (
    enrich_rim_probe_row,
    evaluate_lift_spine_equivariance,
    evaluate_probe_hygiene,
)

GATE_ID = "tokyo_eye_equ_geoopt_restore"
DISPLAY_LINEAGE = "Tokyo Eye EQU"
MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine"
MLFLOW_RUN_NAME = "tokyo_eye_equ_geoopt_restore"
DEFAULT_MANIFEST = Path("manifests/equ_cold_boot_v1.json")
DEFAULT_PINS = Path("data/gates/tokyo_eye_equ_geoopt_restore_pins.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_equ_geoopt_restore.json")
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
RV_SAT_CEILING = 0.50
RV_SPREAD_FLOOR = 0.15
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
            "eqf_equ_theme_biology",
            "tokyo_eye_equ_theme_biology",
            "theme_biology",
            "eqf_equ_lift_radius",
            "tokyo_eye_equ_lift_radius",
            "lift_radius",
        ]
    )
)


class GeooptRestoreGuardError(RuntimeError):
    """Refuses start when restore / pure-hyp / pin checks fail."""


def assert_not_forbidden_ckpt(path: Path | str) -> None:
    p = Path(path)
    text = str(p)
    for token in FORBIDDEN_CKPT_SUBSTRINGS:
        if token in text:
            raise GeooptRestoreGuardError(
                f"forbidden checkpoint path token {token!r}: {p}"
            )
    _assert_not_forbidden_boot(p)


def load_pins(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_PINS
    if not p.is_file():
        return {
            "frontend_ckpt": str(DEFAULT_FRONTEND_CKPT),
            "frontend_bank_sha256": PINNED_FRONTEND_SHA256,
            "source_manifest": str(DEFAULT_MANIFEST),
            "forbid_se3_lite": True,
            "lift": "geoopt.manifolds.PoincareBall",
        }
    return json.loads(p.read_text())


def assert_geoopt_restore_preflight(system: nn.Module) -> dict[str, Any]:
    """Prove Equiformer-pool frontend + geoopt lift; veto SE(3)-lite."""
    frontend = getattr(system, "frontend", None)
    if frontend is None:
        raise GeooptRestoreGuardError("no frontend on system")
    mode = getattr(frontend, "frontend_mode", None)
    if mode != "equiformer_v3_pool":
        raise GeooptRestoreGuardError(
            f"forbid_se3_lite: frontend_mode={mode!r} (need equiformer_v3_pool)"
        )
    if getattr(frontend, "live_backbone", False):
        raise GeooptRestoreGuardError("forbid_se3_lite: live_backbone=True")
    cls = frontend.__class__.__name__
    if cls == "StubEquiformerFrontend":
        raise GeooptRestoreGuardError("forbid_se3_lite: StubEquiformerFrontend present")

    spine = getattr(system, "spine", None)
    if spine is None:
        raise GeooptRestoreGuardError("no spine on system")
    proj = getattr(spine, "projector", None)
    if proj is None:
        raise GeooptRestoreGuardError("spine missing projector")
    manifold = getattr(proj, "manifold", None)
    man_name = type(manifold).__name__ if manifold is not None else None
    if man_name != "PoincareBall":
        raise GeooptRestoreGuardError(
            f"lift must be geoopt PoincareBall, got {man_name!r}"
        )
    # Ensure projector module is geoopt-backed (has expmap0 path)
    if not hasattr(manifold, "expmap0"):
        raise GeooptRestoreGuardError("manifold missing expmap0")

    return {
        "frontend_mode": mode,
        "frontend_class": cls,
        "lift": "geoopt.manifolds.PoincareBall",
        "forbid_se3_lite": True,
        "preflight_pass": True,
    }


def build_geoopt_restore_stamp(
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
            "frontend": "equiformer_v3_pool",
            "lift": "geoopt.manifolds.PoincareBall",
            "forbid_se3_lite": True,
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
                "docs/superpowers/specs/2026-09-16-tokyoeye-equ-geoopt-assembly-contract.md",
                "docs/superpowers/plans/2026-09-16-tokyoeye-equ-geoopt-restore.md",
                "experiments/training/v8/run_tokyo_eye_equ_geoopt_restore.py",
            ],
        }
    )
    if extra:
        stamp.update(dict(extra))
    return stamp


__all__ = [
    "GATE_ID",
    "DISPLAY_LINEAGE",
    "MLFLOW_EXPERIMENT",
    "MLFLOW_RUN_NAME",
    "DEFAULT_MANIFEST",
    "DEFAULT_PINS",
    "DEFAULT_STAMP",
    "DEFAULT_FRONTEND_CKPT",
    "PINNED_FRONTEND_SHA256",
    "QuadraticRadiusController",
    "GeooptRestoreGuardError",
    "assert_frontend_bank",
    "assert_geoopt_restore_preflight",
    "assert_not_forbidden_ckpt",
    "assert_pure_hyp_strict",
    "build_geoopt_restore_stamp",
    "enrich_rim_probe_row",
    "epoch_is_best_eligible",
    "evaluate_lift_spine_equivariance",
    "evaluate_probe_hygiene",
    "hyp_radius",
    "load_boot_split",
    "load_pins",
    "normalized_routing_entropy",
    "RV_EPOCHS",
    "RV_PROBE_EVERY",
    "RV_TAU_START",
    "RV_TAU_END",
    "RV_WRAP_MAX",
    "RV_BOUNDARY_RADIUS",
    "RV_LR_HYP",
    "RV_LR_BACKBONE",
    "RV_WEIGHT_DECAY",
    "RV_SAT_CEILING",
    "RV_SPREAD_FLOOR",
    "RV_H_NORM_FLOOR",
    "RV_EQUIV_CEILING",
    "RV_MIN_TRAIN",
    "RV_BEST_MOE_MIN",
    "RV_NUM_EXPERTS",
    "RV_VOLUME_COEFF",
    "RV_VOLUME_SIGMA",
    "RV_VOLUME_LAM",
    "RV_VOLUME_MU",
    "RV_VOLUME_LAM_MEAN",
    "RV_VOLUME_LAM_BARRIER",
    "RV_TAU_CLAMP_MODE",
]
