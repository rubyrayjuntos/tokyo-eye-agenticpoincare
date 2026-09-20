"""tokyo_eye_equ_theme_restore — theme dehydron AUPRC + geometry HOLD helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from experiments.training.v8.equ_lift_radius import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    DISPLAY_LINEAGE,
    FORBIDDEN_CKPT_SUBSTRINGS as _LIFT_FORBIDDEN,
    MLFLOW_EXPERIMENT,
    PINNED_FRONTEND_SHA256,
    RV_EQUIV_CEILING,
    RV_H_NORM_FLOOR,
    RV_SAT_CEILING,
    RV_SPREAD_FLOOR,
    RV_TAU_CLAMP_MODE,
    RV_WRAP_MAX,
    assert_frontend_bank,
    assert_not_forbidden_ckpt as _assert_not_forbidden_lift,
    enrich_rim_probe_row,
    epoch_is_best_eligible,
    evaluate_lift_spine_equivariance,
    evaluate_probe_hygiene,
    load_pins,
)
from science.tokyo_eye.v8.metrics import binary_auprc

GATE_ID = "tokyo_eye_equ_theme_restore"
TAG = "[equ-theme-restore]"

# Re-export train knobs from geoopt_restore (same geometry recipe).
from experiments.training.v8.equ_lift_radius import (  # noqa: E402
    DEFAULT_PINS as _LIFT_PINS,
    QuadraticRadiusController,
    RV_BEST_MOE_MIN,
    RV_BOUNDARY_RADIUS,
    RV_EPOCHS,
    RV_LR_BACKBONE,
    RV_LR_HYP,
    RV_MIN_TRAIN,
    RV_NUM_EXPERTS,
    RV_PROBE_EVERY,
    RV_TAU_END,
    RV_TAU_START,
    RV_VOLUME_COEFF,
    RV_VOLUME_LAM,
    RV_VOLUME_LAM_BARRIER,
    RV_VOLUME_LAM_MEAN,
    RV_VOLUME_MU,
    RV_VOLUME_SIGMA,
    RV_WEIGHT_DECAY,
    LiftRadiusGuardError as ThemeRestoreGuardError,
    assert_pure_hyp_strict,
    load_boot_split,
    normalized_routing_entropy,
)

DEFAULT_STAMP = Path("data/gates/tokyo_eye_equ_theme_restore.json")
DEFAULT_PINS = Path("data/gates/tokyo_eye_equ_theme_restore_pins.json")
MLFLOW_RUN_NAME = GATE_ID
CorrectStartGuardError = ThemeRestoreGuardError

TB_THEME_AUPRC_FLOOR = 0.55
TB_MEAN_THEME_AUPRC = 0.60
TB_THEME_COLLAPSE = 0.40
TB_MIN_THEMES_PASS = 4
TB_N_PROBE = 6

DEFAULT_SPINE_INIT = Path(
    "checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt"
)

FORBIDDEN_CKPT_SUBSTRINGS = tuple(
    dict.fromkeys(
        list(_LIFT_FORBIDDEN)
        + [
            "eqf_equ_rim_volume",
            "eqf_equ_shell_unpack",
            "eqf_equ_correct_start",
            "eqf_equ_theme_biology",
            "eqf_equ_theme_signal",
            "tokyo_eye_equ_theme_biology",
            "tokyo_eye_equ_theme_signal",
            "champion",
            "@champion",
            "c1_topology",
            # lift_radius SE3-lite lineage — not this card's spine_init
            "eqf_equ_lift_radius",
            "tokyo_eye_equ_lift_radius",
        ]
    )
)


def assert_not_forbidden_ckpt(path: Path | str) -> None:
    p = Path(path)
    text = str(p)
    # Explicit QUALIFIED geoopt_restore best is the only allowed warm spine.
    if "eqf_equ_geoopt_restore" in text and "tokyoeye_best" in text:
        return
    for token in FORBIDDEN_CKPT_SUBSTRINGS:
        if token and token in text:
            raise RuntimeError(f"forbidden ckpt token {token!r} in {text}")
    _assert_not_forbidden_lift(p)


def theme_dehydron_auprc(
    *,
    system: nn.Module,
    batch: Mapping[str, Any],
    tau_ceiling: float,
) -> dict[str, float]:
    system.eval()
    with torch.no_grad():
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=float(tau_ceiling),
        )
        scores = torch.sigmoid(out["mechanism_score"]).detach()
        labels = batch["dehydron_labels"].detach().float()
        auprc = binary_auprc(scores, labels)
        hard = out["moe_aux"]["routing"].argmax(dim=-1)
        n = int(hard.numel())
        masses = {f"hard_e{i}": float((hard == i).sum().item()) / max(n, 1) for i in range(4)}
        r = out["z_hyp"].norm(dim=-1)
    return {
        "theme_auprc": float(auprc) if math.isfinite(float(auprc)) else float("nan"),
        "dehydron_frac": float(labels.mean().item()) if labels.numel() else float("nan"),
        "r_mean": float(r.mean().item()),
        "r_std": float(r.std(unbiased=False).item()) if r.numel() > 1 else 0.0,
        **masses,
    }


def evaluate_theme_restore(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    theme_floor: float = TB_THEME_AUPRC_FLOOR,
    mean_floor: float = TB_MEAN_THEME_AUPRC,
    collapse_floor: float = TB_THEME_COLLAPSE,
    min_pass: int = TB_MIN_THEMES_PASS,
) -> dict[str, Any]:
    loaded = [r for r in probe_rows if r.get("loaded")]
    auprcs = []
    per_theme = []
    for r in loaded:
        a = r.get("theme_auprc")
        if a is None or not math.isfinite(float(a)):
            per_theme.append(
                {
                    "theme": r.get("theme"),
                    "pdb_id": r.get("pdb_id"),
                    "auprc": None,
                    "pass_floor": False,
                }
            )
            continue
        av = float(a)
        auprcs.append(av)
        per_theme.append(
            {
                "theme": r.get("theme"),
                "pdb_id": r.get("pdb_id"),
                "auprc": av,
                "pass_floor": av >= float(theme_floor),
            }
        )
    n_pass = sum(1 for t in per_theme if t.get("pass_floor"))
    mean_a = float(sum(auprcs) / len(auprcs)) if auprcs else 0.0
    min_a = float(min(auprcs)) if auprcs else 0.0
    return {
        "n_themes_loaded": len(loaded),
        "n_themes_with_auprc": len(auprcs),
        "n_themes_pass_floor": n_pass,
        "mean_theme_auprc": mean_a,
        "min_theme_auprc": min_a,
        "per_theme": per_theme,
        "theme_auprc_coverage_gate": n_pass >= int(min_pass) and len(auprcs) >= int(min_pass),
        "mean_theme_auprc_gate": mean_a >= float(mean_floor) and len(auprcs) > 0,
        "theme_auprc_floor_gate": (min_a >= float(collapse_floor)) if auprcs else False,
        "thresholds": {
            "theme_floor": theme_floor,
            "mean_floor": mean_floor,
            "collapse_floor": collapse_floor,
            "min_pass": min_pass,
        },
    }


def build_theme_restore_stamp(
    *,
    pure_hyp: Mapping[str, Any],
    hygiene: Mapping[str, Any],
    biology: Mapping[str, Any],
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
        "equiv_residual_gate": {
            "pass": bool(equiv_ok),
            "value": float(equiv_residual) if equiv_residual is not None else float("nan"),
            "threshold": RV_EQUIV_CEILING,
            "scope": "lift_spine_rotate_v",
        },
        "theme_auprc_coverage_gate": {
            "pass": bool(biology.get("theme_auprc_coverage_gate")),
            "value": float(biology.get("n_themes_pass_floor", 0)),
            "threshold": float(TB_MIN_THEMES_PASS),
            "detail": f">= {TB_MIN_THEMES_PASS}/6 themes AUPRC>={TB_THEME_AUPRC_FLOOR}",
        },
        "mean_theme_auprc_gate": {
            "pass": bool(biology.get("mean_theme_auprc_gate")),
            "value": float(biology.get("mean_theme_auprc", 0.0)),
            "threshold": TB_MEAN_THEME_AUPRC,
        },
        "theme_auprc_floor_gate": {
            "pass": bool(biology.get("theme_auprc_floor_gate")),
            "value": float(biology.get("min_theme_auprc", 0.0)),
            "threshold": TB_THEME_COLLAPSE,
        },
    }
    # Soft MoE logged, not sealed
    moe_diag = {
        "moe_liveness_soft_h_norm": float(hygiene.get("mean_h_norm", 0.0)),
        "moe_liveness_threshold_watch": RV_H_NORM_FLOOR,
        "sealed": False,
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
            "biology": dict(biology),
            "moe_diagnostic": moe_diag,
            "equiv_diagnostics": {
                "lift_spine_residual": equiv_residual,
                "full_system_residual": full_system_residual,
            },
            "pure_hyp": {
                "pass": bool(pure_hyp.get("pure_hyp_pass")),
                "findings": pure_hyp.get("findings"),
            },
            "biology_pass": bool(all_pass),
            "affinity_pass": False,
            "artifacts": [
                "docs/superpowers/specs/2026-09-16-tokyoeye-equ-theme-restore-design.md",
                "docs/superpowers/plans/2026-09-16-tokyoeye-equ-theme-restore.md",
                "experiments/training/v8/run_tokyo_eye_equ_theme_restore.py",
            ],
        }
    )
    if extra:
        stamp.update(dict(extra))
    return stamp


__all__ = [
    "DEFAULT_FRONTEND_CKPT",
    "DEFAULT_MANIFEST",
    "DEFAULT_PINS",
    "DEFAULT_SPINE_INIT",
    "DEFAULT_STAMP",
    "DISPLAY_LINEAGE",
    "FORBIDDEN_CKPT_SUBSTRINGS",
    "GATE_ID",
    "MLFLOW_EXPERIMENT",
    "MLFLOW_RUN_NAME",
    "PINNED_FRONTEND_SHA256",
    "QuadraticRadiusController",
    "RV_BOUNDARY_RADIUS",
    "RV_EPOCHS",
    "RV_EQUIV_CEILING",
    "RV_LR_BACKBONE",
    "RV_LR_HYP",
    "RV_MIN_TRAIN",
    "RV_NUM_EXPERTS",
    "RV_PROBE_EVERY",
    "RV_SAT_CEILING",
    "RV_SPREAD_FLOOR",
    "RV_TAU_CLAMP_MODE",
    "RV_TAU_END",
    "RV_TAU_START",
    "RV_VOLUME_COEFF",
    "RV_VOLUME_LAM",
    "RV_VOLUME_LAM_BARRIER",
    "RV_VOLUME_LAM_MEAN",
    "RV_VOLUME_MU",
    "RV_VOLUME_SIGMA",
    "RV_WEIGHT_DECAY",
    "RV_WRAP_MAX",
    "TAG",
    "TB_MEAN_THEME_AUPRC",
    "TB_MIN_THEMES_PASS",
    "TB_THEME_AUPRC_FLOOR",
    "TB_THEME_COLLAPSE",
    "ThemeRestoreGuardError",
    "CorrectStartGuardError",
    "assert_frontend_bank",
    "assert_not_forbidden_ckpt",
    "assert_pure_hyp_strict",
    "build_theme_restore_stamp",
    "enrich_rim_probe_row",
    "epoch_is_best_eligible",
    "evaluate_lift_spine_equivariance",
    "evaluate_probe_hygiene",
    "evaluate_theme_restore",
    "load_boot_split",
    "load_pins",
    "normalized_routing_entropy",
    "theme_dehydron_auprc",
]
