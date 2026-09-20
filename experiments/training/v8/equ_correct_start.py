"""Tokyo Eye EQU correct_start — gates, pins, pure-hyp helpers (CPU-safe)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from experiments.training.v8.equ_cold_boot import (
    FORBIDDEN_CKPT_SUBSTRINGS,
    PINNED_FRONTEND_SHA256,
    SHOCK_PDBS,
    assert_frontend_bank,
    assert_not_forbidden_ckpt,
    load_boot_split,
    sha256_path,
)
from science.tokyo_eye.v8.engine import CurriculumRadiusController

GATE_ID = "tokyo_eye_equ_correct_start"
DISPLAY_LINEAGE = "Tokyo Eye EQU"
MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine"
MLFLOW_RUN_NAME = "tokyo_eye_equ_correct_start"
DEFAULT_MANIFEST = Path("manifests/equ_cold_boot_v1.json")
DEFAULT_PINS = Path("data/gates/tokyo_eye_equ_correct_start_pins.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_equ_correct_start.json")
DEFAULT_FRONTEND_CKPT = Path(
    "checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt"
)

# Plan §5 / sealed gates
CS_EPOCHS = 20
CS_PROBE_EVERY = 5
CS_TAU_START = 0.60
CS_TAU_END = 0.95
CS_WRAP_MAX = 1
CS_BOUNDARY_RADIUS = 0.80
CS_LR_HYP = 3.0e-4
CS_LR_BACKBONE = 1.0e-5
CS_WEIGHT_DECAY = 1.0e-4
CS_SAT_CEILING = 0.50
CS_SPREAD_FLOOR = 0.15
CS_H_NORM_FLOOR = 0.60
CS_EQUIV_CEILING = 1.0e-5
CS_MIN_TRAIN = 6
CS_BEST_MOE_MIN = 0.02
CS_NUM_EXPERTS = 4


class CorrectStartGuardError(RuntimeError):
    """Refuses start when pure-hyp / pin / forbidden init checks fail."""


class QuadraticRadiusController(CurriculumRadiusController):
    """τ_ceiling(e) = τ_start + (τ_max - τ_start) * (e/E)^2 (plan §5.2)."""

    def tau_ceiling(self, epoch: int) -> float:
        e = max(0, min(int(epoch), self.total_epochs))
        t = (e / float(self.total_epochs)) ** 2
        return self.tau_start + t * (self.tau_end - self.tau_start)


def load_pins(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_PINS
    if not p.is_file():
        return {
            "frontend_ckpt": str(DEFAULT_FRONTEND_CKPT),
            "frontend_bank_sha256": PINNED_FRONTEND_SHA256,
            "source_manifest": str(DEFAULT_MANIFEST),
        }
    return json.loads(p.read_text())


def static_pure_hyp_pass(
    paths: Sequence[Path | str] | None = None,
) -> dict[str, Any]:
    """AST/source veto for post-lift tangent geometry substitutes."""
    targets = [Path(p) for p in paths] if paths else [
        Path("science/tokyo_eye/v8/attention.py"),
        Path("science/tokyo_eye/v8/affinity_head.py"),
    ]
    forbidden = (
        "_tangent_linear",
        "def _tangent_linear",
        "TangentLinear",
        "TangentPool",
        "use_tangent_shortcut=True",
        "u_self + u_att",
        "W_o(log_map_zero",
        "t_pool",
    )
    findings: dict[str, list[str]] = {}
    all_hits: list[str] = []
    missing: list[str] = []
    for path in targets:
        if not path.is_file():
            missing.append(str(path))
            continue
        src = path.read_text(encoding="utf-8")
        hits = [tok for tok in forbidden if tok in src]
        # Linear-on-log0 classic
        if "lin(log_map_zero" in src.replace(" ", ""):
            hits.append("lin(log_map_zero)")
        findings[str(path)] = sorted(set(hits))
        all_hits.extend(f"{path.name}:{h}" for h in findings[str(path)])
    passed = (not missing) and (not all_hits)
    return {
        "pure_hyp_pass": passed,
        "findings": findings,
        "missing": missing,
        "hits": all_hits,
    }


def assert_pure_hyp_strict(system: nn.Module | None = None) -> dict[str, Any]:
    """Hard veto: static sources + live module must not carry tangent Linear path."""
    report = static_pure_hyp_pass()
    if not report["pure_hyp_pass"]:
        raise CorrectStartGuardError(
            f"pure_hyp_strict FAIL static: {report['hits'] or report['missing']}"
        )
    if system is not None:
        for name, mod in system.named_modules():
            if hasattr(mod, "_tangent_linear"):
                raise CorrectStartGuardError(
                    f"pure_hyp_strict FAIL live module has _tangent_linear: {name}"
                )
            cls = mod.__class__.__name__
            if cls in {"TangentLinear", "TangentPool"}:
                raise CorrectStartGuardError(
                    f"pure_hyp_strict FAIL live class {cls} at {name}"
                )
            if getattr(mod, "use_tangent_shortcut", False):
                raise CorrectStartGuardError(
                    f"pure_hyp_strict FAIL use_tangent_shortcut at {name}"
                )
        # Expect GyroOrthogonalMap on attention layers
        has_gyro = any(
            m.__class__.__name__ == "GyroOrthogonalMap" for _, m in system.named_modules()
        )
        if not has_gyro:
            raise CorrectStartGuardError(
                "pure_hyp_strict FAIL: no GyroOrthogonalMap found in live system"
            )
    report["live_ok"] = True
    return report


def normalized_routing_entropy(
    routing: torch.Tensor, *, num_experts: int = CS_NUM_EXPERTS
) -> float:
    """H_norm in [0, 1] from mean expert usage (plan §4.3)."""
    if routing.ndim != 2:
        raise ValueError("routing must be [N, K]")
    avg = routing.detach().float().mean(dim=0).clamp_min(1e-9)
    ent = float(-(avg * avg.log()).sum().item())
    max_ent = math.log(float(num_experts))
    if max_ent <= 0:
        return 0.0
    return float(ent / max_ent)


def hyp_radius(z: torch.Tensor, *, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """r_H = (1/√c) log((1+√c||x||)/(1-√c||x||))."""
    r = torch.linalg.vector_norm(z, dim=-1).clamp(max=(1.0 / math.sqrt(c)) - 1e-4)
    sqrt_c = math.sqrt(c)
    return (1.0 / sqrt_c) * torch.log((1.0 + sqrt_c * r) / (1.0 - sqrt_c * r + eps))


def evaluate_probe_hygiene(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    sat_ceiling: float = CS_SAT_CEILING,
    spread_floor: float = CS_SPREAD_FLOOR,
    h_norm_floor: float = CS_H_NORM_FLOOR,
) -> dict[str, Any]:
    loaded = [r for r in probe_rows if r.get("loaded")]
    sats = [
        float(r["boundary_saturation"])
        for r in loaded
        if r.get("boundary_saturation") is not None
    ]
    spreads = [
        float(r["radius_spread"])
        for r in loaded
        if r.get("radius_spread") is not None and math.isfinite(float(r["radius_spread"]))
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
    sat_pass = mean_sat < float(sat_ceiling)
    spread_pass = mean_spread > float(spread_floor)
    h_pass = mean_h >= float(h_norm_floor)
    finite_pass = finite and len(loaded) == 6
    return {
        "n_probe_loaded": len(loaded),
        "mean_sat": mean_sat,
        "mean_spread": mean_spread,
        "mean_h_norm": mean_h,
        "finite_h2": finite_pass,
        "probe_sat_gate": sat_pass,
        "radius_spread_gate": spread_pass,
        "moe_liveness_gate": h_pass,
        "finite_h2_gate": finite_pass,
    }


@torch.no_grad()
def evaluate_equivariance_residual(
    system: nn.Module,
    batch: Mapping[str, Any],
    *,
    tau_ceiling: float,
    seed: int = 0,
) -> float:
    """max ||z(x) - z(R·x)|| on Poincaré features (SO(3) invariance check)."""
    system.eval()
    device = batch["x"].device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    r, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    if torch.det(r) < 0:
        r = r.clone()
        r[:, 0] = -r[:, 0]
    r = r.to(device=device, dtype=batch["x"].dtype)

    out0 = system(
        batch["x"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceiling,
    )
    x_rot = batch["x"] @ r.T
    out1 = system(
        x_rot,
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceiling,
    )
    diff = torch.linalg.vector_norm(out0["z_hyp"] - out1["z_hyp"], dim=-1)
    return float(diff.max().item())


def epoch_is_best_eligible(last_step: Mapping[str, Any]) -> bool:
    if float(last_step.get("nan_abort") or 0.0) >= 1.0:
        return False
    loss = last_step.get("loss_total")
    if loss is None or not math.isfinite(float(loss)):
        return False
    mean_r = last_step.get("diag_mean_radius")
    z_finite = mean_r is not None and math.isfinite(float(mean_r))
    moe_min = last_step.get("moe_load_min")
    return bool(z_finite) and moe_min is not None and float(moe_min) >= CS_BEST_MOE_MIN


def build_correct_start_stamp(
    *,
    pure_hyp: Mapping[str, Any],
    hygiene: Mapping[str, Any],
    equiv_residual: float | None,
    frontend_sha256: str,
    mlflow_run_id: str | None,
    extra: Mapping[str, Any] | None = None,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    equiv_ok = (
        equiv_residual is not None
        and math.isfinite(float(equiv_residual))
        and float(equiv_residual) < CS_EQUIV_CEILING
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
            "threshold": CS_SAT_CEILING,
        },
        "radius_spread_gate": {
            "pass": bool(hygiene.get("radius_spread_gate")),
            "value": float(hygiene.get("mean_spread", 0.0)),
            "threshold": CS_SPREAD_FLOOR,
        },
        "finite_h2_gate": {
            "pass": bool(hygiene.get("finite_h2_gate")),
            "value": 1.0 if hygiene.get("finite_h2_gate") else 0.0,
            "threshold": 1.0,
        },
        "moe_liveness_gate": {
            "pass": bool(hygiene.get("moe_liveness_gate")),
            "value": float(hygiene.get("mean_h_norm", 0.0)),
            "threshold": CS_H_NORM_FLOOR,
        },
        "equiv_residual_gate": {
            "pass": bool(equiv_ok),
            "value": float(equiv_residual) if equiv_residual is not None else float("nan"),
            "threshold": CS_EQUIV_CEILING,
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
            "execution_state": "TRAIN_COMPLETE_QUALIFIED" if all_pass else "TRAIN_COMPLETE_FAILED",
            "mlflow_experiment": MLFLOW_EXPERIMENT,
            "mlflow_run_id": mlflow_run_id,
            "frontend_sha256": frontend_sha256,
            "gates": gates,
            "hygiene": dict(hygiene),
            "pure_hyp": {
                "pass": bool(pure_hyp.get("pure_hyp_pass")),
                "findings": pure_hyp.get("findings"),
            },
            "artifacts": [
                "docs/superpowers/specs/2026-09-15-tokyoeye-equ-correct-start-design.md",
                "docs/superpowers/plans/2026-09-15-tokyoeye-equ-correct-start.md",
                "experiments/training/v8/run_tokyo_eye_equ_correct_start.py",
            ],
        }
    )
    if extra:
        stamp["run_extra"] = dict(extra)
    # Preserve approval metadata
    if prior:
        for k in ("approved_at", "approver", "design", "plan", "governance"):
            if k in prior:
                stamp[k] = prior[k]
        if prior.get("status") in {"APPROVED_LOCKED", "APPROVED"}:
            stamp["approval_status"] = "APPROVED_LOCKED"
    return stamp


__all__ = [
    "CS_EPOCHS",
    "CS_LR_BACKBONE",
    "CS_LR_HYP",
    "CS_TAU_END",
    "CS_TAU_START",
    "CorrectStartGuardError",
    "DEFAULT_FRONTEND_CKPT",
    "DEFAULT_MANIFEST",
    "DEFAULT_STAMP",
    "DISPLAY_LINEAGE",
    "GATE_ID",
    "MLFLOW_EXPERIMENT",
    "MLFLOW_RUN_NAME",
    "PINNED_FRONTEND_SHA256",
    "QuadraticRadiusController",
    "assert_frontend_bank",
    "assert_not_forbidden_ckpt",
    "assert_pure_hyp_strict",
    "build_correct_start_stamp",
    "epoch_is_best_eligible",
    "evaluate_equivariance_residual",
    "evaluate_probe_hygiene",
    "load_boot_split",
    "load_pins",
    "normalized_routing_entropy",
    "sha256_path",
    "static_pure_hyp_pass",
]
