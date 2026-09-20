#!/usr/bin/env python3
"""TokyoEye MLflow experiment harness (Sprint 5).

Standing rule: prose without a failing test always loses to a convenient default.

Isolated under ``experiments/training/v8/``. Does not load v7/v66 checkpoints.

Assembly gate (Architecture SSOT — ENFORCED on the governed path):
  1. ``--frontend equiformer_pool`` (default). SE(3)-lite / stub fails unless
     ``--allow-off-path-frontend`` is set explicitly (non-claim diagnostics /
     pre-registered cheap signal-existence pilots only).
  2. Live-forward ``pure_hyp_pass`` tracer must pass (``pure_hyp_ok`` only if
     checked; skip under off-path is not a clean seal).
  3. ``--claim-bearing-biology`` refuses unless the batch is *structurally*
     non-leaking (targets do not reconstruct from ``edge_type``) and
     ``--log-biology-grad-sources N>0``. A ``--biology-non-leaking-target``
     flag is advisory only and cannot waive a still-leaking batch.
  4. Pool constructibility deps (ase / torch-scatter / torch-cluster / lmdb / e3nn)
     are checked with an explicit missing-deps message.

``--frontend`` selects the Euclidean frontend:
  * ``equiformer_pool`` (default, governed) — ``EquiformerPoolFrontend`` (real
    EquiformerV3 cut before energy_block; SE(3)-lite forbidden). Cold random-init
    (§2.3) unless ``--frontend-warm-mptrj`` + ``--equiformer-ckpt``.
  * ``stub`` — ``StubEquiformerFrontend`` / live SE(3)-lite; requires
    ``--allow-off-path-frontend``.

Default Equiformer path (warm / sealed cards):
  ``checkpoints/v8/pretrained/equiformer_v3_baseline.pt``
Weight map:
  ``science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json``

See ``docs/TOKYOEYE_ARCHITECTURE_SSOT.md`` and ``science/tokyo_eye/v8/assembly_gate.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
)
from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_CKPT,
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    apply_weight_map,
    build_param_groups,
    evaluate_geometry_health,
    load_weight_map,
)
from science.tokyo_eye.v8.assembly_gate import (
    AssemblyGateError,
    assert_governed_assembly,
)
from science.tokyo_eye.v8.biology_grad_sources import (
    dehydron_grad_by_source,
    sdrp_grad_by_source,
)
from science.tokyo_eye.v8.heads import mechanism_margin_loss_v2, sdrp_cross_entropy
from science.tokyo_eye.v8.loader import (
    DEFAULT_CHAIN,
    DEFAULT_PDB_DIR,
    DEFAULT_PDB_ID,
    TokyoEyeCuratedDataset,
    load_structure_batch,
)
from science.tokyo_eye.v8.metrics import binary_auprc
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.r0_r5_graph import (
    R0_COVALENT,
    R5_LOCAL_NEIGHBORHOOD,
    build_r0_r5_graph,
    chemistry_gate_features,
    get_dehydron_wrap_max,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord
from science.tokyo_eye.v8.freeze_reconciliation import (
    CANONICAL_MLFLOW_EXPERIMENT,
    freeze_reconciliation_mlflow_params,
    resolve_v8_mlflow_experiment,
)
from experiments.training.v8.equ_lift_radius import (
    RV_EQUIV_CEILING,
    evaluate_lift_spine_equivariance,
)
from science.tokyo_eye.v8.moe_eval_gates import eval_moe_utilization

MLFLOW_EXPERIMENT_DEFAULT = CANONICAL_MLFLOW_EXPERIMENT
SDRP_LOSS_COEFF = 0.1
# §2.1: eval MoE utilization — SSOT is ``moe_eval_gates.eval_moe_utilization``.
# Do not invent a second threshold set in this harness.
EVAL_LOAD_FLOOR = 0.10
EVAL_ENTROPY_NORM_FLOOR = 0.85
EVAL_N_ALIVE_REQUIRED = 4
EVAL_ALIVE_FLOOR = 0.05
# Schema id for logged eval MoE keys. Do not reuse retired names
# (``eval_moe_h_norm`` meant embedding-norm health under EQU cards; §2.5).
EVAL_MOE_METRIC_SCHEMA = "moe_eval_utilization_v1"


def _mlflow_uri_reachable(uri: str, *, timeout_s: float = 2.0) -> bool:
    """Fail fast when the tracking server is down (avoid multi-minute hang)."""
    from urllib.parse import urlparse
    from urllib.request import urlopen

    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return True  # file: / other backends — let mlflow handle
    probe = f"{parsed.scheme}://{parsed.netloc}/"
    try:
        with urlopen(probe, timeout=float(timeout_s)) as resp:  # noqa: S310
            return int(getattr(resp, "status", 200) or 200) < 500
    except Exception:  # noqa: BLE001
        return False


def _dual_seal_diagnostics(
    diagnostics: PoincareDiagnosticsEngine,
    out: dict[str, Any],
) -> dict[str, float]:
    """§2.2: tag sat/spread/entropy by representation (lift / attn / post-MoE)."""
    metrics: dict[str, float] = {}
    for tag, key in (
        ("pre_moe_lift", "z_lift"),
        ("pre_moe_attn", "z_attn"),
        ("post_moe", "z_hyp"),
    ):
        tensor = out.get(key)
        if tensor is None:
            continue
        diag = diagnostics.summarize(tensor)
        for k, val in diag.items():
            metrics[f"diag_{tag}_{k}"] = (
                float(val) if not isinstance(val, bool) else float(val)
            )
    # Back-compat aliases = deployed (post-MoE) embedding only.
    post = diagnostics.summarize(out["z_hyp"])
    for k, val in post.items():
        metrics[f"diag_{k}"] = float(val) if not isinstance(val, bool) else float(val)
    metrics["diag_boundary_saturation_pct"] = metrics["diag_boundary_saturation"] * 100.0
    metrics["diag_manifold_entropy"] = metrics["diag_radial_entropy"]
    return metrics


@torch.no_grad()
def _eval_mode_routing_metrics(
    system: TokyoEyeV8WithFrontend,
    batch: dict[str, Any],
    *,
    tau_ceiling: float,
    load_floor: float = EVAL_LOAD_FLOOR,
    entropy_norm_floor: float = EVAL_ENTROPY_NORM_FLOOR,
    n_alive_required: int = EVAL_N_ALIVE_REQUIRED,
    alive_floor: float = EVAL_ALIVE_FLOOR,
) -> dict[str, Any]:
    """§2.1: eval argmax routing — single call site via ``eval_moe_utilization``.

    Logged keys use ``eval_moe_entropy_norm`` only. The retired name
    ``eval_moe_h_norm`` (embedding-norm health under EQU cards; addendum
    §2.5) is **not** emitted — do not alias or repoint it.
    """
    was_training = system.training
    system.eval()
    out = system(
        batch["x"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceiling,
        chem=batch.get("gate_chem"),
    )
    if was_training:
        system.train()
    routing = out["moe_aux"]["routing"]
    util = eval_moe_utilization(
        routing,
        min_load=float(load_floor),
        alive_floor=float(alive_floor),
        n_alive_required=int(n_alive_required),
        entropy_norm_floor=float(entropy_norm_floor),
    )
    load_list = list(util["load"])
    gates = util["gates"]
    ent_norm = float(gates["moe_eval_entropy_norm"]["value"])
    metrics: dict[str, Any] = {
        "eval_moe_metric_schema": EVAL_MOE_METRIC_SCHEMA,
        "eval_moe_load_min": float(gates["moe_eval_min_load"]["value"]),
        "eval_moe_n_alive": float(gates["moe_eval_n_alive"]["value"]),
        "eval_moe_entropy_norm": ent_norm,
        "eval_moe_load_floor": float(load_floor),
        "eval_moe_entropy_norm_floor": float(entropy_norm_floor),
        "eval_moe_n_alive_required": float(n_alive_required),
        "eval_moe_load_floor_pass": (
            1.0 if gates["moe_eval_min_load"]["pass"] else 0.0
        ),
        "eval_moe_entropy_norm_pass": (
            1.0 if gates["moe_eval_entropy_norm"]["pass"] else 0.0
        ),
        "eval_moe_n_alive_pass": (
            1.0 if gates["moe_eval_n_alive"]["pass"] else 0.0
        ),
        # Liveness = full moe_eval_utilization seal (min_load ∧ n_alive ∧ entropy).
        "eval_moe_liveness_pass": 1.0 if util["all_pass"] else 0.0,
    }
    for i, v_load in enumerate(load_list):
        metrics[f"eval_moe_load_e{i}"] = float(v_load)
    return metrics


def _log_delta_equiv(
    system: TokyoEyeV8WithFrontend,
    batch: dict[str, Any],
    *,
    tau_ceiling: float,
    seed: int = 0,
) -> dict[str, float]:
    """Δ_equiv on the harness path (logged; lift_spine not a smoke veto yet).

    ``lift_spine`` freezes frontend ``(s,v)`` and rotates ``v`` only. Option B
    lift uses ``â`` from ``v``, so a generic ``Linear(vector→hidden)`` moves
    ``z_hyp`` by construction — the ``1e-5`` ceiling is not a fair spine gate
    under that protocol. ``full_system`` with ``--frontend stub`` routes through
    ``live_se3_lite``; with ``--frontend equiformer_pool`` it uses the real
    EquiformerV3 cut (still diagnostic for Δ_equiv until protocol AMEND).
    """
    eq = evaluate_lift_spine_equivariance(
        system, batch, tau_ceiling=tau_ceiling, seed=seed
    )
    lift = float(eq["lift_spine_residual"])
    full = float(eq["full_system_residual"])
    return {
        "delta_equiv_lift_spine": lift,
        "delta_equiv_full_system": full,
        "delta_equiv_ceiling": float(RV_EQUIV_CEILING),
        # Diagnostic only until protocol AMEND or real Equiformer bind.
        "delta_equiv_lift_spine_pass": (
            1.0
            if math.isfinite(lift) and lift < float(RV_EQUIV_CEILING)
            else 0.0
        ),
        "delta_equiv_is_smoke_veto": 0.0,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synthetic_residues(n: int = 24) -> list[ResidueRecord]:
    records: list[ResidueRecord] = []
    for i in range(n):
        x = float(i) * 3.8
        records.append(
            ResidueRecord(
                chain_label="A",
                residue_index=i + 1,
                residue_name="ALA",
                atoms=(
                    AtomRecord("N", "N", np.array([x, 0.0, 0.0])),
                    AtomRecord("CA", "C", np.array([x + 1.5, 0.0, 0.0])),
                    AtomRecord("C", "C", np.array([x + 2.5, 0.0, 0.0])),
                    AtomRecord("O", "O", np.array([x + 2.5, 1.2, 0.0])),
                    AtomRecord("CB", "C", np.array([x + 1.5, 1.5, 0.0])),
                ),
            )
        )
    return records


def _ca_features(records: list[ResidueRecord]) -> torch.Tensor:
    rows = []
    for r in records:
        ca = r.get_atom("CA")
        assert ca is not None
        rows.append(ca.coord.astype(np.float32))
    return torch.from_numpy(np.stack(rows, axis=0))


def _make_batch(device: torch.device) -> dict[str, Any]:
    records = _synthetic_residues(24)
    graph = build_r0_r5_graph(records)
    x = _ca_features(records).to(device)
    n = x.shape[0]
    # Dehydron-ish binary labels from edge types touching node (synthetic)
    et = torch.tensor(graph.edge_type, dtype=torch.long, device=device)
    ei = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    labels = torch.zeros(n, device=device)
    if et.numel():
        r2 = (et == 2).nonzero(as_tuple=False).view(-1)
        if r2.numel():
            labels[ei[0, r2]] = 1.0
            labels[ei[1, r2]] = 1.0
    # Ensure some positives for AUPRC
    if float(labels.sum()) < 1:
        labels[: max(1, n // 4)] = 1.0
    chem = chemistry_gate_features(
        n, graph.edge_index, graph.edge_type, x.detach().cpu().numpy()
    )
    return {
        "x": x,
        "edge_index": ei,
        "edge_type": et,
        "sdrp_target": torch.randint(0, 5, (n,), device=device),
        "mechanism_pos": torch.rand(n, device=device) * 0.5 + 0.5,
        "mechanism_neg": torch.rand(n, device=device) * 0.4,
        "dehydron_labels": labels,
        "gate_chem": torch.from_numpy(chem).to(device),
        "num_nodes": n,
        "graph_meta": graph.meta,
    }


def build_system(
    cfg: dict[str, Any],
    *,
    equiformer_ckpt: Path | None,
    device: torch.device,
    freeze_backbone: bool = False,
) -> tuple[TokyoEyeV8WithFrontend, dict[str, Any]]:
    scalar_dim = int(cfg["scalar_dim"])
    vector_dim = int(cfg["vector_dim"])
    hidden_dim = int(cfg["hidden_dim"])
    live = not bool(freeze_backbone)
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
        live_backbone=live,
    )
    load_info: dict[str, Any] = {"mode": "stub_random_init", "backbone_mode": "stub"}
    ckpt = equiformer_ckpt
    # Addendum §2.3: cold random-init is the default frontend. Do not auto-load
    # MPtrj from checkpoint_path_default.
    if ckpt is not None and Path(ckpt).is_file():
        load_info = apply_weight_map(frontend, ckpt, cfg)
        load_info["mode"] = "weight_map_loaded"
        load_info["checkpoint"] = str(ckpt)
    elif ckpt is not None:
        load_info = {
            "mode": "stub_missing_ckpt",
            "requested": str(ckpt),
            "fallback": "StubEquiformerFrontend",
        }
    load_info["backbone_mode"] = "live_se3_lite" if live else "frozen_stub"
    load_info["live_backbone"] = live

    spine = TokyoEyesHyperbolicV8(
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        hidden_dim=hidden_dim,
        num_attn_layers=2,
        num_sdrp_classes=5,
        c=float(cfg.get("curvature_c", 1.0)),
        moe_temperature=float(cfg.get("gumbel_tau_start", 1.0)),
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    return system, load_info




def build_equiformer_pool_system(
    cfg: dict[str, Any],
    *,
    equiformer_ckpt: Path | None,
    device: torch.device,
    freeze_backbone: bool = True,
    max_neighbors: int | None = 50,
    cold_init: bool = False,
) -> tuple[TokyoEyeV8WithFrontend, dict[str, Any]]:
    """EquiformerV3-to-pool builder (geoopt_restore / freeze-recon frontend).

    Forbids StubEquiformerFrontend / SE(3)-lite on this path.
    Addendum §2.3: ``cold_init=True`` builds the architecture with random
    weights (no MPtrj warm start). Warm MPtrj load requires a real ckpt and
    ``cold_init=False``.
    """
    from science.tokyo_eye.v8.equiformer_pool_frontend import EquiformerPoolFrontend

    ckpt: Path | None = None if equiformer_ckpt is None else Path(equiformer_ckpt)
    if not cold_init:
        if ckpt is None or not ckpt.is_file():
            raise FileNotFoundError(
                f"Equiformer MPtrj ckpt missing: {equiformer_ckpt!s}"
            )
    elif ckpt is not None and not ckpt.is_file():
        # Cold path ignores a missing optional path; architecture only.
        ckpt = None

    scalar_dim = int(cfg["scalar_dim"])
    vector_dim = int(cfg["vector_dim"])
    hidden_dim = int(cfg["hidden_dim"])
    frontend = EquiformerPoolFrontend(
        equiformer_ckpt=ckpt,
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        freeze=bool(freeze_backbone),
        max_neighbors=max_neighbors,
        cold_init=bool(cold_init),
    )
    load_info = dict(frontend.load_info())
    load_info["backbone_mode"] = "equiformer_v3_pool"
    load_info["live_backbone"] = False
    load_info["forbid_se3_lite"] = True
    load_info["cold_init"] = bool(cold_init)
    if not cold_init and (
        int(load_info.get("missing", -1)) != 0
        or int(load_info.get("unexpected", -1)) != 0
    ):
        raise RuntimeError(
            f"MPtrj load not clean: missing={load_info.get('missing')} "
            f"unexpected={load_info.get('unexpected')}"
        )

    spine = TokyoEyesHyperbolicV8(
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        hidden_dim=hidden_dim,
        num_attn_layers=2,
        num_sdrp_classes=5,
        c=float(cfg.get("curvature_c", 1.0)),
        moe_temperature=float(cfg.get("gumbel_tau_start", 1.0)),
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    return system, load_info


def hyperbolic_volume_loss(
    z: torch.Tensor,
    *,
    c: float = 1.0,
    sigma_target: float = 0.20,
    lam_spread: float = 0.5,
    eps: float = 1e-7,
    mu_target: float | None = None,
    lam_mean: float = 0.0,
    lam_barrier: float = 1.0,
) -> torch.Tensor:
    """Radial unpack: optional mean-radius target + spread hinge + barrier."""
    r2 = (z * z).sum(dim=-1).clamp(max=(1.0 / c) - 1e-4)
    barrier = -torch.log((1.0 - float(c) * r2).clamp_min(eps)).mean()
    r = torch.sqrt(r2.clamp_min(0.0)).clamp(max=(1.0 / math.sqrt(c)) - 1e-4)
    sqrt_c = math.sqrt(float(c))
    r_h = (1.0 / sqrt_c) * torch.log((1.0 + sqrt_c * r) / (1.0 - sqrt_c * r + eps))
    spread = r_h.std(unbiased=False) if r_h.numel() > 1 else z.new_zeros(())
    spread_pen = float(lam_spread) * torch.relu(float(sigma_target) - spread)
    if mu_target is not None and float(lam_mean) > 0.0:
        mean_pen = float(lam_mean) * (r.mean() - float(mu_target)) ** 2
    else:
        mean_pen = z.new_zeros(())
    return float(lam_barrier) * barrier + spread_pen + mean_pen


def _gate_param_grad_norm(system: TokyoEyeV8WithFrontend) -> float:
    """L2 norm of gate_fc1+fc2 parameter grads (router audit SSOT)."""
    moe = system.spine.moe
    parts: list[torch.Tensor] = []
    for mod in (moe.gate_fc1, moe.gate_fc2):
        for p in mod.parameters():
            if p.grad is not None:
                parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return 0.0
    return float(torch.cat(parts).norm().item())


def _zero_gate_grads(system: TokyoEyeV8WithFrontend) -> None:
    for mod in (system.spine.moe.gate_fc1, system.spine.moe.gate_fc2):
        for p in mod.parameters():
            p.grad = None


def run_epoch(
    system: TokyoEyeV8WithFrontend,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    epoch: int,
    radius: CurriculumRadiusController,
    gumbel: GumbelTemperatureSchedule,
    diagnostics: PoincareDiagnosticsEngine,
    cv_coeff: float,
    moe_quota_coeff: float = 5.0,
    eval_proxy_quota_coeff: float = 140.0,
    explore_epsilon: float = 0.0,
    eps_at_floor: bool = False,
    log_gate_grad_sources: bool = False,
    log_biology_grad_sources: bool = False,
    max_grad_norm: float = 1.0,
    telemetry: dict[str, Any] | None = None,
    margin_coeff: float = 1.0,
    sdrp_coeff: float = SDRP_LOSS_COEFF,
    volume_coeff: float = 0.0,
    volume_sigma_target: float = 0.20,
    volume_lam_spread: float = 0.5,
    volume_mu_target: float | None = None,
    volume_lam_mean: float = 0.0,
    volume_lam_barrier: float = 1.0,
    dehydron_coeff: float = 1.0,
) -> dict[str, float]:
    system.train()
    tau_ceil = radius.tau_ceiling(epoch)
    gumbel_tau = gumbel.temperature(epoch)
    system.set_moe_temperature(gumbel_tau)
    system.set_moe_explore_epsilon(float(explore_epsilon))

    out = system(
        batch["x"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceil,
        chem=batch.get("gate_chem"),
    )
    biology_grad_metrics: dict[str, float] = {}
    if log_biology_grad_sources:
        # Attribute biology loss grads into z_hyp vs h_euc (Architecture SSOT).
        z_b = out["z_hyp"].detach().requires_grad_(True)
        h_b = out["h_euc"].detach().requires_grad_(True)
        spine = system.spine
        bio = dehydron_grad_by_source(
            spine.mechanism_head,
            z_hyp=z_b,
            h_euc=h_b,
            dehydron_labels=batch["dehydron_labels"],
            retain_graph=True,
        )
        for k, v in bio.items():
            if isinstance(v, (int, float)):
                biology_grad_metrics[f"dehydron_{k}"] = float(v)
        if float(sdrp_coeff) > 0.0:
            z_s = out["z_hyp"].detach().requires_grad_(True)
            h_s = out["h_euc"].detach().requires_grad_(True)
            sdrp_g = sdrp_grad_by_source(
                spine.sdrp_head,
                z_hyp=z_s,
                h_euc=h_s,
                sdrp_target=batch["sdrp_target"],
                tau_ceiling=float(tau_ceil),
                retain_graph=False,
            )
            for k, v in sdrp_g.items():
                if isinstance(v, (int, float)):
                    biology_grad_metrics[f"sdrp_{k}"] = float(v)
        print(
            f"[tokyoeye] biology_grad_sources epoch={epoch} "
            f"dehydron_hyp={biology_grad_metrics.get('dehydron_biology_grad_hyp', 0.0):.4f} "
            f"dehydron_euc={biology_grad_metrics.get('dehydron_biology_grad_euc_skip', 0.0):.4f} "
            f"dehydron_euc_share={biology_grad_metrics.get('dehydron_biology_grad_euc_share', 0.0):.3f}"
        )
        # Attribution backward must not contaminate the train step.
        optimizer.zero_grad(set_to_none=True)
    loss_sdrp = sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
    loss_margin = mechanism_margin_loss_v2(
        out["mechanism_score"],
        batch["mechanism_pos"],
        batch["mechanism_neg"],
    )
    loss_cv_raw = out["moe_aux"]["cv_loss"]
    loss_quota_raw = out["moe_aux"].get("quota_loss")
    if loss_quota_raw is None:
        loss_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_majority_raw = out["moe_aux"].get("majority_hinge_loss")
    if loss_majority_raw is None:
        loss_majority_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_switch_raw = out["moe_aux"].get("switch_lb_loss")
    if loss_switch_raw is None:
        loss_switch_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_soft_quota_raw = out["moe_aux"].get("soft_quota_loss")
    if loss_soft_quota_raw is None:
        loss_soft_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_eval_proxy_raw = out["moe_aux"].get("eval_proxy_lb_loss")
    if loss_eval_proxy_raw is None:
        loss_eval_proxy_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_eval_proxy_quota_raw = out["moe_aux"].get("eval_proxy_quota_loss")
    if loss_eval_proxy_quota_raw is None:
        loss_eval_proxy_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    # Distinct §8 mechanisms — not CV+quota alone (addendum #13 / §2.1).
    # eval_proxy_quota_coeff is independent of STE moe_quota_coeff: Jacobian
    # equalization vs Switch-LB needs ~140 (preflight), not the STE×5 scale.
    loss_balance = (
        loss_cv_raw * float(cv_coeff)
        + loss_quota_raw * float(moe_quota_coeff)
        + loss_majority_raw
        + loss_switch_raw
        + loss_soft_quota_raw * float(moe_quota_coeff)
        + loss_eval_proxy_raw
        + loss_eval_proxy_quota_raw * float(eval_proxy_quota_coeff)
    )
    # Primary biophysical gate: binary dehydron incidence via mechanism score logits.
    loss_dehydron = torch.nn.functional.binary_cross_entropy_with_logits(
        out["mechanism_score"], batch["dehydron_labels"]
    )
    if float(volume_coeff) > 0.0:
        loss_vol = hyperbolic_volume_loss(
            out["z_hyp"],
            c=float(getattr(getattr(system, "spine", system), "c", 1.0)),
            sigma_target=float(volume_sigma_target),
            lam_spread=float(volume_lam_spread),
            mu_target=volume_mu_target,
            lam_mean=float(volume_lam_mean),
            lam_barrier=float(volume_lam_barrier),
        )
    else:
        loss_vol = out["z_hyp"].new_zeros(())
    loss_task = (
        float(dehydron_coeff) * loss_dehydron
        + float(sdrp_coeff) * loss_sdrp
        + float(margin_coeff) * loss_margin
        + float(volume_coeff) * loss_vol
    )
    loss = loss_task + loss_balance

    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        metrics_fail = {
            "loss_total": float("nan"),
            "tau_ceiling": float(tau_ceil),
            "gumbel_temperature": float(gumbel_tau),
            "nan_abort": 1.0,
        }
        print(f"[tokyoeye] ERROR non-finite loss at epoch={epoch}; skipping step")
        return metrics_fail

    gate_grad_by_source: dict[str, float] = {}
    if log_gate_grad_sources:
        # Same isolation method as eval_proxy_reweight_preflight (gate param norms).
        source_terms = {
            "task": loss_task,
            "cv_STE": loss_cv_raw * float(cv_coeff),
            "quota_STE": loss_quota_raw * float(moe_quota_coeff),
            "majority_STE": loss_majority_raw,
            "switch_lb": loss_switch_raw,
            "soft_quota": loss_soft_quota_raw * float(moe_quota_coeff),
            "eval_proxy_lb": loss_eval_proxy_raw,
            "eval_proxy_quota": loss_eval_proxy_quota_raw
            * float(eval_proxy_quota_coeff),
        }
        for name, term in source_terms.items():
            _zero_gate_grads(system)
            if float(term.detach()) == 0.0:
                gate_grad_by_source[name] = 0.0
                continue
            term.backward(retain_graph=True)
            gate_grad_by_source[name] = _gate_param_grad_norm(system)
        optimizer.zero_grad(set_to_none=True)

    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(system.parameters(), max_grad_norm))
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        print(f"[tokyoeye] ERROR non-finite grad_norm at epoch={epoch}; skipping step")
        return {
            "loss_total": float(loss.detach()),
            "tau_ceiling": float(tau_ceil),
            "gumbel_temperature": float(gumbel_tau),
            "grad_norm": float("nan"),
            "nan_abort": 1.0,
        }
    optimizer.step()

    evid = out["evidence"]
    risk = (evid[:, 3] / evid[:, 2].clamp_min(1e-4)).detach()
    dehydron_score = torch.sigmoid(out["mechanism_score"]).detach()
    auprc = binary_auprc(dehydron_score, batch["dehydron_labels"])
    load = out["moe_aux"].get("load")
    if load is None:
        load = out["moe_aux"]["routing"].mean(dim=0)
    load_list = [float(x) for x in load.detach().tolist()]

    metrics = {
        "loss_total": float(loss.detach()),
        "loss_dehydron": float(loss_dehydron.detach()),
        "dehydron_coeff": float(dehydron_coeff),
        "loss_sdrp": float(loss_sdrp.detach()),
        "loss_margin": float(loss_margin.detach()),
        "loss_vol": float(loss_vol.detach()),
        "volume_coeff": float(volume_coeff),
        "loss_cv": float(loss_balance.detach()),
        "moe_quota_loss": float(loss_quota_raw.detach()),
        "moe_majority_hinge_loss": float(loss_majority_raw.detach()),
        "moe_switch_lb_loss": float(loss_switch_raw.detach()),
        "moe_soft_quota_loss": float(loss_soft_quota_raw.detach()),
        "moe_eval_proxy_lb_loss": float(loss_eval_proxy_raw.detach()),
        "moe_eval_proxy_quota_loss": float(loss_eval_proxy_quota_raw.detach()),
        "eval_proxy_quota_coeff": float(eval_proxy_quota_coeff),
        "moe_quota_coeff": float(moe_quota_coeff),
        "sdrp_coeff": float(sdrp_coeff),
        "tau_ceiling": float(tau_ceil),
        "scheduled_tau": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "explore_epsilon": float(explore_epsilon),
        "eps_at_floor": 1.0 if eps_at_floor else 0.0,
        "grad_norm": grad_norm,
        "max_grad_norm": float(max_grad_norm),
        "margin_coeff": float(margin_coeff),
        "val_dehydron_auprc": float(auprc),
        "epistemic_risk_mean": float(risk.mean()),
        "dehydron_frac": float(batch.get("dehydron_frac", batch["dehydron_labels"].mean())),
        "num_nodes": float(batch.get("num_nodes", batch["x"].shape[0])),
        "moe_load_min": float(min(load_list) if load_list else 0.0),
        "explore_override_frac": float(
            out["moe_aux"].get("explore_override_frac", 0.0)
        ),
        "edge_frac_r0": float(
            (batch["edge_type"] == R0_COVALENT).float().mean().cpu()
        )
        if batch["edge_type"].numel()
        else 0.0,
        "edge_frac_r5": float(
            (batch["edge_type"] == R5_LOCAL_NEIGHBORHOOD).float().mean().cpu()
        )
        if batch["edge_type"].numel()
        else 0.0,
    }
    for i, v_load in enumerate(load_list):
        metrics[f"moe_load_e{i}"] = float(v_load)

    # §2.2 dual-seal: do not leave sat/spread as an untagged post-MoE-only number.
    metrics.update(_dual_seal_diagnostics(diagnostics, out))

    # §2.1: eval-mode argmax load — train moe_load_* is continuity only, not evidence alone.
    eval_metrics = _eval_mode_routing_metrics(
        system,
        batch,
        tau_ceiling=tau_ceil,
        load_floor=float(
            (telemetry or {}).get("eval_load_floor", EVAL_LOAD_FLOOR)
        ),
    )
    metrics.update(eval_metrics)

    # Δ_equiv under the same batch/tau the train step just used.
    metrics.update(
        _log_delta_equiv(system, batch, tau_ceiling=tau_ceil, seed=int(epoch))
    )

    if gate_grad_by_source:
        for name, gnorm in gate_grad_by_source.items():
            metrics[f"gate_grad_{name}"] = float(gnorm)
        q = gate_grad_by_source.get("eval_proxy_quota", 0.0)
        lb = gate_grad_by_source.get("eval_proxy_lb", 0.0)
        task_g = gate_grad_by_source.get("task", 0.0)
        cv_g = gate_grad_by_source.get("cv_STE", 0.0)
        metrics["gate_grad_quota_over_lb"] = (
            float(q / lb) if lb > 0.0 else float("inf")
        )
        metrics["gate_grad_quota_over_task"] = (
            float(q / task_g) if task_g > 0.0 else float("inf")
        )
        metrics["gate_grad_quota_over_cv"] = (
            float(q / cv_g) if cv_g > 0.0 else float("inf")
        )
        print(
            f"[tokyoeye] gate_grad_sources epoch={epoch} "
            f"task={task_g:.2e} lb={lb:.3f} quota={q:.3f} "
            f"cv={cv_g:.3f} quota/lb={metrics['gate_grad_quota_over_lb']:.3f}"
        )

    if biology_grad_metrics:
        metrics.update(biology_grad_metrics)

    tele = telemetry or {}
    health = evaluate_geometry_health(
        metrics,
        oversmooth_entropy_floor=float(tele.get("oversmooth_entropy_floor", 0.20)),
        boundary_saturation_pct_ceiling=float(
            tele.get("boundary_saturation_pct_ceiling", 40.0)
        ),
    )
    metrics.update(health)
    if health["warn_oversmooth"]:
        print(
            f"[tokyoeye] WARN oversmooth: diag_manifold_entropy="
            f"{metrics['diag_manifold_entropy']:.3f} <= "
            f"{tele.get('oversmooth_entropy_floor', 0.20)} — boost mechanism margin"
        )
    if health["warn_boundary_blowout"]:
        print(
            f"[tokyoeye] WARN boundary: diag_boundary_saturation_pct="
            f"{metrics['diag_boundary_saturation_pct']:.1f} > "
            f"{tele.get('boundary_saturation_pct_ceiling', 40.0)} — hold grad clip=1.0"
        )
    return metrics


@torch.no_grad()
def _eval_mode_routing_metrics_pooled(
    system: TokyoEyeV8WithFrontend,
    batches: list[dict[str, Any]],
    *,
    tau_ceiling: float,
    load_floor: float = EVAL_LOAD_FLOOR,
    entropy_norm_floor: float = EVAL_ENTROPY_NORM_FLOOR,
    n_alive_required: int = EVAL_N_ALIVE_REQUIRED,
    alive_floor: float = EVAL_ALIVE_FLOOR,
) -> dict[str, Any]:
    """Node-weighted pooled eval routing over many structures (MoE/ε card R)."""
    was_training = system.training
    system.eval()
    routings: list[torch.Tensor] = []
    for batch in batches:
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau_ceiling,
            chem=batch.get("gate_chem"),
        )
        routings.append(out["moe_aux"]["routing"])
    if was_training:
        system.train()
    routing = torch.cat(routings, dim=0)
    util = eval_moe_utilization(
        routing,
        min_load=float(load_floor),
        alive_floor=float(alive_floor),
        n_alive_required=int(n_alive_required),
        entropy_norm_floor=float(entropy_norm_floor),
    )
    load_list = list(util["load"])
    gates = util["gates"]
    ent_norm = float(gates["moe_eval_entropy_norm"]["value"])
    metrics: dict[str, Any] = {
        "eval_moe_metric_schema": EVAL_MOE_METRIC_SCHEMA,
        "eval_moe_load_min": float(gates["moe_eval_min_load"]["value"]),
        "eval_moe_n_alive": float(gates["moe_eval_n_alive"]["value"]),
        "eval_moe_entropy_norm": ent_norm,
        "eval_moe_load_floor": float(load_floor),
        "eval_moe_entropy_norm_floor": float(entropy_norm_floor),
        "eval_moe_n_alive_required": float(n_alive_required),
        "eval_moe_load_floor_pass": (
            1.0 if gates["moe_eval_min_load"]["pass"] else 0.0
        ),
        "eval_moe_entropy_norm_pass": (
            1.0 if gates["moe_eval_entropy_norm"]["pass"] else 0.0
        ),
        "eval_moe_n_alive_pass": (
            1.0 if gates["moe_eval_n_alive"]["pass"] else 0.0
        ),
        "eval_moe_liveness_pass": 1.0 if util["all_pass"] else 0.0,
        "eval_moe_pooled_nodes": float(routing.shape[0]),
        "eval_moe_pooled_structures": float(len(batches)),
    }
    for i, v_load in enumerate(load_list):
        metrics[f"eval_moe_load_e{i}"] = float(v_load)
    return metrics


def run_epoch_mean_structures(
    system: TokyoEyeV8WithFrontend,
    optimizer: torch.optim.Optimizer,
    batches: list[dict[str, Any]],
    *,
    epoch: int,
    radius: CurriculumRadiusController,
    gumbel: GumbelTemperatureSchedule,
    diagnostics: PoincareDiagnosticsEngine,
    cv_coeff: float,
    moe_quota_coeff: float = 5.0,
    eval_proxy_quota_coeff: float = 140.0,
    explore_epsilon: float = 0.0,
    eps_at_floor: bool = False,
    max_grad_norm: float = 1.0,
    telemetry: dict[str, Any] | None = None,
    margin_coeff: float = 1.0,
    sdrp_coeff: float = SDRP_LOSS_COEFF,
    dehydron_coeff: float = 1.0,
) -> dict[str, float]:
    """One optimizer step = mean loss/grad over ``batches`` (MoE/ε card R).

    Prefer mean (not sum) so loss/grad scale stays comparable to single-structure
    ``run_epoch``. Eval MoE utilization is pooled node-weighted over all batches.
    """
    if not batches:
        raise ValueError("batches must be non-empty")
    n = len(batches)
    system.train()
    tau_ceil = radius.tau_ceiling(epoch)
    gumbel_tau = gumbel.temperature(epoch)
    system.set_moe_temperature(gumbel_tau)
    system.set_moe_explore_epsilon(float(explore_epsilon))
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    last_out: dict[str, Any] | None = None
    last_batch = batches[-1]
    for batch in batches:
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau_ceil,
            chem=batch.get("gate_chem"),
        )
        last_out = out
        loss_sdrp = sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
        loss_margin = mechanism_margin_loss_v2(
            out["mechanism_score"],
            batch["mechanism_pos"],
            batch["mechanism_neg"],
        )
        loss_cv_raw = out["moe_aux"]["cv_loss"]
        loss_quota_raw = out["moe_aux"].get("quota_loss")
        if loss_quota_raw is None:
            loss_quota_raw = torch.zeros((), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype)
        loss_majority_raw = out["moe_aux"].get("majority_hinge_loss")
        if loss_majority_raw is None:
            loss_majority_raw = torch.zeros((), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype)
        loss_switch_raw = out["moe_aux"].get("switch_lb_loss")
        if loss_switch_raw is None:
            loss_switch_raw = torch.zeros((), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype)
        loss_soft_quota_raw = out["moe_aux"].get("soft_quota_loss")
        if loss_soft_quota_raw is None:
            loss_soft_quota_raw = torch.zeros((), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype)
        loss_eval_proxy_raw = out["moe_aux"].get("eval_proxy_lb_loss")
        if loss_eval_proxy_raw is None:
            loss_eval_proxy_raw = torch.zeros((), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype)
        loss_eval_proxy_quota_raw = out["moe_aux"].get("eval_proxy_quota_loss")
        if loss_eval_proxy_quota_raw is None:
            loss_eval_proxy_quota_raw = torch.zeros(
                (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
            )
        loss_balance = (
            loss_cv_raw * float(cv_coeff)
            + loss_quota_raw * float(moe_quota_coeff)
            + loss_majority_raw
            + loss_switch_raw
            + loss_soft_quota_raw * float(moe_quota_coeff)
            + loss_eval_proxy_raw
            + loss_eval_proxy_quota_raw * float(eval_proxy_quota_coeff)
        )
        loss_dehydron = torch.nn.functional.binary_cross_entropy_with_logits(
            out["mechanism_score"], batch["dehydron_labels"]
        )
        loss_task = (
            float(dehydron_coeff) * loss_dehydron
            + float(sdrp_coeff) * loss_sdrp
            + float(margin_coeff) * loss_margin
        )
        loss = (loss_task + loss_balance) / float(n)
        if not torch.isfinite(loss):
            return {
                "loss_total": float("nan"),
                "tau_ceiling": float(tau_ceil),
                "gumbel_temperature": float(gumbel_tau),
                "nan_abort": 1.0,
            }
        loss.backward()
        loss_sum += float(loss.detach())

    grad_norm = float(torch.nn.utils.clip_grad_norm_(system.parameters(), max_grad_norm))
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss_total": float(loss_sum),
            "tau_ceiling": float(tau_ceil),
            "gumbel_temperature": float(gumbel_tau),
            "grad_norm": float("nan"),
            "nan_abort": 1.0,
        }
    optimizer.step()

    assert last_out is not None
    load = last_out["moe_aux"].get("load")
    if load is None:
        load = last_out["moe_aux"]["routing"].mean(dim=0)
    load_list = [float(x) for x in load.detach().tolist()]
    metrics: dict[str, float] = {
        "loss_total": float(loss_sum),
        "tau_ceiling": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "grad_norm": float(grad_norm),
        "explore_epsilon": float(explore_epsilon),
        "eps_at_floor": 1.0 if eps_at_floor else 0.0,
        "nan_abort": 0.0,
        "mean_structures": float(n),
        "moe_quota_coeff": float(moe_quota_coeff),
        "eval_proxy_quota_coeff": float(eval_proxy_quota_coeff),
        "num_nodes": float(sum(int(b["x"].shape[0]) for b in batches)),
    }
    for i, v_load in enumerate(load_list):
        metrics[f"moe_load_e{i}"] = float(v_load)
    metrics["moe_load_min"] = float(min(load_list) if load_list else 0.0)
    metrics.update(
        _eval_mode_routing_metrics_pooled(
            system,
            batches,
            tau_ceiling=tau_ceil,
            load_floor=float((telemetry or {}).get("eval_load_floor", EVAL_LOAD_FLOOR)),
        )
    )
    metrics.update(
        _dual_seal_diagnostics(diagnostics, last_out)
    )
    # Dehydron AUPRC on last structure only (continuity; R bar is pooled MoE).
    dehydron_score = torch.sigmoid(last_out["mechanism_score"]).detach()
    metrics["val_dehydron_auprc"] = binary_auprc(
        dehydron_score, last_batch["dehydron_labels"]
    )
    metrics["dehydron_frac"] = float(
        last_batch["dehydron_labels"].float().mean().item()
    )
    return metrics


def _log_r0_r5_graph_recipe(
    *,
    pdb_dir: Path,
    use_graph_cache: bool,
) -> dict[str, Any]:
    """Log R0–R5 fractions at freeze wrap (addendum §2.7). Runtime retune is forbidden."""
    info: dict[str, Any] = {
        "checked": True,
        "retuned": False,
        "wrap_max": get_dehydron_wrap_max(),
        "dehydron_wrap_frozen": get_dehydron_wrap_max(),
    }
    try:
        batch = load_structure_batch(
            "4OBE",
            "A",
            pdb_dir=pdb_dir,
            device="cpu",
            use_graph_cache=use_graph_cache,
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        print(f"[tokyoeye] WARN: 4OBE graph recipe log skipped ({exc})")
        return info
    frac = float(batch.get("dehydron_frac", 0.0))
    meta = batch.get("graph_meta") or {}
    info.update(
        {
            "dehydron_frac": frac,
            "n_r0": meta.get("n_r0"),
            "n_r1": meta.get("n_r1"),
            "n_r2": meta.get("n_r2"),
            "n_r3": meta.get("n_r3"),
            "n_r4": meta.get("n_r4"),
            "n_r5": meta.get("n_r5"),
        }
    )
    for key in (
        "edge_frac_r0",
        "edge_frac_r1",
        "edge_frac_r2",
        "edge_frac_r3",
        "edge_frac_r4",
        "edge_frac_r5",
    ):
        if key in meta:
            info[key] = meta[key]
    print(
        f"[tokyoeye] 4OBE R0–R5 (wrap={get_dehydron_wrap_max()}): "
        f"dehydron_frac={frac:.3f} "
        + " ".join(
            f"{k}={meta.get(k)}"
            for k in ("n_r0", "n_r1", "n_r2", "n_r3", "n_r4", "n_r5")
        )
    )
    return info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TokyoEye MLflow experiment runner")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/v8/runs"))
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument(
        "--frontend",
        type=str,
        default="equiformer_pool",
        choices=["stub", "equiformer_pool"],
        help=(
            "Frontend path (governed default=equiformer_pool). "
            "stub=StubEquiformerFrontend / SE(3)-lite — requires "
            "--allow-off-path-frontend. "
            "equiformer_pool=EquiformerPoolFrontend (cold random-init §2.3 unless "
            "--equiformer-ckpt is set with --frontend-warm-mptrj)."
        ),
    )
    p.add_argument(
        "--allow-off-path-frontend",
        "--allow-stub-frontend",
        action="store_true",
        dest="allow_off_path_frontend",
        help=(
            "Explicit non-claim diagnostic: allow --frontend stub / SE(3)-lite. "
            "Governed runs must not set this (Architecture SSOT assembly gate). "
            "SE(3)-lite pilots are permitted only when pre-registered, disclaimed, "
            "and only as a cheap signal-existence check before a pool confirmatory run."
        ),
    )
    p.add_argument(
        "--claim-bearing-biology",
        action="store_true",
        help=(
            "Tag this run as claim-bearing biology. Assembly gate then refuses "
            "unless the batch is structurally non-leaking (dehydron/SDRP targets "
            "do not reconstruct from edge_type) and --log-biology-grad-sources N>0. "
            "Incompatible with --allow-off-path-frontend."
        ),
    )
    p.add_argument(
        "--biology-non-leaking-target",
        action="store_true",
        help=(
            "Advisory declaration only (logged on load_info). Does NOT waive the "
            "structural edge_type↔target leakage check — a still-leaking batch fails "
            "even with this flag set."
        ),
    )
    p.add_argument(
        "--frontend-warm-mptrj",
        action="store_true",
        help=(
            "With --frontend=equiformer_pool, load MPtrj weights from "
            "--equiformer-ckpt (opt-in; default remains cold random-init)."
        ),
    )
    p.add_argument("--equiformer-ckpt", type=Path, default=None)
    p.add_argument("--mlflow-uri", type=str, default="")
    p.add_argument("--mlflow-experiment", type=str, default="")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1-epoch synthetic smoke")
    p.add_argument("--pdb", type=str, default=DEFAULT_PDB_ID, help="Mode A PDB id")
    p.add_argument("--chain", type=str, default=DEFAULT_CHAIN, help="Mode A chain")
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Mode B/C Stage A manifest (proteins[] + enabled)",
    )
    p.add_argument(
        "--export-viewers",
        action="store_true",
        help="After training, write Poincaré HTML for last structure",
    )
    p.add_argument("--sdrp-coeff", type=float, default=SDRP_LOSS_COEFF)
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Use stub trunks only (no SE(3)-lite / bank grads)",
    )
    p.add_argument(
        "--no-graph-cache",
        action="store_true",
        help="Bypass pdb_cache/v8_graph_cache and rebuild R0–R5 graphs",
    )
    p.add_argument(
        "--gumbel-schedule",
        type=str,
        default="",
        choices=["", "exponential", "linear"],
        help="Gumbel cool-down (default: config / exponential)",
    )
    p.add_argument(
        "--moe-mode",
        type=str,
        default="live",
        choices=["live", "ablated"],
        help=(
            "MoE routing mode: live=hard Gumbel/argmax (default); "
            "ablated=fixed expert-0 one-hot, no MoE claim "
            "(moe_claim=none; decay_floor_hold=N/A_ABLATED)"
        ),
    )
    p.add_argument(
        "--cv-coeff",
        type=float,
        default=None,
        help="Override MoE CV load-balance coefficient",
    )
    p.add_argument(
        "--moe-quota-coeff",
        type=float,
        default=None,
        help="Override STE/soft min-load quota coefficient",
    )
    p.add_argument(
        "--eval-proxy-quota-coeff",
        type=float,
        default=None,
        help=(
            "Override eval_proxy min-load quota coefficient "
            "(default: weight map / 140 — equalizes vs Switch-LB)"
        ),
    )
    p.add_argument(
        "--log-gate-grad-sources",
        type=int,
        default=3,
        metavar="N",
        help=(
            "On the first N epochs, log gate_fc1+fc2 param-grad norms "
            "broken out by loss source (0 disables)"
        ),
    )
    p.add_argument(
        "--log-biology-grad-sources",
        type=int,
        default=0,
        metavar="N",
        help=(
            "On the first N epochs, log biology-loss grad norms into z_hyp vs "
            "h_euc (Architecture SSOT; required before claim-bearing biology). "
            "0 disables"
        ),
    )
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Continue from a prior TokyoEye system state_dict (Mode C best)",
    )
    p.add_argument(
        "--join-active-mlflow-run",
        action="store_true",
        help="Log into the currently active MLflow run (do not start/end a run)",
    )
    p.add_argument(
        "--taxonomy-domain",
        type=str,
        default="",
        help="Governance domain (geometric|biologic|chemical); sets taxonomy experiment",
    )
    p.add_argument(
        "--taxonomy-subsystem",
        type=str,
        default="",
        help="Governance subsystem under equiformer-v3-moe lineage",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
    _set_seed(args.seed)
    cfg = load_weight_map(args.weight_map)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[tokyoeye] WARN: CUDA requested but unavailable — falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[tokyoeye] device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    run_name = args.run_name or (
        f"tokyoeye_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    freeze = bool(args.freeze_backbone) or bool(cfg.get("freeze_backbone", False))
    frontend_kind = str(args.frontend).strip().lower()
    if frontend_kind == "equiformer_pool":
        # §2.3: cold random-init is the default; warm MPtrj is opt-in only.
        cold = not bool(args.frontend_warm_mptrj)
        if args.frontend_warm_mptrj and args.equiformer_ckpt is None:
            raise SystemExit(
                "--frontend-warm-mptrj requires --equiformer-ckpt"
            )
        system, load_info = build_equiformer_pool_system(
            cfg,
            equiformer_ckpt=args.equiformer_ckpt,
            device=device,
            freeze_backbone=freeze,
            cold_init=cold,
        )
    else:
        system, load_info = build_system(
            cfg,
            equiformer_ckpt=args.equiformer_ckpt,
            device=device,
            freeze_backbone=freeze,
        )
    load_info["frontend_cli"] = frontend_kind
    if args.init_ckpt is not None and Path(args.init_ckpt).is_file():
        blob = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = system.load_state_dict(state, strict=False)
        load_info["init_ckpt"] = str(args.init_ckpt)
        load_info["init_missing"] = len(missing)
        load_info["init_unexpected"] = len(unexpected)
        print(
            f"[tokyoeye] init_ckpt={args.init_ckpt} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    moe_mode = str(getattr(args, "moe_mode", "live") or "live").strip().lower()
    system.set_moe_mode(moe_mode)
    load_info["moe_mode"] = moe_mode
    print(f"[tokyoeye] moe_mode={moe_mode}")

    # Architecture SSOT assembly gate (ENFORCED): pool frontend + deps + pure_hyp
    # (+ claim-bearing third leg when tagged).
    allow_off = bool(getattr(args, "allow_off_path_frontend", False))
    claim_bio = bool(getattr(args, "claim_bearing_biology", False))
    non_leak_flag = bool(getattr(args, "biology_non_leaking_target", False))
    log_bio_n = int(getattr(args, "log_biology_grad_sources", 0) or 0)
    load_info["allow_off_path_frontend"] = allow_off
    load_info["claim_bearing_biology"] = claim_bio
    load_info["biology_non_leaking_target_flag"] = non_leak_flag
    try:
        probe = _make_batch(device)

        def _assembly_forward():
            return system(
                probe["x"],
                probe["edge_index"],
                probe["edge_type"],
                tau_ceiling=0.70,
                chem=probe.get("gate_chem"),
            )

        gate_result = assert_governed_assembly(
            frontend_kind=frontend_kind,
            load_info=load_info,
            system=system,
            forward_fn=_assembly_forward,
            allow_off_path_frontend=allow_off,
            require_pure_hyp=True,
            check_deps=True,
            claim_bearing_biology=claim_bio,
            # Structural check against the probe batch — CLI non-leak flag
            # cannot waive a still-leaking batch.
            claim_batch=probe if claim_bio else None,
            biology_grad_sources_logged=(log_bio_n > 0),
        )
        load_info["assembly_gate"] = gate_result.detail
        load_info["assembly_gate_passed"] = True
        load_info["pure_hyp_checked"] = gate_result.pure_hyp_checked
        load_info["pure_hyp_ok"] = gate_result.pure_hyp_ok
        print(
            f"[tokyoeye] assembly_gate PASS frontend={frontend_kind} "
            f"pure_hyp_checked={gate_result.pure_hyp_checked} "
            f"pure_hyp_ok={gate_result.pure_hyp_ok}"
        )
    except AssemblyGateError as exc:
        load_info["assembly_gate_passed"] = False
        raise SystemExit(f"[tokyoeye] assembly_gate FAIL: {exc}") from exc

    groups = build_param_groups(
        system.frontend,
        system.spine,
        lr_backbone=float(cfg["lr_backbone"]),
        lr_hyperbolic=float(cfg["lr_hyperbolic"]),
        freeze_backbone=freeze,
    )
    print(f"[tokyoeye] backbone_mode={load_info.get('backbone_mode')}")
    optimizer = torch.optim.Adam(groups)

    radius = CurriculumRadiusController(
        float(cfg["tau_start"]), float(cfg["tau_end"]), args.epochs
    )
    gumbel_schedule = (
        args.gumbel_schedule
        or str(cfg.get("gumbel_schedule", "exponential"))
    ).strip().lower()
    gumbel_alpha = cfg.get("gumbel_exp_alpha")
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        args.epochs,
        schedule=gumbel_schedule,
        alpha=float(gumbel_alpha) if gumbel_alpha is not None else None,
        half_epochs=int(cfg.get("gumbel_exp_half_epochs", 12)),
    )
    print(
        f"[tokyoeye] gumbel_schedule={gumbel.schedule} alpha={gumbel.alpha:.4f} "
        f"tau0={gumbel.temperature(0):.3f} tau12={gumbel.temperature(12):.3f}"
    )
    half_ep = int(cfg.get("gumbel_exp_half_epochs", 12))
    eps_sched = EpsilonGreedySchedule(
        float(cfg.get("eps_start", 0.20)),
        float(cfg.get("eps_end", 0.0)),
        half_epochs=int(cfg.get("eps_half_epochs", half_ep)),
    )
    print(
        f"[tokyoeye] eps_schedule start={eps_sched.eps_start:.3f} "
        f"end={eps_sched.eps_end:.3f} half={eps_sched.half_epochs} "
        f"eps0={eps_sched.epsilon(0):.4f} eps12={eps_sched.epsilon(12):.4f}"
    )
    cv_coeff = float(
        args.cv_coeff if args.cv_coeff is not None else cfg.get("cv_coeff", 10.0)
    )
    moe_quota_coeff = float(
        args.moe_quota_coeff
        if args.moe_quota_coeff is not None
        else cfg.get("moe_quota_coeff", 5.0)
    )
    eval_proxy_quota_coeff = float(
        args.eval_proxy_quota_coeff
        if args.eval_proxy_quota_coeff is not None
        else cfg.get("eval_proxy_quota_coeff", 140.0)
    )
    log_gate_grad_n = int(args.log_gate_grad_sources)
    log_biology_grad_n = int(args.log_biology_grad_sources)
    print(
        f"[tokyoeye] cv_coeff={cv_coeff} moe_quota_coeff={moe_quota_coeff} "
        f"eval_proxy_quota_coeff={eval_proxy_quota_coeff}"
    )
    diagnostics = PoincareDiagnosticsEngine()

    dataset: TokyoEyeCuratedDataset | None = None
    if args.smoke:
        data_mode = "synthetic"
        print("[tokyoeye] data_mode=synthetic (--smoke)")
    else:
        use_cache = not bool(args.no_graph_cache)
        retune_info = _log_r0_r5_graph_recipe(
            pdb_dir=args.pdb_dir,
            use_graph_cache=use_cache,
        )
        dataset = TokyoEyeCuratedDataset(
            pdb_code=args.pdb,
            chain=args.chain,
            manifest_path=args.manifest,
            pdb_dir=args.pdb_dir,
            use_graph_cache=use_cache,
        )
        data_mode = dataset.mode
        print(
            f"[tokyoeye] data_mode={data_mode} n_structures={len(dataset)} "
            f"pdb_dir={args.pdb_dir} graph_cache={use_cache} "
            f"wrap_max={get_dehydron_wrap_max()}"
        )
        if dataset.manifest_path:
            print(f"[tokyoeye] manifest={dataset.manifest_path}")
        if retune_info.get("error"):
            print(f"[tokyoeye] graph recipe warn: {retune_info['error']}")

    use_mlflow = not args.no_mlflow
    mlflow = None
    joined_active = False
    if use_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        uri = args.mlflow_uri or "http://mlflow:5000"
        if not _mlflow_uri_reachable(uri):
            print(
                f"[tokyoeye] MLflow unreachable at {uri} (connect timeout); "
                "continuing without tracking — start the server before the next "
                "governed run so addendum_id/pure_hyp_pass_version land on full-stack"
            )
            use_mlflow = False
            mlflow = None
    if use_mlflow:
        uri = args.mlflow_uri or "http://mlflow:5000"
        try:
            mlflow.set_tracking_uri(uri)
            exp = resolve_v8_mlflow_experiment(
                cli_experiment=args.mlflow_experiment,
                cfg=cfg,
                taxonomy_domain=args.taxonomy_domain,
                taxonomy_subsystem=args.taxonomy_subsystem,
            )
            if args.taxonomy_domain and args.taxonomy_subsystem:
                from science.tokyo_eye.governance.train_pipeline import (
                    ensure_taxonomy_experiment,
                )

                ensure_taxonomy_experiment(
                    args.taxonomy_domain,
                    args.taxonomy_subsystem,
                    tracking_uri=uri,
                )
            mlflow.set_experiment(exp)
            rec = freeze_reconciliation_mlflow_params()
            rec["mlflow_experiment"] = str(exp)
            if args.join_active_mlflow_run and mlflow.active_run() is not None:
                joined_active = True
            else:
                mlflow.start_run(run_name=run_name)
            mlflow.set_tags(
                {
                    "addendum_id": rec["addendum_id"],
                    "pure_hyp_pass_version": rec["pure_hyp_pass_version"],
                    "moe_mode": moe_mode,
                    "moe_claim": "none" if moe_mode == "ablated" else "live",
                    **(
                        {
                            "diagnostic": "true",
                            "do_not_promote": "true",
                        }
                        if moe_mode == "ablated"
                        else {}
                    ),
                }
            )
            mlflow.log_params(
                {
                    "addendum_id": rec["addendum_id"],
                    "pure_hyp_pass_version": rec["pure_hyp_pass_version"],
                    "mlflow_experiment": rec["mlflow_experiment"],
                    "curvature_c": cfg.get("curvature_c", 1.0),
                    "tau_start": cfg["tau_start"],
                    "tau_end": cfg["tau_end"],
                    "tau_max": cfg["tau_end"],
                    "gumbel_tau_start": cfg["gumbel_tau_start"],
                    "gumbel_tau_end": cfg["gumbel_tau_end"],
                    "gumbel_schedule": gumbel.schedule,
                    "gumbel_exp_alpha": gumbel.alpha,
                    "cv_coeff": cv_coeff,
                    "moe_quota_coeff": moe_quota_coeff,
                    "eval_proxy_quota_coeff": eval_proxy_quota_coeff,
                    "moe_quota_floor": cfg.get("moe_quota_floor", 0.05),
                    "moe_mode": moe_mode,
                    "sdrp_coeff": args.sdrp_coeff,
                    "lr_backbone": cfg["lr_backbone"],
                    "lr_hyperbolic": cfg["lr_hyperbolic"],
                    "scalar_dim": cfg["scalar_dim"],
                    "vector_dim": cfg["vector_dim"],
                    "hidden_dim": cfg["hidden_dim"],
                    "equiformer_mode": load_info.get("mode"),
                    "backbone_mode": load_info.get("backbone_mode"),
                    "live_backbone": load_info.get("live_backbone"),
                    "frontend_cli": load_info.get("frontend_cli", "stub"),
                    "eval_moe_metric_schema": EVAL_MOE_METRIC_SCHEMA,
                    "data_mode": data_mode,
                    "pdb": args.pdb if not args.smoke else "synthetic",
                    "chain": args.chain if not args.smoke else "-",
                    "seed": args.seed,
                    "epochs": args.epochs,
                }
            )
            mlflow.log_dict(load_info, "equiformer_load_info.json")
        except Exception as exc:  # noqa: BLE001 — allow local smoke without MLflow
            print(f"[tokyoeye] MLflow unavailable ({exc}); continuing without tracking")
            use_mlflow = False
            if mlflow is not None and not joined_active:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
            mlflow = None

    history: list[dict[str, Any]] = []
    best_path = out_dir / "tokyoeye_best.pt"
    last_path = out_dir / "tokyoeye_last.pt"
    best_loss = float("inf")
    margin_coeff = 1.0
    telemetry = dict(cfg.get("telemetry") or {})
    boost = float(telemetry.get("margin_boost_on_oversmooth", 1.5))
    last_batch: dict[str, Any] | None = None

    for epoch in range(args.epochs):
        # One graph per step (Mode A fixed; Mode B round-robin).
        if args.smoke:
            batch = _make_batch(device)
        else:
            assert dataset is not None
            batch = dataset.get_on_device(epoch % len(dataset), device)
        last_batch = batch
        eps_t = eps_sched.epsilon(epoch)
        eps_at_floor = bool(eps_t <= eps_sched.eps_end + 1e-12)
        metrics = run_epoch(
            system,
            optimizer,
            batch,
            epoch=epoch,
            radius=radius,
            gumbel=gumbel,
            diagnostics=diagnostics,
            cv_coeff=cv_coeff,
            moe_quota_coeff=moe_quota_coeff,
            eval_proxy_quota_coeff=eval_proxy_quota_coeff,
            explore_epsilon=eps_t,
            eps_at_floor=eps_at_floor,
            log_gate_grad_sources=(log_gate_grad_n > 0 and epoch < log_gate_grad_n),
            log_biology_grad_sources=(
                log_biology_grad_n > 0 and epoch < log_biology_grad_n
            ),
            telemetry=telemetry,
            margin_coeff=margin_coeff,
            sdrp_coeff=float(args.sdrp_coeff),
        )
        if eps_at_floor:
            # §2.6 decay-floor hold: surface eval liveness once training wheels are off.
            print(
                f"[tokyoeye] eps_floor epoch={epoch} eps={eps_t:.4f} "
                f"eval_n_alive={metrics.get('eval_moe_n_alive', float('nan'))} "
                f"eval_load_min={metrics.get('eval_moe_load_min', float('nan')):.3f} "
                f"eval_entropy_norm={metrics.get('eval_moe_entropy_norm', float('nan')):.3f}"
            )
        metrics["pdb_id_step"] = 0.0  # placeholder for numeric loggers
        if "pdb_id" in batch:
            metrics["structure_tag"] = 1.0
            meta = batch.get("graph_meta") or {}
            print(
                f"[tokyoeye] structure={batch['pdb_id']}:{batch.get('chain', '?')} "
                f"N={int(metrics['num_nodes'])} dehydron_frac={metrics['dehydron_frac']:.3f} "
                f"n_r1={meta.get('n_r1')} n_r2={meta.get('n_r2')} "
                f"moe_load=[{metrics.get('moe_load_e0', 0):.3f},"
                f"{metrics.get('moe_load_e1', 0):.3f},"
                f"{metrics.get('moe_load_e2', 0):.3f},"
                f"{metrics.get('moe_load_e3', 0):.3f}] "
                f"eval_moe_load=[{metrics.get('eval_moe_load_e0', 0):.3f},"
                f"{metrics.get('eval_moe_load_e1', 0):.3f},"
                f"{metrics.get('eval_moe_load_e2', 0):.3f},"
                f"{metrics.get('eval_moe_load_e3', 0):.3f}] "
                f"eval_entropy_norm={metrics.get('eval_moe_entropy_norm', float('nan')):.3f}"
            )
        if metrics.get("warn_oversmooth", 0.0) >= 1.0:
            margin_coeff = max(margin_coeff, boost)
        history.append(metrics)
        print(
            f"[tokyoeye] epoch={epoch} loss={metrics['loss_total']:.4f} "
            f"tau={metrics['tau_ceiling']:.3f} gumbel={metrics['gumbel_temperature']:.3f} "
            f"r_post={metrics.get('diag_post_moe_mean_radius', metrics.get('diag_mean_radius', float('nan'))):.3f} "
            f"r_lift={metrics.get('diag_pre_moe_lift_mean_radius', float('nan')):.3f} "
            f"spread_post={metrics.get('diag_post_moe_radius_spread', float('nan')):.4f} "
            f"spread_lift={metrics.get('diag_pre_moe_lift_radius_spread', float('nan')):.4f} "
            f"eval_load_min={metrics.get('eval_moe_load_min', float('nan')):.3f} "
            f"Δ_equiv={metrics.get('delta_equiv_lift_spine', float('nan')):.2e} "
            f"auprc={metrics.get('val_dehydron_auprc', float('nan')):.3f}"
        )
        if metrics.get("nan_abort", 0.0) >= 1.0:
            print("[tokyoeye] aborting run due to non-finite loss/grads")
            if use_mlflow and mlflow is not None:
                mlflow.log_metric("nan_abort", 1.0, step=epoch)
                mlflow.end_run(status="FAILED")
            raise SystemExit(2)
        if use_mlflow and mlflow is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, float(v), step=epoch)

        payload = {
            "epoch": epoch,
            "model": system.state_dict(),
            "cfg": cfg,
            "load_info": load_info,
            "metrics": metrics,
        }
        torch.save(payload, last_path)
        if metrics["loss_total"] < best_loss:
            best_loss = metrics["loss_total"]
            torch.save(payload, best_path)

    summary = {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "best_loss": best_loss,
        "best_checkpoint": str(best_path),
        "load_info": load_info,
        "data_mode": data_mode,
        "history": history,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if use_mlflow and mlflow is not None:
        mlflow.log_artifact(str(out_dir / "run_summary.json"))
        mlflow.log_artifact(str(best_path))
        if not joined_active:
            mlflow.end_run()

    if args.export_viewers and last_batch is not None and not args.smoke:
        from experiments.training.v8.export_viewers import export_v8_structure_viewers

        system.eval()
        with torch.no_grad():
            out = system(
                last_batch["x"],
                last_batch["edge_index"],
                last_batch["edge_type"],
                tau_ceiling=float(cfg["tau_end"]),
                chem=last_batch.get("gate_chem"),
            )
        paths = export_v8_structure_viewers(
            pdb_id=str(last_batch.get("pdb_id", args.pdb)),
            z_hyp=out["z_hyp"].detach().cpu(),
            dehydron_labels=last_batch["dehydron_labels"].detach().cpu(),
            mechanism_score=out["mechanism_score"].detach().cpu(),
            evidence=out["evidence"].detach().cpu(),
            checkpoint_path=str(best_path),
            curvature=float(cfg.get("curvature_c", 1.0)),
            z_attn=out["z_attn"].detach().cpu(),
            z_lift=out["z_lift"].detach().cpu(),
            h_euc=out["h_euc"].detach().cpu(),
            expert_id=out["moe_aux"]["routing"].argmax(dim=-1).detach().cpu(),
        )
        print(json.dumps({"exported_viewers": paths}, indent=2))

    print(json.dumps({"ok": True, "best_checkpoint": str(best_path)}, indent=2))


if __name__ == "__main__":
    main()
