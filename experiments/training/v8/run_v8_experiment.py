#!/usr/bin/env python3
"""TokyoEye MLflow experiment harness (Sprint 5).

Isolated under ``experiments/training/v8/``. Does not load v7/v66 checkpoints.

Default Equiformer path (convention):
  ``checkpoints/v8/pretrained/equiformer_v3_baseline.pt``
Weight map:
  ``science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json``

If the Equiformer checkpoint is missing, runs with ``StubEquiformerFrontend``
at matching ``scalar_dim`` / ``vector_dim`` (smoke / unit path).
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
    get_dehydron_wrap_max,
    set_dehydron_wrap_max,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord

MLFLOW_EXPERIMENT_DEFAULT = "tokyoeye/equiformer-v3-moe/geometric/full-stack"
SDRP_LOSS_COEFF = 0.1


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
    return {
        "x": x,
        "edge_index": ei,
        "edge_type": et,
        "sdrp_target": torch.randint(0, 5, (n,), device=device),
        "mechanism_pos": torch.rand(n, device=device) * 0.5 + 0.5,
        "mechanism_neg": torch.rand(n, device=device) * 0.4,
        "dehydron_labels": labels,
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
    if ckpt is None:
        default = Path(cfg.get("checkpoint_path_default", DEFAULT_CKPT))
        if default.is_file():
            ckpt = default
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
    max_grad_norm: float = 1.0,
    telemetry: dict[str, Any] | None = None,
    margin_coeff: float = 1.0,
    sdrp_coeff: float = SDRP_LOSS_COEFF,
) -> dict[str, float]:
    system.train()
    tau_ceil = radius.tau_ceiling(epoch)
    gumbel_tau = gumbel.temperature(epoch)
    system.set_moe_temperature(gumbel_tau)

    out = system(
        batch["x"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceil,
    )
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
    loss_balance = (
        loss_cv_raw * float(cv_coeff)
        + loss_quota_raw * float(moe_quota_coeff)
    )
    # Primary biophysical gate: binary dehydron incidence via mechanism score logits.
    loss_dehydron = torch.nn.functional.binary_cross_entropy_with_logits(
        out["mechanism_score"], batch["dehydron_labels"]
    )
    loss = (
        loss_dehydron
        + float(sdrp_coeff) * loss_sdrp
        + float(margin_coeff) * loss_margin
        + loss_balance
    )

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
        "loss_sdrp": float(loss_sdrp.detach()),
        "loss_margin": float(loss_margin.detach()),
        "loss_cv": float(loss_balance.detach()),
        "moe_quota_loss": float(loss_quota_raw.detach()),
        "sdrp_coeff": float(sdrp_coeff),
        "tau_ceiling": float(tau_ceil),
        "scheduled_tau": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "grad_norm": grad_norm,
        "max_grad_norm": float(max_grad_norm),
        "margin_coeff": float(margin_coeff),
        "val_dehydron_auprc": float(auprc),
        "epistemic_risk_mean": float(risk.mean()),
        "dehydron_frac": float(batch.get("dehydron_frac", batch["dehydron_labels"].mean())),
        "num_nodes": float(batch.get("num_nodes", batch["x"].shape[0])),
        "moe_load_min": float(min(load_list) if load_list else 0.0),
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
    diag = diagnostics.summarize(out["z_hyp"])
    for k, val in diag.items():
        metrics[f"diag_{k}"] = float(val) if not isinstance(val, bool) else float(val)
    metrics["diag_boundary_saturation_pct"] = metrics["diag_boundary_saturation"] * 100.0
    metrics["diag_manifold_entropy"] = metrics["diag_radial_entropy"]

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


def _maybe_retune_wrap_max_from_4obe(
    *,
    pdb_dir: Path,
    use_graph_cache: bool,
    frac_ceiling: float = 0.60,
) -> dict[str, Any]:
    """Sprint 8: if 4OBE dehydron_frac ≥ ceiling, retune τ from wrap histogram.

    Starts at the empirical median, then descends until ``dehydron_frac < ceiling``
    or ``τ = 0`` (cone wrap compresses counts so median alone can remain too high).
    """
    info: dict[str, Any] = {
        "checked": True,
        "retuned": False,
        "wrap_max_before": get_dehydron_wrap_max(),
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
        print(f"[tokyoeye] WARN: 4OBE wrap retune skipped ({exc})")
        return info
    frac = float(batch.get("dehydron_frac", 0.0))
    meta = batch.get("graph_meta") or {}
    wraps = list(meta.get("hbond_wrap_counts") or [])
    info.update(
        {
            "dehydron_frac": frac,
            "n_r1": meta.get("n_r1"),
            "n_r2": meta.get("n_r2"),
            "n_wrap_samples": len(wraps),
        }
    )
    print(
        f"[tokyoeye] 4OBE biophys gate: dehydron_frac={frac:.3f} "
        f"n_r1={meta.get('n_r1')} n_r2={meta.get('n_r2')} "
        f"wrap_max={get_dehydron_wrap_max()}"
    )
    if frac < float(frac_ceiling):
        return info
    if not wraps:
        print("[tokyoeye] WARN: dehydron_frac high but no wrap histogram; leaving τ unchanged")
        return info
    arr = np.asarray(wraps, dtype=np.float64)
    hist, edges = np.histogram(arr, bins=min(20, max(5, len(np.unique(arr)))))
    print("[tokyoeye] 4OBE wrap histogram (frac≥0.60 → median-then-descend retune):")
    for i, c in enumerate(hist.tolist()):
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}): {c}")

    tau = int(np.median(arr))
    chosen_frac = frac
    while tau >= 0:
        set_dehydron_wrap_max(tau)
        batch = load_structure_batch(
            "4OBE",
            "A",
            pdb_dir=pdb_dir,
            device="cpu",
            use_graph_cache=False,  # force rebuild at new τ
        )
        chosen_frac = float(batch.get("dehydron_frac", 0.0))
        print(
            f"[tokyoeye] retune try τ={tau} → dehydron_frac={chosen_frac:.3f} "
            f"n_r1={batch.get('graph_meta', {}).get('n_r1')} "
            f"n_r2={batch.get('graph_meta', {}).get('n_r2')}"
        )
        if chosen_frac < float(frac_ceiling) or tau == 0:
            break
        tau -= 1

    print(
        f"[tokyoeye] retuning DEHYDRON_WRAP_MAX {info['wrap_max_before']} → {get_dehydron_wrap_max()}"
    )
    info["retuned"] = True
    info["wrap_max_after"] = get_dehydron_wrap_max()
    info["wrap_median"] = int(np.median(arr))
    info["dehydron_frac_after"] = chosen_frac
    return info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TokyoEye MLflow experiment runner")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/tokyoeye/runs"))
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
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
        "--cv-coeff",
        type=float,
        default=None,
        help="Override MoE CV load-balance coefficient",
    )
    p.add_argument(
        "--moe-quota-coeff",
        type=float,
        default=None,
        help="Override min-load quota coefficient",
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
    system, load_info = build_system(
        cfg,
        equiformer_ckpt=args.equiformer_ckpt,
        device=device,
        freeze_backbone=freeze,
    )
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
    cv_coeff = float(
        args.cv_coeff if args.cv_coeff is not None else cfg.get("cv_coeff", 10.0)
    )
    moe_quota_coeff = float(
        args.moe_quota_coeff
        if args.moe_quota_coeff is not None
        else cfg.get("moe_quota_coeff", 5.0)
    )
    print(f"[tokyoeye] cv_coeff={cv_coeff} moe_quota_coeff={moe_quota_coeff}")
    diagnostics = PoincareDiagnosticsEngine()

    dataset: TokyoEyeCuratedDataset | None = None
    if args.smoke:
        data_mode = "synthetic"
        print("[tokyoeye] data_mode=synthetic (--smoke)")
    else:
        use_cache = not bool(args.no_graph_cache)
        retune_info = _maybe_retune_wrap_max_from_4obe(
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
        if retune_info.get("retuned"):
            print(f"[tokyoeye] wrap retune applied: {retune_info}")

    use_mlflow = not args.no_mlflow
    mlflow = None
    joined_active = False
    if use_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        uri = args.mlflow_uri or "http://mlflow:5000"
        try:
            mlflow.set_tracking_uri(uri)
            if args.taxonomy_domain and args.taxonomy_subsystem:
                from science.tokyo_eye.governance.train_pipeline import (
                    ensure_taxonomy_experiment,
                )

                exp = ensure_taxonomy_experiment(
                    args.taxonomy_domain,
                    args.taxonomy_subsystem,
                    tracking_uri=uri,
                )
            else:
                exp = args.mlflow_experiment or cfg.get(
                    "mlflow_experiment", MLFLOW_EXPERIMENT_DEFAULT
                )
                mlflow.set_experiment(exp)
            if args.join_active_mlflow_run and mlflow.active_run() is not None:
                joined_active = True
            else:
                mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
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
                    "moe_quota_floor": cfg.get("moe_quota_floor", 0.05),
                    "sdrp_coeff": args.sdrp_coeff,
                    "lr_backbone": cfg["lr_backbone"],
                    "lr_hyperbolic": cfg["lr_hyperbolic"],
                    "scalar_dim": cfg["scalar_dim"],
                    "vector_dim": cfg["vector_dim"],
                    "hidden_dim": cfg["hidden_dim"],
                    "equiformer_mode": load_info.get("mode"),
                    "backbone_mode": load_info.get("backbone_mode"),
                    "live_backbone": load_info.get("live_backbone"),
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
            telemetry=telemetry,
            margin_coeff=margin_coeff,
            sdrp_coeff=float(args.sdrp_coeff),
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
                f"{metrics.get('moe_load_e3', 0):.3f}]"
            )
        if metrics.get("warn_oversmooth", 0.0) >= 1.0:
            margin_coeff = max(margin_coeff, boost)
        history.append(metrics)
        print(
            f"[tokyoeye] epoch={epoch} loss={metrics['loss_total']:.4f} "
            f"tau={metrics['tau_ceiling']:.3f} gumbel={metrics['gumbel_temperature']:.3f} "
            f"r_mean={metrics.get('diag_mean_radius', float('nan')):.3f} "
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
