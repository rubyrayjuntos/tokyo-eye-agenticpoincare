"""Tokyo Eye EQU lift_radius — sealed 20-epoch volume + invariant lift train.

Fresh spine under pure-hyp freeze. No correct_start / champion / affinity / C1 θ.
Learnable α lift + L_rad + final_only τ clamp; sealed rim gate formulas.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from experiments.training.v8.c1_topology_curriculum import (
    freeze_entire_frontend,
    spine_param_group,
)
from experiments.training.v8.equ_theme_signal import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    DEFAULT_STAMP,
    DISPLAY_LINEAGE,
    GATE_ID,
    MLFLOW_EXPERIMENT,
    MLFLOW_RUN_NAME,
    PINNED_FRONTEND_SHA256,
    QuadraticRadiusController,
    RV_BOUNDARY_RADIUS as CS_BOUNDARY_RADIUS,
    RV_EPOCHS as CS_EPOCHS,
    RV_LR_BACKBONE as CS_LR_BACKBONE,
    RV_LR_HYP as CS_LR_HYP,
    RV_MIN_TRAIN as CS_MIN_TRAIN,
    RV_NUM_EXPERTS as CS_NUM_EXPERTS,
    RV_PROBE_EVERY as CS_PROBE_EVERY,
    RV_TAU_END as CS_TAU_END,
    RV_TAU_START as CS_TAU_START,
    RV_VOLUME_COEFF,
    RV_VOLUME_LAM,
    RV_VOLUME_LAM_BARRIER,
    RV_VOLUME_LAM_MEAN,
    RV_VOLUME_MU,
    RV_VOLUME_SIGMA,
    RV_TAU_CLAMP_MODE,
    RV_WEIGHT_DECAY as CS_WEIGHT_DECAY,
    RV_WRAP_MAX as CS_WRAP_MAX,
    ThemeBiologyGuardError as CorrectStartGuardError,
    assert_frontend_bank,
    assert_not_forbidden_ckpt,
    assert_pure_hyp_strict,
    build_theme_signal_stamp as build_correct_start_stamp,
    enrich_rim_probe_row,
    epoch_is_best_eligible,
    evaluate_lift_spine_equivariance,
    evaluate_probe_hygiene,
    evaluate_theme_biology,
    theme_dehydron_auprc,
    DEFAULT_SPINE_INIT,
    load_boot_split,
    load_pins,
    normalized_routing_entropy,
)
from experiments.training.v8.run_b0_topology_observation import (
    _ca_records,
    _device,
    _first_ca_complete_chain,
    observe_one,
)
from experiments.training.v8.run_v8_experiment import build_system, run_epoch
from science.dtie.common.curvature_values import require_learned_curvature
from science.tokyo_eye.v8.engine import (
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
)
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.loader import (
    DEFAULT_PDB_DIR,
    ensure_pdb_cached,
    load_structure_batch,
)
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max, set_dehydron_wrap_max

TAG = "[equ-theme-signal]"


def _lock_frontend_eval(system: nn.Module) -> None:
    orig = system.train

    def _train(mode: bool = True):
        orig(mode)
        system.frontend.eval()
        return system

    system.train = _train  # type: ignore[method-assign]


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _resolve_entry(entry: dict[str, Any], pdb_dir: Path) -> dict[str, Any]:
    out = dict(entry)
    flags = dict(entry.get("flags") or {})
    pdb_id = str(out["pdb_id"])
    chain = str(out["chain"])
    pdb_path = ensure_pdb_cached(pdb_id, pdb_dir)
    if flags.get("first_ca_polymer"):
        records = _ca_records(pdb_path, chain, model1=bool(flags.get("nmr_model1")))
        if not records:
            alt = _first_ca_complete_chain(pdb_path)
            if alt and alt != chain:
                out["chain"] = alt
                out["chain_resolved"] = "first_ca_polymer"
    return out


def _load_train_batch(
    entry: dict[str, Any],
    *,
    device: torch.device,
    pdb_dir: Path,
    graph_cache_dir: Path,
) -> dict[str, Any]:
    return load_structure_batch(
        str(entry["pdb_id"]),
        str(entry["chain"]),
        pdb_dir=pdb_dir,
        device=device,
        graph_cache_dir=graph_cache_dir,
    )


def _enrich_h_norm(row: dict[str, Any], out_routing: torch.Tensor | None = None) -> dict[str, Any]:
    """Attach H_norm from MoE load vector when present on observe row."""
    if row.get("h_norm") is not None:
        return row
    loads = [
        float(row[k])
        for k in (f"moe_load_e{i}" for i in range(CS_NUM_EXPERTS))
        if row.get(k) is not None
    ]
    if loads:
        # Reconstruct soft usage from load shares for entropy
        t = torch.tensor(loads, dtype=torch.float32).unsqueeze(0)
        # normalized_routing_entropy expects [N,K]; mean over N of one-hot ≈ load
        # Use load row as a single multinomial over experts
        row["h_norm"] = normalized_routing_entropy(t, num_experts=CS_NUM_EXPERTS)
    elif out_routing is not None:
        row["h_norm"] = normalized_routing_entropy(
            out_routing, num_experts=CS_NUM_EXPERTS
        )
    return row


def _observe_panel(
    entries: list[dict[str, Any]],
    *,
    system: nn.Module,
    curvature: float,
    device: torch.device,
    pdb_dir: Path,
    diagnostics: PoincareDiagnosticsEngine,
    tau_ceil: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        try:
            row = observe_one(
                entry=entry,
                system=system,  # type: ignore[arg-type]
                curvature=curvature,
                tau_ceil=tau_ceil,
                device=device,
                pdb_dir=pdb_dir,
                theta=GATE_ID,
                diagnostics=diagnostics,
            )
            if row.get("loaded"):
                row = _enrich_h_norm(row)
                try:
                    from science.tokyo_eye.v8.loader import load_structure_batch, ensure_pdb_cached
                    entry2 = dict(entry)
                    ensure_pdb_cached(str(entry2["pdb_id"]), pdb_dir)
                    batch = load_structure_batch(
                        str(entry2["pdb_id"]),
                        str(entry2["chain"]),
                        pdb_dir=pdb_dir,
                        device=device,
                        graph_cache_dir=pdb_dir / "v8_graph_cache",
                    )
                    with torch.no_grad():
                        outz = system(
                            batch["x"],
                            batch["edge_index"],
                            batch["edge_type"],
                            tau_ceiling=tau_ceil,
                        )
                    row = enrich_rim_probe_row(
                        row, z_hyp=outz["z_hyp"], c=float(curvature)
                    )

                    bio = theme_dehydron_auprc(
                        system=system, batch=batch, tau_ceiling=tau_ceil
                    )
                    row.update(bio)
                    if entry.get("theme"):
                        row["theme"] = entry.get("theme")
                except Exception as _enr_exc:  # noqa: BLE001
                    row = enrich_rim_probe_row(row)
                    row["rim_enrich_error"] = f"{type(_enr_exc).__name__}: {_enr_exc}"
        except Exception as exc:  # noqa: BLE001
            row = {
                "pdb_id": entry["pdb_id"],
                "chain": entry["chain"],
                "theme": entry.get("theme"),
                "loaded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
    return rows


def _train_viable(
    train: list[dict[str, Any]],
    skipped: set[tuple[str, str]],
    *,
    min_train: int = CS_MIN_TRAIN,
) -> bool:
    remaining = [
        e for e in train if (str(e["pdb_id"]), str(e["chain"])) not in skipped
    ]
    return len(remaining) >= int(min_train)


def _write_stamp(path: Path, stamp: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2) + "\n")
    print(
        f"{TAG} stamp -> {path} status={stamp.get('status')} "
        f"exec={stamp.get('execution_state')}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo Eye EQU lift_radius")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pins", type=Path, default=Path("data/gates/tokyo_eye_equ_theme_signal_pins.json"))
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--frontend-ckpt", type=Path, default=DEFAULT_FRONTEND_CKPT)
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--graph-cache-dir", type=Path, default=Path("data/graph_cache"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/tokyoeye/runs"))
    p.add_argument("--out-stamp", type=Path, default=DEFAULT_STAMP)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--allow-stub-frontend", action="store_true")
    p.add_argument("--spine-init", type=Path, default=DEFAULT_SPINE_INIT,
                    help="QUALIFIED lift_radius best ckpt to warm-start")
    p.add_argument("--fresh-spine", action="store_true",
                    help="Ignore spine-init; random spine (debug only)")
    p.add_argument("--dehydron-coeff", type=float, default=2.5)
    p.add_argument(
        "--pure-hyp-strict",
        action="store_true",
        default=True,
        help="Refuse start unless static+live pure-hyp pass (default on)",
    )
    p.add_argument(
        "--no-pure-hyp-strict",
        action="store_false",
        dest="pure_hyp_strict",
        help="Disable pure-hyp veto (not for sealed card)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pdb_dir = Path(args.pdb_dir)
    graph_cache_dir = Path(args.graph_cache_dir)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    pins = load_pins(args.pins)
    expected_sha = str(pins.get("frontend_bank_sha256") or PINNED_FRONTEND_SHA256)
    frontend_path = Path(args.frontend_ckpt)
    if str(pins.get("frontend_ckpt")) and args.frontend_ckpt == DEFAULT_FRONTEND_CKPT:
        frontend_path = Path(str(pins["frontend_ckpt"]))

    assert_not_forbidden_ckpt(frontend_path)
    if not frontend_path.is_file():
        if args.allow_stub_frontend:
            print(f"{TAG} WARN missing frontend {frontend_path}; stub path")
            digest = "stub"
        else:
            raise FileNotFoundError(f"frontend bank missing: {frontend_path}")
    else:
        digest = assert_frontend_bank(frontend_path, expected=expected_sha)

    prior_stamp: dict[str, Any] = {}
    if args.out_stamp.is_file():
        prior_stamp = json.loads(args.out_stamp.read_text())

    pure_report = assert_pure_hyp_strict(None) if args.pure_hyp_strict else {
        "pure_hyp_pass": True,
        "findings": {},
        "skipped": True,
    }

    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    set_dehydron_wrap_max(CS_WRAP_MAX)
    split = load_boot_split(args.manifest)
    train = [_resolve_entry(e, pdb_dir) for e in split["train"]]
    probe = [_resolve_entry(e, pdb_dir) for e in split["probe"]]
    skipped: set[tuple[str, str]] = set()
    skip_reasons: dict[str, str] = {}

    cfg = dict(load_weight_map(args.weight_map))
    cfg["lr_hyperbolic"] = CS_LR_HYP
    cfg["lr_backbone"] = CS_LR_BACKBONE
    cfg.setdefault("gumbel_tau_start", 1.0)
    cfg.setdefault("gumbel_tau_end", 0.5)
    cfg.setdefault("cv_coeff", 10.0)
    cfg.setdefault("moe_quota_coeff", 5.0)

    system, load_info = build_system(
        cfg,
        equiformer_ckpt=None if digest == "stub" else frontend_path,
        device=device,
        freeze_backbone=True,
    )
    if load_info.get("mode") not in {"weight_map_loaded", "stub_random_init"} and not args.allow_stub_frontend:
        raise RuntimeError(f"unexpected frontend load: {load_info}")
    if load_info.get("mode") == "stub_missing_ckpt" and not args.allow_stub_frontend:
        raise RuntimeError(f"frontend failed to load: {load_info}")

    if args.pure_hyp_strict:
        pure_report = assert_pure_hyp_strict(system)

    curvature = (
        float(system.spine.c.detach().cpu())
        if hasattr(system.spine.c, "detach")
        else float(system.spine.c)
    )
    require_learned_curvature(curvature, context="equ correct_start c")

    # Geometry Pass: freeze entire Equiformer (cold-boot GPU path). Plan backbone
    # LR logged but not applied until affinity/biology Pass unfreezes adapters.
    n_frozen = freeze_entire_frontend(system)
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    if not args.fresh_spine:
        spine_path = Path(args.spine_init)
        if not spine_path.is_file():
            raise FileNotFoundError(f"spine_init missing: {spine_path}")
        assert_not_forbidden_ckpt(spine_path)
        blob_init = torch.load(spine_path, map_location=device, weights_only=False)
        state_init = blob_init["model"] if isinstance(blob_init, dict) and "model" in blob_init else blob_init
        missing_i, unexpected_i = system.load_state_dict(state_init, strict=False)
        print(
            f"{TAG} spine_init={spine_path} missing={len(missing_i)} unexpected={len(unexpected_i)}",
            flush=True,
        )
    else:
        print(f"{TAG} fresh_spine=True (no lift_radius warm start)", flush=True)
    print(f"{TAG} tau_clamp_mode={system.spine.tau_clamp_mode}")
    _lock_frontend_eval(system)
    optimizer = torch.optim.AdamW(
        spine_param_group(system, lr=CS_LR_HYP),
        weight_decay=CS_WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    radius = QuadraticRadiusController(CS_TAU_START, CS_TAU_END, int(args.epochs))
    half = max(1, int(args.epochs) // 2)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        int(args.epochs),
        schedule="exponential",
        alpha=None,
        half_epochs=half,
    )
    diagnostics_train = PoincareDiagnosticsEngine(boundary_radius=CS_BOUNDARY_RADIUS)
    diagnostics_h3 = PoincareDiagnosticsEngine(boundary_radius=CS_BOUNDARY_RADIUS)
    cv_coeff = float(cfg.get("cv_coeff", 10.0))
    moe_quota_coeff = float(cfg.get("moe_quota_coeff", 5.0))
    telemetry = dict(cfg.get("telemetry") or {})

    run_name = f"eqf_equ_theme_signal_{date.today().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "tokyoeye_best.pt"
    last_path = out_dir / "tokyoeye_last.pt"

    print(
        f"{TAG} frontend sha256={digest[:16]}… mode={load_info.get('mode')} "
        f"c={curvature} wrap_max={get_dehydron_wrap_max()} device={device} "
        f"frozen_frontend_tensors={n_frozen}"
    )
    print(
        f"{TAG} spine AdamW lr={CS_LR_HYP} wd={CS_WEIGHT_DECAY} "
        f"tau_quad={CS_TAU_START}->{CS_TAU_END} epochs={args.epochs} "
        f"n_train={len(train)} n_probe={len(probe)} pure_hyp={pure_report.get('pure_hyp_pass')}"
    )

    use_mlflow = not args.no_mlflow
    mlflow = None
    mlflow_run_id: str | None = None
    if use_mlflow:
        try:
            import mlflow as _mlflow
            import os

            mlflow = _mlflow
            uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow.start_run(run_name=MLFLOW_RUN_NAME)
            mlflow_run_id = str(mlflow.active_run().info.run_id)
            mlflow.set_tags(
                {
                    "card": GATE_ID,
                    "display_lineage": DISPLAY_LINEAGE,
                    "init": "lift_radius_qualified_warmstart",
                    "alias_untouched": "true",
                    "biology_pass": "pending",
                    "affinity_pass": "false",
                    "pearson": "parked",
                    "pure_hyp_strict": "true" if args.pure_hyp_strict else "false",
                    "tau_schedule": "quadratic",
                    "freeze": "entire_frontend_geometry_pass",
                }
            )
            mlflow.log_params(
                {
                    "epochs": int(args.epochs),
                    "n_train": len(train),
                    "n_probe": len(probe),
                    "tau_start": CS_TAU_START,
                    "tau_end": CS_TAU_END,
                    "tau_schedule": "quadratic",
                    "wrap_max": CS_WRAP_MAX,
                    "boundary_radius": CS_BOUNDARY_RADIUS,
                    "lr_hyperbolic": CS_LR_HYP,
                    "lr_backbone_planned": CS_LR_BACKBONE,
                    "lr_backbone_applied": 0.0,
                    "weight_decay": CS_WEIGHT_DECAY,
                    "freeze": "entire_frontend",
                    "frontend_sha256": digest,
                    "seed": int(args.seed),
                    "probe_every": int(args.probe_every),
                    "pure_hyp_pass": 1 if pure_report.get("pure_hyp_pass") else 0,
                }
            )
            print(f"{TAG} MLflow run_id={mlflow_run_id} uri={uri} exp={MLFLOW_EXPERIMENT}")
        except Exception as exc:  # noqa: BLE001
            print(f"{TAG} MLflow unavailable ({exc}); continuing without tracking")
            use_mlflow = False
            mlflow = None

    # Mark stamp as training
    if prior_stamp:
        running = dict(prior_stamp)
        running["execution_state"] = "TRAIN_RUNNING"
        running["mlflow_run_id"] = mlflow_run_id
        _write_stamp(args.out_stamp, running)

    best_mean = float("inf")
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    abort_reason: str | None = None
    home_entry = next(e for e in train if e["pdb_id"] == "4OBE")
    final_probe_rows: list[dict[str, Any]] = []
    equiv_residual: float | None = None

    def _abort_stamp(reason: str, probe_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        hygiene = evaluate_probe_hygiene(probe_rows or [])
        biology = evaluate_theme_biology(probe_rows or [])
        stamp = build_correct_start_stamp(
            pure_hyp=pure_report,
            hygiene=hygiene,
            biology=biology,
            equiv_residual=None,
            full_system_residual=None,
            frontend_sha256=digest,
            mlflow_run_id=mlflow_run_id,
            prior=prior_stamp,
            extra={
                "abort_reason": reason,
                "n_skip": len(skipped),
                "skip_reasons": skip_reasons,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "best_epoch": best_epoch,
                "out_dir": str(out_dir),
            },
        )
        stamp["status"] = "FAILED"
        stamp["execution_state"] = "TRAIN_ABORTED"
        _write_stamp(args.out_stamp, stamp)
        return stamp

    try:
        for epoch in range(int(args.epochs)):
            step_losses: list[float] = []
            last_metrics: dict[str, Any] | None = None
            for entry in train:
                key = (str(entry["pdb_id"]), str(entry["chain"]))
                if key in skipped:
                    continue
                try:
                    batch = _load_train_batch(
                        entry,
                        device=device,
                        pdb_dir=pdb_dir,
                        graph_cache_dir=graph_cache_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped.add(key)
                    skip_reasons[f"{key[0]}:{key[1]}"] = f"{type(exc).__name__}: {exc}"
                    print(f"{TAG} skip load {key[0]}:{key[1]} ({exc})")
                    if not _train_viable(train, skipped):
                        abort_reason = f"train_starved after skip {key[0]}:{key[1]}"
                        raise SystemExit(3) from exc
                    continue

                def _step() -> dict[str, float]:
                    return run_epoch(
                        system,
                        optimizer,
                        batch,
                        epoch=epoch,
                        radius=radius,
                        gumbel=gumbel,
                        diagnostics=diagnostics_train,
                        cv_coeff=cv_coeff,
                        moe_quota_coeff=moe_quota_coeff,
                        telemetry=telemetry,
                        volume_coeff=RV_VOLUME_COEFF,
                        dehydron_coeff=float(args.dehydron_coeff),
                        volume_sigma_target=RV_VOLUME_SIGMA,
                        volume_lam_spread=RV_VOLUME_LAM,
                        volume_mu_target=RV_VOLUME_MU,
                        volume_lam_mean=RV_VOLUME_LAM_MEAN,
                        volume_lam_barrier=RV_VOLUME_LAM_BARRIER,
                    )

                try:
                    metrics = _step()
                except Exception as exc:  # noqa: BLE001
                    if _is_oom(exc):
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        try:
                            metrics = _step()
                        except Exception as exc2:
                            if _is_oom(exc2):
                                skipped.add(key)
                                skip_reasons[f"{key[0]}:{key[1]}"] = "cuda_oom"
                                print(f"{TAG} skip OOM {key[0]}:{key[1]}")
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                if not _train_viable(train, skipped):
                                    abort_reason = f"train_starved after OOM {key[0]}:{key[1]}"
                                    raise SystemExit(3) from exc2
                                continue
                            raise
                    else:
                        raise

                if metrics.get("nan_abort", 0.0) >= 1.0:
                    abort_reason = f"nan_abort epoch={epoch} {key[0]}:{key[1]}"
                    raise SystemExit(2)
                last_metrics = metrics
                if math.isfinite(float(metrics["loss_total"])):
                    step_losses.append(float(metrics["loss_total"]))
                print(
                    f"{TAG} epoch={epoch} {key[0]}:{key[1]} "
                    f"loss={metrics['loss_total']:.4f} tau={metrics['tau_ceiling']:.3f} "
                    f"gumbel={metrics['gumbel_temperature']:.3f} "
                    f"r={metrics.get('diag_mean_radius', float('nan')):.3f} "
                    f"moe_min={metrics.get('moe_load_min', float('nan')):.3f}"
                )
                if use_mlflow and mlflow is not None:
                    step = epoch * len(train) + train.index(entry)
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            mlflow.log_metric(k, float(v), step=step)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if not step_losses or last_metrics is None:
                abort_reason = f"no_successful_steps epoch={epoch}"
                raise SystemExit(2)

            epoch_mean = float(sum(step_losses) / len(step_losses))
            history.append(
                {
                    "epoch": epoch,
                    "epoch_mean_loss": epoch_mean,
                    "n_steps": len(step_losses),
                    "last": last_metrics,
                }
            )
            payload = {
                "epoch": epoch,
                "model": system.state_dict(),
                "cfg": cfg,
                "curvature": curvature,
                "metrics": last_metrics,
                "epoch_mean_loss": epoch_mean,
                "display_lineage": DISPLAY_LINEAGE,
                "gate_id": GATE_ID,
                "init": "lift_radius_qualified_warmstart",
                "frontend_sha256": digest,
            }
            torch.save(payload, last_path)
            if epoch_is_best_eligible(last_metrics) and epoch_mean < best_mean:
                best_mean = epoch_mean
                best_epoch = epoch
                torch.save(payload, best_path)
                print(f"{TAG} new best epoch={epoch} mean_loss={epoch_mean:.4f}")
            if use_mlflow and mlflow is not None:
                mlflow.log_metric("epoch_mean_loss", epoch_mean, step=epoch)
                mlflow.log_metric("n_skip", float(len(skipped)), step=epoch)
                mlflow.log_metric("tau_ceiling", float(radius.tau_ceiling(epoch)), step=epoch)

            if (epoch + 1) % int(args.probe_every) == 0:
                print(f"{TAG} probe after epoch {epoch + 1}")
                probe_rows = _observe_panel(
                    probe,
                    system=system,
                    curvature=curvature,
                    device=device,
                    pdb_dir=pdb_dir,
                    diagnostics=diagnostics_h3,
                    tau_ceil=CS_TAU_END,
                )
                hygiene = evaluate_probe_hygiene(probe_rows)
                if use_mlflow and mlflow is not None:
                    mlflow.log_metric("probe_mean_sat", hygiene["mean_sat"], step=epoch)
                    mlflow.log_metric("probe_mean_spread", hygiene["mean_spread"], step=epoch)
                    mlflow.log_metric("probe_mean_h_norm", hygiene["mean_h_norm"], step=epoch)
                    bio_mid = evaluate_theme_biology(probe_rows)
                    mlflow.log_metric("probe_mean_theme_auprc", bio_mid["mean_theme_auprc"], step=epoch)
                    mlflow.log_metric("probe_n_themes_pass_auprc", bio_mid["n_themes_pass_floor"], step=epoch)
                    for tr in bio_mid.get("per_theme", []):
                        if tr.get("auprc") is not None and tr.get("theme"):
                            mlflow.log_metric("theme_auprc_%s" % tr["theme"], float(tr["auprc"]), step=epoch)

        if best_path.is_file():
            best_blob = torch.load(best_path, map_location=device, weights_only=False)
            system.load_state_dict(best_blob["model"], strict=False)
            print(f"{TAG} restored best epoch={best_epoch} mean_loss={best_mean:.4f}")
        else:
            print(f"{TAG} no eligible best ckpt — observing last weights")

        final_probe_rows = _observe_panel(
            probe,
            system=system,
            curvature=curvature,
            device=device,
            pdb_dir=pdb_dir,
            diagnostics=diagnostics_h3,
            tau_ceil=CS_TAU_END,
        )
        home_rows = _observe_panel(
            [home_entry],
            system=system,
            curvature=curvature,
            device=device,
            pdb_dir=pdb_dir,
            diagnostics=diagnostics_h3,
            tau_ceil=CS_TAU_END,
        )

        # Equivariance residual on first loaded probe batch
        equiv_residual = None
        full_system_residual = None
        for entry in probe:
            try:
                batch = _load_train_batch(
                    entry,
                    device=device,
                    pdb_dir=pdb_dir,
                    graph_cache_dir=graph_cache_dir,
                )
                eq = evaluate_lift_spine_equivariance(
                    system,
                    batch,
                    tau_ceiling=CS_TAU_END,
                    seed=int(args.seed),
                )
                equiv_residual = float(eq["lift_spine_residual"])
                full_system_residual = float(eq["full_system_residual"])
                print(
                    f"{TAG} Δ_equiv_lift={equiv_residual:.6e} "
                    f"Δ_full={full_system_residual:.6e} "
                    f"on {entry['pdb_id']}:{entry['chain']}"
                )
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{TAG} equiv probe skip {entry['pdb_id']}: {exc}")

        hygiene = evaluate_probe_hygiene(final_probe_rows)
        biology = evaluate_theme_biology(final_probe_rows)
        print(f"{TAG} biology={biology}", flush=True)
        stamp = build_correct_start_stamp(
            pure_hyp=pure_report,
            hygiene=hygiene,
            biology=biology,
            equiv_residual=equiv_residual,
            full_system_residual=full_system_residual,
            frontend_sha256=digest,
            mlflow_run_id=mlflow_run_id,
            prior=prior_stamp,
            extra={
                "n_skip": len(skipped),
                "skip_reasons": skip_reasons,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "best_epoch": best_epoch,
                "best_mean_loss": best_mean if best_epoch is not None else None,
                "out_dir": str(out_dir),
                "load_info": load_info,
                "home_probe": home_rows[0] if home_rows else None,
                "per_probe": final_probe_rows,
                "freeze": "entire_frontend_geometry_pass",
                "lr_backbone_applied": 0.0,
            },
        )
        _write_stamp(args.out_stamp, stamp)
        (out_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_mean_loss": best_mean if best_epoch is not None else None,
                    "n_skip": len(skipped),
                    "stamp": str(args.out_stamp),
                    "status": stamp.get("status"),
                    "gates": stamp.get("gates"),
                    "frontend_sha256": digest,
                    "display_lineage": DISPLAY_LINEAGE,
                    "mlflow_run_id": mlflow_run_id,
                    "equiv_residual": equiv_residual,
                },
                indent=2,
            )
            + "\n"
        )
        if use_mlflow and mlflow is not None:
            for gname, g in (stamp.get("gates") or {}).items():
                if isinstance(g, dict):
                    mlflow.log_metric(f"gate_{gname}_pass", 1.0 if g.get("pass") else 0.0)
                    val = g.get("value")
                    if isinstance(val, (int, float)) and math.isfinite(float(val)):
                        mlflow.log_metric(f"gate_{gname}_value", float(val))
            mlflow.set_tag("final_status", stamp.get("status"))
            mlflow.log_artifact(str(args.out_stamp))
            if best_path.is_file():
                mlflow.log_artifact(str(best_path))
            mlflow.log_artifact(str(out_dir / "run_summary.json"))
        print(f"{TAG} DONE status={stamp.get('status')} gates={stamp.get('gates')}")
        return 0 if stamp.get("status") == "QUALIFIED" else 1
    except CorrectStartGuardError as exc:
        print(f"{TAG} guard: {exc}")
        _abort_stamp(str(exc))
        return 4
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        reason = abort_reason or f"system_exit_{code}"
        print(f"{TAG} abort: {reason}")
        _abort_stamp(reason, final_probe_rows)
        return code
    except Exception as exc:  # noqa: BLE001
        print(f"{TAG} fatal: {exc}")
        traceback.print_exc()
        _abort_stamp(f"{type(exc).__name__}: {exc}", final_probe_rows)
        return 1
    finally:
        if use_mlflow and mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
