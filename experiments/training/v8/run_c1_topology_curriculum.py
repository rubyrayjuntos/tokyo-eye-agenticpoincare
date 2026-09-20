"""C1 topology curriculum — 25×150 from models:/TokyoEye@champion.

Equiformer fully frozen. Geometry loss only. Does not retarget aliases,
open Sprint 10.2 / B1, or train the affinity head.
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

from experiments.training.v8.b0_topology_observation import DEFAULT_STAMP as B0_STAMP
from experiments.training.v8.c1_topology_curriculum import (
    C1_BOUNDARY_RADIUS,
    C1_EPOCHS,
    C1_GUMBEL_HALF_EPOCHS,
    C1_HOLDOUT_PROBE_EVERY,
    C1_TAU_END,
    C1_TAU_START,
    C1_WRAP_MAX,
    DEFAULT_MANIFEST,
    DEFAULT_STAMP,
    build_c1_stamp,
    epoch_is_best_eligible,
    freeze_entire_frontend,
    load_c1_split,
    spine_param_group,
    theme_counts_viable,
)
from experiments.training.v8.run_b0_topology_observation import (
    _build_system,
    _ca_records,
    _device,
    _first_ca_complete_chain,
    _load_blob,
    observe_one,
)
from experiments.training.v8.run_v8_experiment import run_epoch
from science.dtie.common.curvature_values import require_learned_curvature
from science.tokyo_eye.governance.resolve import resolve_alias_checkpoint
from science.tokyo_eye.governance.vault import sha256_file
from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
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

MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/full-stack"
MLFLOW_RUN_NAME = "c1_topology_curriculum_champion_init"


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


def _observe_panel(
    entries: list[dict[str, Any]],
    *,
    system: nn.Module,
    curvature: float,
    device: torch.device,
    pdb_dir: Path,
    diagnostics: PoincareDiagnosticsEngine,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    system.eval()
    for entry in entries:
        rows.append(
            observe_one(
                entry=entry,
                system=system,  # type: ignore[arg-type]
                curvature=curvature,
                tau_ceil=C1_TAU_END,
                device=device,
                pdb_dir=pdb_dir,
                theta="champion",
                diagnostics=diagnostics,
            )
        )
    return rows


def _holdout_vs_b0(
    holdout_rows: list[dict[str, Any]],
    b0_stamp_path: Path,
) -> list[dict[str, Any]]:
    if not b0_stamp_path.is_file():
        return []
    b0 = json.loads(b0_stamp_path.read_text())
    by_key = {
        (str(r.get("pdb_id")), str(r.get("chain"))): r
        for r in (b0.get("per_pdb") or [])
        if str(r.get("theta", "champion")) == "champion"
    }
    out: list[dict[str, Any]] = []
    keys = (
        "mean_rho_sheet",
        "mean_rho_helix",
        "mean_rho_coil",
        "boundary_saturation",
        "mean_radius",
        "moe_load_min",
    )
    for row in holdout_rows:
        key = (str(row.get("pdb_id")), str(row.get("chain")))
        prior = by_key.get(key)
        rec: dict[str, Any] = {
            "pdb_id": key[0],
            "chain": key[1],
            "b0_present": prior is not None,
            "note": "H5 compare vs B0 same PDB; not a Pass.",
        }
        if prior is None:
            out.append(rec)
            continue
        for k in keys:
            rec[f"c1_{k}"] = row.get(k)
            rec[f"b0_{k}"] = prior.get(k)
        out.append(rec)
    return out


def _write_stamp(path: Path, stamp: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2, default=str) + "\n")


def run(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="C1 topology curriculum 25×150")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--out-stamp", type=Path, default=DEFAULT_STAMP)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/tokyoeye/runs"))
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--epochs", type=int, default=C1_EPOCHS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--champion-ckpt", type=Path, default=None, help="Override only for tests")
    p.add_argument(
        "--affinity-watch",
        action="store_true",
        help="Optional end-of-run Core Pearson; abort only if Core < 0.30",
    )
    args = p.parse_args(argv)
    if args.affinity_watch:
        raise SystemExit(
            "C1 --affinity-watch is specified but not wired on this first run; "
            "omit the flag (spec: skip to save GPU)."
        )

    pdb_dir = args.pdb_dir
    if pdb_dir is None:
        docker_cache = Path("/tmp/dtie_pdb_cache")
        pdb_dir = docker_cache if docker_cache.is_dir() else DEFAULT_PDB_DIR
    graph_cache_dir = pdb_dir / "v8_graph_cache"
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    set_dehydron_wrap_max(C1_WRAP_MAX)
    split = load_c1_split(args.manifest)
    train = [_resolve_entry(e, pdb_dir) for e in split["train"]]
    holdout = [_resolve_entry(e, pdb_dir) for e in split["holdout"]]
    skipped: set[tuple[str, str]] = set()
    skip_reasons: dict[str, str] = {}

    cfg = load_weight_map(args.weight_map)
    if args.champion_ckpt is not None:
        ckpt_path = Path(args.champion_ckpt)
        resolved = {
            "path": ckpt_path,
            "version": "override",
            "run_id": None,
            "alias": "champion",
        }
    else:
        resolved = resolve_alias_checkpoint(
            alias="champion", tracking_uri=args.tracking_uri or None
        )
        ckpt_path = Path(resolved["path"])
    digest = sha256_file(ckpt_path)
    blob = _load_blob(ckpt_path, device)
    system, curvature = _build_system(cfg, blob, device)
    require_learned_curvature(curvature, context="c1 champion c")
    freeze_entire_frontend(system)
    _lock_frontend_eval(system)
    optimizer = torch.optim.Adam(spine_param_group(system, lr=float(cfg["lr_hyperbolic"])))
    radius = CurriculumRadiusController(C1_TAU_START, C1_TAU_END, C1_EPOCHS)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        C1_EPOCHS,
        schedule="exponential",
        alpha=None,
        half_epochs=C1_GUMBEL_HALF_EPOCHS,
    )
    diagnostics_train = PoincareDiagnosticsEngine(boundary_radius=C1_BOUNDARY_RADIUS)
    diagnostics_h3 = PoincareDiagnosticsEngine(boundary_radius=C1_BOUNDARY_RADIUS)
    cv_coeff = float(cfg.get("cv_coeff", 10.0))
    moe_quota_coeff = float(cfg.get("moe_quota_coeff", 5.0))
    telemetry = dict(cfg.get("telemetry") or {})

    run_name = f"eqf_c1_topology_curriculum_{date.today().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "tokyoeye_best.pt"
    last_path = out_dir / "tokyoeye_last.pt"

    print(
        f"[c1] champion v{resolved.get('version')} sha256={digest[:16]}… "
        f"c={curvature} wrap_max={get_dehydron_wrap_max()} device={device}"
    )
    print(
        f"[c1] freeze frontend; spine lr={cfg['lr_hyperbolic']} "
        f"tau={C1_TAU_START}->{C1_TAU_END} gumbel half_epochs={C1_GUMBEL_HALF_EPOCHS} "
        f"alpha=None epochs={args.epochs} n_train={len(train)}"
    )

    use_mlflow = not args.no_mlflow
    mlflow = None
    mlflow_run_id: str | None = None
    if use_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            uri = args.tracking_uri or "http://mlflow:5000"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow.start_run(run_name=MLFLOW_RUN_NAME)
            mlflow_run_id = str(mlflow.active_run().info.run_id)
            mlflow.set_tags(
                {
                    "card": "c1_topology_curriculum",
                    "alias_untouched": "true",
                    "biology_pass": "false",
                    "init": "models:/TokyoEye@champion",
                }
            )
            mlflow.log_params(
                {
                    "epochs": int(args.epochs),
                    "n_train": len(train),
                    "n_holdout": len(holdout),
                    "tau_start": C1_TAU_START,
                    "tau_end": C1_TAU_END,
                    "wrap_max": C1_WRAP_MAX,
                    "gumbel_half_epochs": C1_GUMBEL_HALF_EPOCHS,
                    "gumbel_alpha": "None",
                    "boundary_radius": C1_BOUNDARY_RADIUS,
                    "lr_hyperbolic": cfg["lr_hyperbolic"],
                    "freeze": "entire_frontend",
                    "champion_sha256": digest,
                    "champion_version": resolved.get("version"),
                    "seed": int(args.seed),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[c1] MLflow unavailable ({exc}); continuing without tracking")
            use_mlflow = False
            mlflow = None

    best_mean = float("inf")
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    abort_reason: str | None = None
    kras_entry = next(e for e in train if e["pdb_id"] == "4OBE")

    def _abort_stamp(reason: str, holdout_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = holdout_rows or []
        kras_row = None
        stamp = build_c1_stamp(
            rows,
            kras_row,
            champion_sha256=digest,
            champion_version=resolved.get("version") or "unknown",
            wrap_max=int(get_dehydron_wrap_max()),
            boundary_radius=C1_BOUNDARY_RADIUS,
            tau_probe=C1_TAU_END,
            champion_run_id=resolved.get("run_id"),
            extra={
                "abort_reason": reason,
                "n_skip": len(skipped),
                "skip_reasons": skip_reasons,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "best_epoch": best_epoch,
                "mlflow_run_id": mlflow_run_id,
                "out_dir": str(out_dir),
            },
        )
        stamp["hygiene_pass"] = False
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
                except Exception as exc:  # noqa: BLE001 — skip, do not swap topology
                    skipped.add(key)
                    skip_reasons[f"{key[0]}:{key[1]}"] = f"{type(exc).__name__}: {exc}"
                    print(f"[c1] skip load {key[0]}:{key[1]} ({exc})")
                    if not theme_counts_viable(train, skipped):
                        abort_reason = f"theme_starved after skip {key[0]}:{key[1]}"
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
                                print(f"[c1] skip OOM {key[0]}:{key[1]}")
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                if not theme_counts_viable(train, skipped):
                                    abort_reason = f"theme_starved after OOM {key[0]}:{key[1]}"
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
                    f"[c1] epoch={epoch} {key[0]}:{key[1]} "
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
            rec = {
                "epoch": epoch,
                "epoch_mean_loss": epoch_mean,
                "n_steps": len(step_losses),
                "last": last_metrics,
            }
            history.append(rec)
            payload = {
                "epoch": epoch,
                "model": system.state_dict(),
                "cfg": cfg,
                "curvature": curvature,
                "metrics": last_metrics,
                "epoch_mean_loss": epoch_mean,
            }
            torch.save(payload, last_path)
            if epoch_is_best_eligible(last_metrics) and epoch_mean < best_mean:
                best_mean = epoch_mean
                best_epoch = epoch
                torch.save(payload, best_path)
                print(f"[c1] new best epoch={epoch} mean_loss={epoch_mean:.4f}")
            if use_mlflow and mlflow is not None:
                mlflow.log_metric("epoch_mean_loss", epoch_mean, step=epoch)
                mlflow.log_metric("n_skip", float(len(skipped)), step=epoch)

            if (epoch + 1) % C1_HOLDOUT_PROBE_EVERY == 0:
                print(f"[c1] holdout probe after epoch {epoch + 1}")
                probe = _observe_panel(
                    holdout,
                    system=system,
                    curvature=curvature,
                    device=device,
                    pdb_dir=pdb_dir,
                    diagnostics=diagnostics_h3,
                )
                if use_mlflow and mlflow is not None:
                    loaded = [r for r in probe if r.get("loaded")]
                    sats = [
                        float(r["boundary_saturation"])
                        for r in loaded
                        if r.get("boundary_saturation") is not None
                    ]
                    if sats:
                        mlflow.log_metric(
                            "holdout_mean_boundary_saturation",
                            float(sum(sats) / len(sats)),
                            step=epoch,
                        )
                    for row in loaded:
                        prefix = f"probe_{row['pdb_id']}_{row['chain']}"
                        for key in (
                            "mean_radius",
                            "boundary_saturation",
                            "moe_load_min",
                            "mean_rho_sheet",
                            "mean_rho_helix",
                            "mean_rho_coil",
                        ):
                            val = row.get(key)
                            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                                mlflow.log_metric(f"{prefix}_{key}", float(val), step=epoch)

        if best_path.is_file():
            best_blob = torch.load(best_path, map_location=device, weights_only=False)
            system.load_state_dict(best_blob["model"], strict=False)
            print(f"[c1] restored best epoch={best_epoch} mean_loss={best_mean:.4f}")
        else:
            print("[c1] no eligible best ckpt — observing last weights")

        holdout_rows = _observe_panel(
            holdout,
            system=system,
            curvature=curvature,
            device=device,
            pdb_dir=pdb_dir,
            diagnostics=diagnostics_h3,
        )
        kras_row = observe_one(
            entry=kras_entry,
            system=system,
            curvature=curvature,
            tau_ceil=C1_TAU_END,
            device=device,
            pdb_dir=pdb_dir,
            theta="champion",
            diagnostics=diagnostics_h3,
        )
        stamp = build_c1_stamp(
            holdout_rows,
            kras_row,
            champion_sha256=digest,
            champion_version=resolved.get("version") or "unknown",
            wrap_max=int(get_dehydron_wrap_max()),
            boundary_radius=C1_BOUNDARY_RADIUS,
            tau_probe=C1_TAU_END,
            champion_run_id=resolved.get("run_id"),
            extra={
                "n_skip": len(skipped),
                "skip_reasons": skip_reasons,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "best_epoch": best_epoch,
                "best_epoch_mean_loss": best_mean if best_epoch is not None else None,
                "mlflow_run_id": mlflow_run_id,
                "out_dir": str(out_dir),
                "holdout_vs_b0": _holdout_vs_b0(holdout_rows, B0_STAMP),
                "history_tail": history[-5:],
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
                    "hygiene_pass": stamp["hygiene_pass"],
                    "biology_pass": False,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        if use_mlflow and mlflow is not None:
            mlflow.log_metrics(
                {
                    "hygiene_pass": 1.0 if stamp["hygiene_pass"] else 0.0,
                    "mean_holdout_boundary_saturation": float(
                        stamp["mean_holdout_boundary_saturation"]
                    ),
                    "n_holdout_moe_spread": float(stamp["n_holdout_moe_spread"]),
                    "n_skip": float(len(skipped)),
                }
            )
            mlflow.log_artifact(str(args.out_stamp))
            mlflow.log_artifact(str(out_dir / "run_summary.json"))
            if best_path.is_file():
                mlflow.log_artifact(str(best_path))
            mlflow.end_run()
        print(
            json.dumps(
                {
                    "ok": True,
                    "hygiene_pass": stamp["hygiene_pass"],
                    "biology_pass": False,
                    "n_skip": len(skipped),
                    "best_epoch": best_epoch,
                    "stamp": str(args.out_stamp),
                    "mlflow_run_id": mlflow_run_id,
                },
                indent=2,
            )
        )
        return stamp
    except SystemExit:
        if abort_reason:
            print(f"[c1] abort: {abort_reason}")
            _abort_stamp(abort_reason)
        if use_mlflow and mlflow is not None:
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[c1] abort: {exc}\n{traceback.format_exc(limit=8)}")
        _abort_stamp(f"{type(exc).__name__}: {exc}")
        if use_mlflow and mlflow is not None:
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        raise


def main() -> None:
    run()


if __name__ == "__main__":
    main()
