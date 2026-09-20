"""Tokyo Eye EQU cold boot — MPtrj bank + fresh hyp spine; geometry-only.

Does not load champion / affinity / C1 weights. Does not retarget aliases.
Equiformer fully frozen. Affinity head not trained.
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

from experiments.training.v8.equ_cold_boot import (
    BOOT_BOUNDARY_RADIUS,
    BOOT_EPOCHS,
    BOOT_MIN_TRAIN,
    BOOT_PROBE_EVERY,
    BOOT_TAU_END,
    BOOT_TAU_START,
    BOOT_WRAP_MAX,
    DEFAULT_MANIFEST,
    DEFAULT_STAMP,
    PINNED_FRONTEND_SHA256,
    assert_frontend_bank,
    assert_not_forbidden_ckpt,
    build_boot_stamp,
    epoch_is_best_eligible,
    freeze_entire_frontend,
    load_boot_split,
    spine_param_group,
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
MLFLOW_RUN_NAME = "equ_cold_boot_geometry_first"
DEFAULT_FRONTEND_CKPT = Path(
    "checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt"
)


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
    for entry in entries:
        try:
            row = observe_one(
                entry=entry,
                system=system,  # type: ignore[arg-type]
                curvature=curvature,
                tau_ceil=BOOT_TAU_END,
                device=device,
                pdb_dir=pdb_dir,
                theta="equ_cold_boot",
                diagnostics=diagnostics,
            )
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
    min_train: int = BOOT_MIN_TRAIN,
) -> bool:
    remaining = [
        e for e in train if (str(e["pdb_id"]), str(e["chain"])) not in skipped
    ]
    return len(remaining) >= int(min_train)


def _write_stamp(path: Path, stamp: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"[equ-boot] stamp -> {path} hygiene_pass={stamp.get('hygiene_pass')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokyo Eye EQU cold boot")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--frontend-ckpt", type=Path, default=DEFAULT_FRONTEND_CKPT)
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--graph-cache-dir", type=Path, default=Path("data/graph_cache"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/tokyoeye/runs"))
    p.add_argument("--out-stamp", type=Path, default=DEFAULT_STAMP)
    p.add_argument("--epochs", type=int, default=BOOT_EPOCHS)
    p.add_argument("--probe-every", type=int, default=BOOT_PROBE_EVERY)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--allow-stub-frontend", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pdb_dir = Path(args.pdb_dir)
    graph_cache_dir = Path(args.graph_cache_dir)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    assert_not_forbidden_ckpt(args.frontend_ckpt)
    frontend_path = Path(args.frontend_ckpt)
    if not frontend_path.is_file():
        if args.allow_stub_frontend:
            print(f"[equ-boot] WARN missing frontend {frontend_path}; stub path")
            digest = "stub"
        else:
            raise FileNotFoundError(f"frontend bank missing: {frontend_path}")
    else:
        digest = assert_frontend_bank(frontend_path)

    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    set_dehydron_wrap_max(BOOT_WRAP_MAX)
    split = load_boot_split(args.manifest)
    train = [_resolve_entry(e, pdb_dir) for e in split["train"]]
    probe = [_resolve_entry(e, pdb_dir) for e in split["probe"]]
    skipped: set[tuple[str, str]] = set()
    skip_reasons: dict[str, str] = {}

    cfg = load_weight_map(args.weight_map)
    # Override knobs from cold-boot contract when present in cfg defaults
    cfg = dict(cfg)
    cfg.setdefault("lr_hyperbolic", 1e-4)
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

    # Fresh spine curvature must remain learned (not hardcoded)
    curvature = float(system.spine.c.detach().cpu()) if hasattr(system.spine.c, "detach") else float(system.spine.c)
    require_learned_curvature(curvature, context="equ cold boot c")
    freeze_entire_frontend(system)
    _lock_frontend_eval(system)
    optimizer = torch.optim.Adam(spine_param_group(system, lr=float(cfg["lr_hyperbolic"])))
    radius = CurriculumRadiusController(BOOT_TAU_START, BOOT_TAU_END, int(args.epochs))
    half = max(1, int(args.epochs) // 2)
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        int(args.epochs),
        schedule="exponential",
        alpha=None,
        half_epochs=half,
    )
    diagnostics_train = PoincareDiagnosticsEngine(boundary_radius=BOOT_BOUNDARY_RADIUS)
    diagnostics_h3 = PoincareDiagnosticsEngine(boundary_radius=BOOT_BOUNDARY_RADIUS)
    cv_coeff = float(cfg.get("cv_coeff", 10.0))
    moe_quota_coeff = float(cfg.get("moe_quota_coeff", 5.0))
    telemetry = dict(cfg.get("telemetry") or {})

    run_name = f"eqf_equ_cold_boot_{date.today().strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "tokyoeye_best.pt"
    last_path = out_dir / "tokyoeye_last.pt"

    print(
        f"[equ-boot] frontend sha256={digest[:16]}… mode={load_info.get('mode')} "
        f"c={curvature} wrap_max={get_dehydron_wrap_max()} device={device}"
    )
    print(
        f"[equ-boot] freeze frontend; fresh spine lr={cfg['lr_hyperbolic']} "
        f"tau={BOOT_TAU_START}->{BOOT_TAU_END} epochs={args.epochs} "
        f"n_train={len(train)} n_probe={len(probe)}"
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
                    "card": "equ_cold_boot",
                    "display_lineage": "Tokyo Eye EQU",
                    "init": "mptrj_bank_plus_fresh_spine",
                    "alias_untouched": "true",
                    "biology_pass": "false",
                    "affinity_pass": "false",
                }
            )
            mlflow.log_params(
                {
                    "epochs": int(args.epochs),
                    "n_train": len(train),
                    "n_probe": len(probe),
                    "tau_start": BOOT_TAU_START,
                    "tau_end": BOOT_TAU_END,
                    "wrap_max": BOOT_WRAP_MAX,
                    "boundary_radius": BOOT_BOUNDARY_RADIUS,
                    "lr_hyperbolic": cfg["lr_hyperbolic"],
                    "freeze": "entire_frontend",
                    "frontend_sha256": digest,
                    "seed": int(args.seed),
                    "probe_every": int(args.probe_every),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[equ-boot] MLflow unavailable ({exc}); continuing without tracking")
            use_mlflow = False
            mlflow = None

    best_mean = float("inf")
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    abort_reason: str | None = None
    home_entry = next(e for e in train if e["pdb_id"] == "4OBE")

    def _abort_stamp(reason: str, probe_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        stamp = build_boot_stamp(
            probe_rows or [],
            None,
            frontend_sha256=digest,
            wrap_max=int(get_dehydron_wrap_max()),
            boundary_radius=BOOT_BOUNDARY_RADIUS,
            tau_probe=BOOT_TAU_END,
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
                except Exception as exc:  # noqa: BLE001
                    skipped.add(key)
                    skip_reasons[f"{key[0]}:{key[1]}"] = f"{type(exc).__name__}: {exc}"
                    print(f"[equ-boot] skip load {key[0]}:{key[1]} ({exc})")
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
                                print(f"[equ-boot] skip OOM {key[0]}:{key[1]}")
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
                    f"[equ-boot] epoch={epoch} {key[0]}:{key[1]} "
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
                "display_lineage": "Tokyo Eye EQU",
                "init": "mptrj_bank_plus_fresh_spine",
                "frontend_sha256": digest,
            }
            torch.save(payload, last_path)
            if epoch_is_best_eligible(last_metrics) and epoch_mean < best_mean:
                best_mean = epoch_mean
                best_epoch = epoch
                torch.save(payload, best_path)
                print(f"[equ-boot] new best epoch={epoch} mean_loss={epoch_mean:.4f}")
            if use_mlflow and mlflow is not None:
                mlflow.log_metric("epoch_mean_loss", epoch_mean, step=epoch)
                mlflow.log_metric("n_skip", float(len(skipped)), step=epoch)

            if (epoch + 1) % int(args.probe_every) == 0:
                print(f"[equ-boot] probe after epoch {epoch + 1}")
                probe_rows = _observe_panel(
                    probe,
                    system=system,
                    curvature=curvature,
                    device=device,
                    pdb_dir=pdb_dir,
                    diagnostics=diagnostics_h3,
                )
                if use_mlflow and mlflow is not None:
                    loaded = [r for r in probe_rows if r.get("loaded")]
                    sats = [
                        float(r["boundary_saturation"])
                        for r in loaded
                        if r.get("boundary_saturation") is not None
                    ]
                    spreads = [
                        float(r["radius_spread"])
                        for r in loaded
                        if r.get("radius_spread") is not None
                    ]
                    if sats:
                        mlflow.log_metric(
                            "probe_mean_boundary_saturation",
                            float(sum(sats) / len(sats)),
                            step=epoch,
                        )
                    if spreads:
                        mlflow.log_metric(
                            "probe_mean_radius_spread",
                            float(sum(spreads) / len(spreads)),
                            step=epoch,
                        )

        if best_path.is_file():
            best_blob = torch.load(best_path, map_location=device, weights_only=False)
            system.load_state_dict(best_blob["model"], strict=False)
            print(f"[equ-boot] restored best epoch={best_epoch} mean_loss={best_mean:.4f}")
        else:
            print("[equ-boot] no eligible best ckpt — observing last weights")

        probe_rows = _observe_panel(
            probe,
            system=system,
            curvature=curvature,
            device=device,
            pdb_dir=pdb_dir,
            diagnostics=diagnostics_h3,
        )
        home_rows = _observe_panel(
            [home_entry],
            system=system,
            curvature=curvature,
            device=device,
            pdb_dir=pdb_dir,
            diagnostics=diagnostics_h3,
        )
        home_row = home_rows[0] if home_rows else None
        stamp = build_boot_stamp(
            probe_rows,
            home_row,
            frontend_sha256=digest,
            wrap_max=int(get_dehydron_wrap_max()),
            boundary_radius=BOOT_BOUNDARY_RADIUS,
            tau_probe=BOOT_TAU_END,
            extra={
                "n_skip": len(skipped),
                "skip_reasons": skip_reasons,
                "best_checkpoint": str(best_path) if best_path.is_file() else None,
                "best_epoch": best_epoch,
                "best_mean_loss": best_mean if best_epoch is not None else None,
                "mlflow_run_id": mlflow_run_id,
                "out_dir": str(out_dir),
                "load_info": load_info,
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
                    "hygiene_pass": stamp.get("hygiene_pass"),
                    "frontend_sha256": digest,
                    "display_lineage": "Tokyo Eye EQU",
                },
                indent=2,
            )
            + "\n"
        )
        if use_mlflow and mlflow is not None:
            for k, v in stamp.items():
                if isinstance(v, bool):
                    mlflow.log_metric(f"gate_{k}", 1.0 if v else 0.0)
                elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                    mlflow.log_metric(f"gate_{k}", float(v))
            mlflow.log_artifact(str(args.out_stamp))
            if best_path.is_file():
                mlflow.log_artifact(str(best_path))
        return 0 if stamp.get("hygiene_pass") else 1
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        reason = abort_reason or f"system_exit_{code}"
        print(f"[equ-boot] abort: {reason}")
        _abort_stamp(reason)
        return code
    except Exception as exc:  # noqa: BLE001
        print(f"[equ-boot] fatal: {exc}")
        traceback.print_exc()
        _abort_stamp(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if use_mlflow and mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
