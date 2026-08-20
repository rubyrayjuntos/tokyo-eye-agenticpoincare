#!/usr/bin/env python3
"""Continue cold Hyp MP funnel with sealed feelers + light MoE specialization + ER fill.

Pre-reg: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-angfill-prereg.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.v5.gnn.model import build_optimizer
from science.dtie.v66.loss import gosp_loss_v6
from science.tokyo_eye.TokyoEye import TokyoEye
from science.training.disc_occupancy import disc_occupancy_from_tensor

PREREG = Path("data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_angfill_prereg.json")
DEFAULT_RUN = "tokyo_eye_v7_cold_hyp_mp_funnel_angfill_v1"
DEFAULT_RESUME = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_v1/v7_cold_hyp_mp_funnel_best.pt"
)
DEFAULT_MANIFEST = Path("manifests/v6_corpus_stage_a_small_v1.json")
MLFLOW_EXPERIMENT = "tokyo-eyes-v7"


def _forbid_warm_resume(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.resolve()
    banned = ("v7_healthy_sealed.pt", "v66_sparsity_champion.pt", "v66_best")
    text = str(resolved)
    if "checkpoints/v66" in text or any(b in text for b in banned):
        raise SystemExit(f"angfill forbids warm resume from {path}")
    if resolved == HEALTHY_V7_CKPT.resolve():
        raise SystemExit("angfill forbids HEALTHY_V7_CKPT resume")


def _disc_r_mean(output: dict[str, Any]) -> float:
    disc = output.get("hyp_projections_2d")
    if disc is None:
        disc = output.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    return float(torch.linalg.vector_norm(disc.float(), dim=-1).mean().item())


def _disc_eff_rank(output: dict[str, Any]) -> float:
    disc = output.get("hyp_projections_2d")
    if disc is None or not torch.is_tensor(disc) or disc.shape[0] < 3:
        return float("nan")
    _, eff = disc_occupancy_from_tensor(disc.float())
    return float(eff.detach().cpu().item())


def _routing_stats(output: dict[str, Any]) -> dict[str, float]:
    w = output.get("expert_weights")
    if w is None or not torch.is_tensor(w):
        return {
            "max_hard_share": float("nan"),
            "mean_max_soft": float("nan"),
            "routing_entropy_mean_residue": float("nan"),
        }
    hard = w.argmax(dim=-1)
    share = torch.bincount(hard, minlength=w.shape[-1]).float()
    share = share / share.sum().clamp_min(1.0)
    Hm = output.get("routing_entropy_mean_residue")
    if torch.is_tensor(Hm):
        Hm_v = float(Hm.detach().cpu().item())
    else:
        Hm_v = float(Hm) if Hm is not None else float("nan")
    return {
        "max_hard_share": float(share.max().item()),
        "mean_max_soft": float(w.max(dim=-1).values.mean().item()),
        "routing_entropy_mean_residue": Hm_v,
    }


def _build_angfill_model(device: str, *, seed: int) -> TokyoEye:
    torch.manual_seed(seed)
    model = TokyoEye(
        node_dim=3,
        hidden=128,
        num_layers=6,
        num_experts=4,
        hyp_mp_primary=True,
        se3_aux=False,
        hyp_mp_layers=3,
        hyp_biology_mp=False,
        hyperbolic_gate=True,
        hyperbolic_expert_mix=False,
        topology_only_gate=True,
        gate_include_sasa=True,
        multi_rel_edge_mp=True,
        role_edge_mp=True,
        disc_projection_path="pre_routing",
        rim_fanout_forward=True,
        rim_fanout_strength=0.14,
        rim_fanout_min_r=0.20,
        geometric_angular_prior=True,
        init_seed=seed,
    )
    return model.to(device)


def _step(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    loss_coeffs: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    if bool(getattr(data, "hyp_biology_mp", False)):
        raise RuntimeError("hyp_biology_mp must stay false")
    n = residue_node_count(data, prot)
    output = model(data)
    if (
        isinstance(output.get("cone_depth"), torch.Tensor)
        and output["cone_depth"].shape[0] > n
    ):
        from experiments.training.v66.train_loop import _slice_residue_outputs

        output = _slice_residue_outputs(output, n)
    trail = output.get("audit_trail") or {}
    if not trail.get("hyp_mp_primary", True):
        raise RuntimeError("hyp_mp_primary required")
    if trail.get("se3_aux"):
        raise RuntimeError("se3_aux must be false")
    target_rho = prot["target_rho"].to(device)
    if target_rho.dim() > 1:
        target_rho = target_rho.squeeze(-1)
    ca = prot["ca_coords"].to(device)
    target_dehydron = prot.get("target_dehydron")
    if target_dehydron is not None:
        target_dehydron = target_dehydron.to(device)
        if target_dehydron.dim() > 1:
            target_dehydron = target_dehydron.squeeze(-1)
    losses = gosp_loss_v6(
        output=output,
        target_rho=target_rho[:n],
        target_dehydron=None if target_dehydron is None else target_dehydron[:n],
        ca_coords=ca[:n],
        topology_depth=False,
        **loss_coeffs,
    )
    return losses, output


def _load_corpus(manifest: Path, pdb_dir: Path, *, max_proteins: int) -> list[dict[str, Any]]:
    raw = json.loads(manifest.read_text())
    prots: list[dict[str, Any]] = []
    for entry in raw["proteins"]:
        if not entry.get("enabled", True):
            continue
        if str(entry.get("role") or "train") not in ("train", "anchor"):
            continue
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain") or "A")
        prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
        if prot is None:
            print(f"WARN: skip {pdb_id}:{chain}", flush=True)
            continue
        prot["pdb_id"] = pdb_id
        prots.append(prot)
        if max_proteins > 0 and len(prots) >= max_proteins:
            break
    if not prots:
        raise RuntimeError(f"no proteins from {manifest}")
    return prots


def _maybe_mlflow(enabled: bool, run_name: str):
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("WARN: mlflow not installed; continuing without tracking", flush=True)
        return None
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow.start_run(run_name=run_name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"checkpoints/v7/runs/{DEFAULT_RUN}"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proteins", type=int, default=12)
    p.add_argument("--min-disc-r-mean-hold", type=float, default=0.25)
    p.add_argument("--min-disc-effective-rank", type=float, default=1.5)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")
    if not args.resume.is_file():
        raise SystemExit(f"missing resume {args.resume}")
    _forbid_warm_resume(args.resume)
    if "cold_hyp_mp_funnel" not in str(args.resume):
        raise SystemExit("--resume must be cold_hyp_mp_funnel lineage checkpoint")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = _load_corpus(
        args.manifest, args.pdb_dir, max_proteins=int(args.max_proteins)
    )
    model = _build_angfill_model(args.device, seed=int(args.seed))
    blob = torch.load(args.resume, map_location=args.device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    init_note = (
        f"resume:{args.resume} partial missing={len(missing)} unexpected={len(unexpected)}"
    )
    print(f"angfill init: {init_note}", flush=True)
    print(
        f"  rim_fanout={model.rim_fanout_forward} "
        f"geom_prior={model.geometric_angular_prior}",
        flush=True,
    )

    model.train()
    model.hyp_mp_primary = True
    model.hyp_biology_mp = False
    if hasattr(model, "se3_aux"):
        model.se3_aux = False
    optimizer = build_optimizer(model, lr=float(args.lr))

    loss_coeffs = {
        "cone_coeff": 0.15,
        "neighborhood_coeff": 0.05,
        "angular_coeff": 0.02,
        "disc_occupancy_coeff": 0.70,
        "disc_depth_scale_coeff": 1.60,
        "disc_depth_scale_target": 0.60,
        "prototype_repulsion_coeff": 1.0,
        "prototype_repulsion_margin": 0.25,
        "balance_coeff": 0.01,
        "routing_entropy_sparsity_coeff": 0.0,
        "routing_load_floor_coeff": 3.0,
        "routing_load_floor_min": 0.05,
        "majority_committed_share_coeff": 0.08,
        "disc_eff_rank_coeff": 1.0,
        "disc_eff_rank_min": float(args.min_disc_effective_rank),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "epochs").mkdir(exist_ok=True)
    readme = args.output_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# tokyo_eye_v7_cold_hyp_mp_funnel_angfill_v1\n\n"
            "Continue cold best: rim_fanout + geom prior + light MoE anti-monopoly + ER.\n"
            f"Resume: `{args.resume}`\n"
            "Spec: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-angfill-prereg.md\n"
        )

    mlf_cm = _maybe_mlflow(not args.no_mlflow, DEFAULT_RUN)
    try:
        if mlf_cm is not None:
            import mlflow

            mlflow.log_params(
                {
                    "lineage_id": DEFAULT_RUN,
                    "resume": str(args.resume),
                    "lr": float(args.lr),
                    "epochs": int(args.epochs),
                    "rim_fanout_forward": True,
                    "geometric_angular_prior": True,
                    "majority_committed_share_coeff": 0.08,
                    "routing_load_floor_coeff": 3.0,
                    "balance_coeff": 0.01,
                    "disc_eff_rank_coeff": 1.0,
                    "min_disc_effective_rank": float(args.min_disc_effective_rank),
                }
            )

        history: list[dict[str, Any]] = []
        best_path = args.output_dir / "v7_cold_hyp_mp_funnel_angfill_best.pt"
        best_score = -1.0
        below_hold = 0

        for epoch in range(1, int(args.epochs) + 1):
            order = list(corpus)
            random.shuffle(order)
            ep_loss: list[float] = []
            ep_disc: list[float] = []
            ep_er: list[float] = []
            ep_hard: list[float] = []
            ep_soft: list[float] = []
            ep_Hm: list[float] = []
            for prot in order:
                optimizer.zero_grad(set_to_none=True)
                losses, output = _step(
                    model, prot, args.device, loss_coeffs=loss_coeffs
                )
                losses["total"].backward()
                optimizer.step()
                ep_loss.append(float(losses["total"].detach().cpu()))
                ep_disc.append(_disc_r_mean(output))
                ep_er.append(_disc_eff_rank(output))
                rs = _routing_stats(output)
                ep_hard.append(rs["max_hard_share"])
                ep_soft.append(rs["mean_max_soft"])
                ep_Hm.append(rs["routing_entropy_mean_residue"])

            disc_mean = float(np.nanmean(ep_disc)) if ep_disc else float("nan")
            er_mean = float(np.nanmean(ep_er)) if ep_er else float("nan")
            hard_mean = float(np.nanmean(ep_hard)) if ep_hard else float("nan")
            soft_mean = float(np.nanmean(ep_soft)) if ep_soft else float("nan")
            Hm_mean = float(np.nanmean(ep_Hm)) if ep_Hm else float("nan")
            row = {
                "epoch": epoch,
                "loss_mean": float(np.mean(ep_loss)) if ep_loss else float("nan"),
                "disc_r_mean": disc_mean,
                "disc_effective_rank_mean": er_mean,
                "max_hard_share_mean": hard_mean,
                "mean_max_soft_mean": soft_mean,
                "routing_entropy_mean_residue": Hm_mean,
                "n_proteins": len(order),
                "init": init_note,
            }
            history.append(row)
            print(
                f"epoch {epoch:03d} loss={row['loss_mean']:.4f} "
                f"disc_r={disc_mean:.4f} ER={er_mean:.4f} "
                f"hard_max={hard_mean:.3f} soft_max={soft_mean:.3f}",
                flush=True,
            )
            if mlf_cm is not None:
                import mlflow

                for k, v in row.items():
                    if k in ("init",) or not isinstance(v, (int, float)):
                        continue
                    if np.isfinite(v):
                        mlflow.log_metric(k, float(v), step=epoch)

            snap = {
                "model_state_dict": model.state_dict(),
                "architecture": {
                    "class": "TokyoEye",
                    "version": "v7",
                    "hyp_mp_primary": True,
                    "se3_aux": False,
                    "hyp_biology_mp": False,
                    "simple_graph": "ca_contact",
                    "curriculum": "funnel_disc_angfill",
                    "cold_init": False,
                    "continue_from_cold_funnel": True,
                    "node_dim": 3,
                    "hidden": 128,
                    "num_layers": 6,
                    "num_experts": 4,
                    "topology_only_gate": True,
                    "gate_include_sasa": True,
                    "hyperbolic_gate": True,
                    "hyperbolic_expert_mix": False,
                    "multi_rel_edge_mp": True,
                    "role_edge_mp": True,
                    "disc_projection_path": "pre_routing",
                    "rim_fanout_forward": True,
                    "rim_fanout_strength": 0.14,
                    "rim_fanout_min_r": 0.20,
                    "geometric_angular_prior": True,
                },
                "training_config": {
                    "phase": "v7_cold_hyp_mp_funnel_angfill",
                    "lineage_id": DEFAULT_RUN,
                    "gnn_lineage": "v7",
                    "model_version": "TokyoEye-v7",
                    "resume": str(args.resume),
                    "init": init_note,
                    "lr": float(args.lr),
                    "loss_coeffs": loss_coeffs,
                    "manifest": str(args.manifest),
                    "max_proteins": int(args.max_proteins),
                },
                "epoch": epoch,
                "health": {
                    "disc_r_mean": disc_mean,
                    "disc_effective_rank_mean": er_mean,
                    "max_hard_share_mean": hard_mean,
                    "mean_max_soft_mean": soft_mean,
                },
                "model_version": "TokyoEye-v7",
            }
            torch.save(snap, args.output_dir / "epochs" / f"epoch_{epoch:03d}.pt")
            torch.save(snap, args.output_dir / "phase_12.pt")

            er_ok = np.isfinite(er_mean) and er_mean > float(args.min_disc_effective_rank)
            disc_ok = np.isfinite(disc_mean) and disc_mean >= float(
                args.min_disc_r_mean_hold
            )
            if disc_ok:
                below_hold = 0
            else:
                below_hold += 1
                if below_hold >= 2:
                    print(
                        f"ABORT: disc_r_mean < {args.min_disc_r_mean_hold} "
                        f"for {below_hold} consecutive epochs",
                        flush=True,
                    )
                    break

            if disc_ok and er_ok:
                # Prefer higher ER, then lower monopoly, then lower loss.
                score = (
                    er_mean
                    - 0.15 * hard_mean
                    - 0.001 * row["loss_mean"]
                )
                if score > best_score:
                    best_score = score
                    torch.save(snap, best_path)
                    torch.save(snap, args.output_dir / "v7_best_disc.pt")
                    print(
                        f"  saved best ER={er_mean:.4f} hard_max={hard_mean:.3f}",
                        flush=True,
                    )

        hist_path = args.output_dir / "train_history.json"
        hist_path.write_text(
            json.dumps(
                {
                    "lineage_id": DEFAULT_RUN,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "init": init_note,
                    "manifest": str(args.manifest),
                    "history": history,
                    "best": str(best_path) if best_path.is_file() else None,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"history → {hist_path}", flush=True)
        if best_path.is_file():
            print(f"best → {best_path}", flush=True)
        else:
            print(
                "NOTE: no best saved (need disc_r≥hold AND ER>min in same epoch)",
                flush=True,
            )
    finally:
        if mlf_cm is not None:
            mlf_cm.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
