#!/usr/bin/env python3
"""Cold Hyp MP + simple Cα graph + funnel/disc curriculum (no Fix-1 warmstart).

Pre-reg: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md
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

PREREG = Path("data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_prereg.json")
DEFAULT_RUN = "tokyo_eye_v7_cold_hyp_mp_funnel_v1"
DEFAULT_MANIFEST = Path("manifests/v6_corpus_stage_a_small_v1.json")
CUTOVER = Path("checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt")


def _disc_r_mean(output: dict[str, Any]) -> float:
    disc = output.get("hyp_projections_2d")
    if disc is None:
        disc = output.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    return float(torch.linalg.vector_norm(disc.float(), dim=-1).mean().item())


def _forbid_warm_resume(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.resolve()
    banned_names = (
        "v7_healthy_sealed.pt",
        "v66_sparsity_champion.pt",
        "v66_best",
    )
    text = str(resolved)
    if "checkpoints/v66" in text or any(b in text for b in banned_names):
        raise SystemExit(
            f"cold lineage forbids warm resume from {path} "
            "(no Fix-1 / HEALTHY_V7 weight load)"
        )
    if resolved == HEALTHY_V7_CKPT.resolve():
        raise SystemExit("cold lineage forbids HEALTHY_V7_CKPT resume")


def _build_cold_model(device: str, *, seed: int) -> TokyoEye:
    torch.manual_seed(seed)
    # Architecture flags match sealed B′ capacity (not sealed weights).
    model = TokyoEye(
        node_dim=3,  # topology_three_vector
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
        init_seed=seed,
    )
    return model.to(device)


def _maybe_load_scaffold(model: TokyoEye, scaffold: Path | None, device: str) -> str:
    if scaffold is None or not scaffold.is_file():
        return "random_init"
    _forbid_warm_resume(scaffold)
    blob = torch.load(scaffold, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    # Scaffold may be node_dim=4 / fewer flags; partial load only.
    missing, unexpected = model.load_state_dict(state, strict=False)
    return f"cutover_scaffold partial missing={len(missing)} unexpected={len(unexpected)}"


def _step(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    loss_coeffs: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    # Simple graph: leave Cα edge_index; do not attach biology MP.
    if bool(getattr(data, "hyp_biology_mp", False)):
        raise RuntimeError("hyp_biology_mp must stay false on cold funnel lineage")
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Same-run epoch only (not HEALTHY_V7 / Fix-1)",
    )
    p.add_argument(
        "--scaffold",
        type=Path,
        default=None,
        help="Optional cutover scaffold partial init (default: pure random)",
    )
    p.add_argument(
        "--use-cutover-scaffold",
        action="store_true",
        help=f"Partial-init from {CUTOVER}",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"checkpoints/v7/runs/{DEFAULT_RUN}"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proteins", type=int, default=12)
    p.add_argument("--min-disc-r-mean-hold", type=float, default=0.25)
    p.add_argument(
        "--disc-hold-warmup-epochs",
        type=int,
        default=6,
        help="Do not abort on disc hold until after this many epochs (cold climb)",
    )
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")
    _forbid_warm_resume(args.resume)
    if args.resume is not None and "cold_hyp_mp_funnel" not in str(args.resume):
        raise SystemExit("--resume only for same-run cold_hyp_mp_funnel epochs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = _load_corpus(
        args.manifest, args.pdb_dir, max_proteins=int(args.max_proteins)
    )
    model = _build_cold_model(args.device, seed=int(args.seed))
    init_note = "random_init"
    if args.resume is not None and args.resume.is_file():
        _forbid_warm_resume(args.resume)
        blob = torch.load(args.resume, map_location=args.device, weights_only=False)
        state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state, strict=False)
        init_note = f"resume:{args.resume}"
    else:
        scaffold = args.scaffold
        if scaffold is None and args.use_cutover_scaffold:
            scaffold = CUTOVER
        if scaffold is not None:
            init_note = _maybe_load_scaffold(model, scaffold, args.device)

    model.train()
    model.hyp_mp_primary = True
    model.hyp_biology_mp = False
    if hasattr(model, "se3_aux"):
        model.se3_aux = False
    optimizer = build_optimizer(model, lr=float(args.lr))

    # Funnel / disc curriculum (B′-style); no allele / Jacobian.
    loss_coeffs = {
        "cone_coeff": 0.15,
        "neighborhood_coeff": 0.05,
        "angular_coeff": 0.02,
        "disc_occupancy_coeff": 0.70,
        "disc_depth_scale_coeff": 1.60,
        "disc_depth_scale_target": 0.60,
        "prototype_repulsion_coeff": 1.0,
        "prototype_repulsion_margin": 0.25,
        "balance_coeff": 0.0,
        "routing_entropy_sparsity_coeff": 0.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "epochs").mkdir(exist_ok=True)
    readme = args.output_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# tokyo_eye_v7_cold_hyp_mp_funnel_v1\n\n"
            "Cold Hyp MP + simple Cα graph + funnel/disc curriculum.\n"
            f"Init: `{init_note}`. No Fix-1 / HEALTHY_V7 warmstart.\n"
            "Spec: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md\n"
        )

    history: list[dict[str, Any]] = []
    best_path = args.output_dir / "v7_cold_hyp_mp_funnel_best.pt"
    best_score = -1.0
    below_hold = 0

    print(f"cold init: {init_note}", flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        order = list(corpus)
        random.shuffle(order)
        ep_losses: list[float] = []
        ep_disc: list[float] = []
        for prot in order:
            optimizer.zero_grad(set_to_none=True)
            losses, output = _step(
                model, prot, args.device, loss_coeffs=loss_coeffs
            )
            losses["total"].backward()
            optimizer.step()
            ep_losses.append(float(losses["total"].detach().cpu()))
            ep_disc.append(_disc_r_mean(output))

        disc_mean = float(np.nanmean(ep_disc)) if ep_disc else float("nan")
        row = {
            "epoch": epoch,
            "loss_mean": float(np.mean(ep_losses)) if ep_losses else float("nan"),
            "disc_r_mean": disc_mean,
            "n_proteins": len(order),
            "init": init_note,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss_mean']:.4f} disc_r={disc_mean:.4f}",
            flush=True,
        )

        snap = {
            "model_state_dict": model.state_dict(),
            "architecture": {
                "class": "TokyoEye",
                "version": "v7",
                "hyp_mp_primary": True,
                "se3_aux": False,
                "hyp_biology_mp": False,
                "simple_graph": "ca_contact",
                "curriculum": "funnel_disc",
                "cold_init": True,
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
            },
            "training_config": {
                "phase": "v7_cold_hyp_mp_funnel",
                "lineage_id": DEFAULT_RUN,
                "gnn_lineage": "v7",
                "model_version": "TokyoEye-v7",
                "init": init_note,
                "lr": float(args.lr),
                "manifest": str(args.manifest),
                "max_proteins": int(args.max_proteins),
            },
            "epoch": epoch,
            "health": {"disc_r_mean": disc_mean},
            "model_version": "TokyoEye-v7",
        }
        torch.save(snap, args.output_dir / "epochs" / f"epoch_{epoch:03d}.pt")
        torch.save(snap, args.output_dir / "phase_12.pt")

        if np.isfinite(disc_mean) and disc_mean >= float(args.min_disc_r_mean_hold):
            below_hold = 0
            score = disc_mean - 0.001 * row["loss_mean"]
            if score > best_score:
                best_score = score
                torch.save(snap, best_path)
                torch.save(snap, args.output_dir / "v7_best_disc.pt")
        elif epoch > int(args.disc_hold_warmup_epochs):
            below_hold += 1
            if below_hold >= 2:
                print(
                    f"ABORT: disc_r_mean < {args.min_disc_r_mean_hold} "
                    f"for {below_hold} consecutive epochs "
                    f"(after warmup {args.disc_hold_warmup_epochs})",
                    flush=True,
                )
                break
        else:
            below_hold = 0  # warmup: climb allowed

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
