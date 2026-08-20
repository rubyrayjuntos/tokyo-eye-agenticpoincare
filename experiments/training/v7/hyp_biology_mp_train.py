#!/usr/bin/env python3
"""Continue-train under tokyo_eye_v7_hyp_biology_mp_v1 (biology-only Hyp MP).

Resume: HEALTHY_V7_CKPT (weights only). Never overwrites sealed Θ.
Pre-reg: docs/specs/tokyo-eye-v7/hyp-biology-mp-prereg.md
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
from science.tokyo_eye.biology_graph import attach_biology_mp_graph
from science.tokyo_eye.thermo_edge_features import resolve_residue_records_for_prot
from science.training.gnn_lineage import load_model_from_checkpoint

PREREG = Path("data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json")
DEFAULT_RUN = "tokyo_eye_v7_hyp_biology_mp_v1"
DEFAULT_MANIFEST = Path("manifests/v6_corpus_stage_a_small_v1.json")


def _disc_r_mean(output: dict[str, Any]) -> float:
    disc = output.get("hyp_projections_2d")
    if disc is None:
        disc = output.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    return float(torch.linalg.vector_norm(disc.float(), dim=-1).mean().item())


def _attach_biology(data: Any, prot: dict[str, Any], n: int, pdb_dir: Path) -> dict[str, Any]:
    records = resolve_residue_records_for_prot(prot, pdb_dir=pdb_dir)
    coords = prot.get("ca_coords")
    if coords is None:
        raise RuntimeError(f"{prot.get('pdb_id')}: missing ca_coords")
    rho = prot["target_rho"].reshape(-1).detach().cpu().numpy()
    attach_biology_mp_graph(
        data,
        coords=coords[:n] if hasattr(coords, "__getitem__") else coords,
        rho=rho[:n],
        residue_ids=list(prot.get("residue_ids") or [])[:n],
        residue_records=records,
    )
    return getattr(data, "biology_mp_audit", {}) or {}


def _step(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    pdb_dir: Path,
    loss_coeffs: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    bio_audit = _attach_biology(data, prot, n, pdb_dir)
    if bio_audit.get("ca_in_mp") is not False:
        raise RuntimeError("ca_in_mp must be false after biology attach")
    if bool(getattr(data, "allow_ca_fallback", True)):
        raise RuntimeError("allow_ca_fallback must be false")
    output = model(data)
    if (
        isinstance(output.get("cone_depth"), torch.Tensor)
        and output["cone_depth"].shape[0] > n
    ):
        from experiments.training.v66.train_loop import _slice_residue_outputs

        output = _slice_residue_outputs(output, n)
    trail = output.get("audit_trail") or {}
    if trail.get("ca_in_mp") not in (False, None) and model.hyp_biology_mp:
        if trail.get("ca_in_mp") is True:
            raise RuntimeError("forward audit ca_in_mp=True — Cα leakage")
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
    return losses, output, bio_audit


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
            print(f"WARN: skip unloadable {pdb_id}:{chain}", flush=True)
            continue
        prot["pdb_id"] = pdb_id
        prots.append(prot)
        if max_proteins > 0 and len(prots) >= max_proteins:
            break
    if not prots:
        raise RuntimeError(f"no proteins loaded from {manifest}")
    return prots


def _fat_snap(
    model: torch.nn.Module,
    resume: Path,
    *,
    epoch: int,
    disc_r_mean: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "hyp_mp_primary": True,
            "hyp_biology_mp": True,
            "se3_aux": False,
            "allow_ca_fallback": False,
        },
        "training_config": {
            "phase": "v7_hyp_biology_mp_continue",
            "lineage_id": DEFAULT_RUN,
            "hyp_biology_mp": True,
            "allow_ca_fallback": False,
            "resume": str(resume),
            **meta,
        },
        "epoch": epoch,
        "health": {"disc_r_mean": disc_r_mean},
        "model_version": "tokyo_eye_v7_hyp_biology_mp",
    }
    try:
        donor = torch.load(resume, map_location="cpu", weights_only=False)
        if isinstance(donor, dict):
            if "architecture" in donor and isinstance(donor["architecture"], dict):
                arch = dict(donor["architecture"])
                arch["hyp_mp_primary"] = True
                arch["hyp_biology_mp"] = True
                arch["se3_aux"] = False
                snap["architecture"] = arch
            if "training_config" in donor and isinstance(donor["training_config"], dict):
                tc = dict(donor["training_config"])
                tc.update(snap["training_config"])
                snap["training_config"] = tc
            for k in ("model_version",):
                if k in donor and k not in snap:
                    snap[k] = donor[k]
    except Exception:  # noqa: BLE001
        pass
    return snap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"checkpoints/v7/runs/{DEFAULT_RUN}"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-proteins", type=int, default=12)
    p.add_argument("--min-disc-r-mean-hold", type=float, default=0.25)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")
    if not args.resume.is_file():
        raise SystemExit(f"missing resume {args.resume}")
    # Hard isolation: never write into sealed path
    sealed = HEALTHY_V7_CKPT.resolve()
    if args.output_dir.resolve() == sealed.parent and (
        args.output_dir / "v7_healthy_sealed.pt"
    ).exists():
        # ok as sibling dir content, but refuse overwriting sealed file name
        pass
    if (args.output_dir / "v7_healthy_sealed.pt").resolve() == sealed:
        raise SystemExit("refusing to use sealed healthy path as train output")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = _load_corpus(args.manifest, args.pdb_dir, max_proteins=int(args.max_proteins))
    model = load_model_from_checkpoint(args.resume, args.device)
    model.train()
    model.hyp_mp_primary = True
    model.hyp_biology_mp = True
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
        "balance_coeff": 0.0,
        "routing_entropy_sparsity_coeff": 0.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "epochs").mkdir(exist_ok=True)
    readme = args.output_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# tokyo_eye_v7_hyp_biology_mp_v1\n\n"
            "Child lineage continue: Hyp MP on biology edges only "
            "(hbond/dehydron/pi_stack/salt_bridge). "
            f"Resume: `{args.resume}`. Do not overwrite sealed healthy.\n"
            "Spec: docs/specs/tokyo-eye-v7/hyp-biology-mp-prereg.md\n"
        )

    history: list[dict[str, Any]] = []
    best_path = args.output_dir / "v7_hyp_biology_mp_best.pt"
    best_score = -1.0
    below_hold = 0

    for epoch in range(1, int(args.epochs) + 1):
        order = list(corpus)
        random.shuffle(order)
        ep_losses: list[float] = []
        ep_disc: list[float] = []
        ep_deg0: list[float] = []
        ep_n_bio: list[int] = []
        for prot in order:
            optimizer.zero_grad(set_to_none=True)
            losses, output, bio = _step(
                model,
                prot,
                args.device,
                pdb_dir=args.pdb_dir,
                loss_coeffs=loss_coeffs,
            )
            losses["total"].backward()
            optimizer.step()
            ep_losses.append(float(losses["total"].detach().cpu()))
            ep_disc.append(_disc_r_mean(output))
            ep_deg0.append(float(bio.get("degree_zero_frac", float("nan"))))
            ep_n_bio.append(int(bio.get("n_biology_edges_undirected", 0)))

        disc_mean = float(np.nanmean(ep_disc)) if ep_disc else float("nan")
        row = {
            "epoch": epoch,
            "loss_mean": float(np.mean(ep_losses)) if ep_losses else float("nan"),
            "disc_r_mean": disc_mean,
            "degree_zero_frac_mean": float(np.nanmean(ep_deg0)) if ep_deg0 else float("nan"),
            "n_biology_edges_mean": float(np.mean(ep_n_bio)) if ep_n_bio else 0.0,
            "n_proteins": len(order),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss_mean']:.4f} "
            f"disc_r={disc_mean:.4f} deg0_frac={row['degree_zero_frac_mean']:.3f} "
            f"n_bio_edges≈{row['n_biology_edges_mean']:.0f}",
            flush=True,
        )

        snap = _fat_snap(
            model,
            args.resume,
            epoch=epoch,
            disc_r_mean=disc_mean,
            meta={
                "lr": float(args.lr),
                "manifest": str(args.manifest),
                "max_proteins": int(args.max_proteins),
            },
        )
        torch.save(snap, args.output_dir / "epochs" / f"epoch_{epoch:03d}.pt")
        torch.save(snap, args.output_dir / "phase_12.pt")

        if np.isfinite(disc_mean) and disc_mean >= float(args.min_disc_r_mean_hold):
            below_hold = 0
            score = disc_mean - 0.001 * row["loss_mean"]
            if score > best_score:
                best_score = score
                torch.save(snap, best_path)
                torch.save(snap, args.output_dir / "v7_best_disc.pt")
        else:
            below_hold += 1
            if below_hold >= 2:
                print(
                    f"ABORT: disc_r_mean < {args.min_disc_r_mean_hold} "
                    f"for {below_hold} consecutive epochs",
                    flush=True,
                )
                break

    hist_path = args.output_dir / "train_history.json"
    hist_path.write_text(
        json.dumps(
            {
                "lineage_id": DEFAULT_RUN,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "resume": str(args.resume),
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
