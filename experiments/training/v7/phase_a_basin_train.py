#!/usr/bin/env python3
"""Phase A nucleotide OFF↔ON basin continue from HEALTHY_V7_CKPT.

Pre-reg: docs/specs/tokyo-eye-v7/phase-a-nucleotide-basin-train-prereg.md
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
from science.dtie.common.kras_g12_graft import neighborhood_n12
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.dtie.v5.gnn.model import build_optimizer
from science.dtie.v66.loss import gosp_loss_v6
from science.tokyo_eye.basin_contrastive import (
    basin_contrastive_margin_loss,
    r_star_indices,
    structure_embedding_logmap0,
)
from science.training.gnn_lineage import load_model_from_checkpoint

PREREG = Path("data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json")
DEFAULT_MANIFEST = Path("manifests/v7_phase_a_nucleotide_basin_v1.json")
DEFAULT_RUN = "tokyo_eye_v7_phase_a_nucleotide_basin_v1"


def _curvature(output: dict[str, Any], model: torch.nn.Module) -> torch.Tensor:
    c = (output.get("audit_trail") or {}).get("curvature_value")
    if c is not None:
        return c if torch.is_tensor(c) else torch.tensor(float(c))
    return model.curvature


def _disc_r_mean(output: dict[str, Any]) -> float:
    disc = output.get("hyp_projections_2d")
    if disc is None:
        disc = output.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    r = torch.linalg.vector_norm(disc.float(), dim=-1)
    return float(r.mean().item())


def _graph_indices_for_rstar(prot: dict[str, Any], partner_for_n12: dict[str, Any] | None) -> list[int]:
    ids = list(prot.get("residue_ids") or [])
    idx_map = residue_index_map(ids)
    present = set(idx_map.keys())
    n12: list[int] = []
    try:
        if partner_for_n12 is not None:
            n12 = neighborhood_n12(partner_for_n12, prot)
        else:
            n12 = neighborhood_n12(prot, prot)
    except Exception:  # noqa: BLE001 — fall back to switch-only
        n12 = []
    resseqs = r_star_indices(present, n12)
    return [idx_map[r] for r in resseqs if r in idx_map]


def _physics_loss(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    loss_coeffs: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    output = model(data)
    if (
        isinstance(output.get("cone_depth"), torch.Tensor)
        and output["cone_depth"].shape[0] > n
    ):
        from experiments.training.v66.train_loop import _slice_residue_outputs

        output = _slice_residue_outputs(output, n)
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
    return losses, output, data


def _load_proteins(manifest: Path, pdb_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(manifest.read_text())
    offs: list[dict[str, Any]] = []
    ons: list[dict[str, Any]] = []
    for entry in raw["proteins"]:
        if not entry.get("enabled", True):
            continue
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain") or "A")
        prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
        if prot is None:
            raise FileNotFoundError(f"failed to load {pdb_id}:{chain}")
        prot["nucleotide_basin"] = str(entry.get("nucleotide_basin") or "").upper()
        prot["pdb_id"] = pdb_id
        if prot["nucleotide_basin"] == "OFF":
            offs.append(prot)
        elif prot["nucleotide_basin"] == "ON":
            ons.append(prot)
        else:
            raise ValueError(f"{pdb_id}: nucleotide_basin must be OFF or ON")
    if not offs or not ons:
        raise RuntimeError("need at least one OFF and one ON protein")
    return offs, ons


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--output-dir", type=Path, default=Path(f"checkpoints/v7/runs/{DEFAULT_RUN}"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--basin-coeff", type=float, default=0.10)
    p.add_argument("--basin-margin", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-disc-r-mean-hold", type=float, default=0.25)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg stamp {PREREG}")
    if not args.resume.is_file():
        raise SystemExit(f"missing resume {args.resume}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    offs, ons = _load_proteins(args.manifest, args.pdb_dir)
    model = load_model_from_checkpoint(args.resume, args.device)
    model.train()
    # Ensure hyp MP primary
    if hasattr(model, "hyp_mp_primary"):
        model.hyp_mp_primary = True
    optimizer = build_optimizer(model, lr=float(args.lr))

    # Mild disc health coeffs (aligned with B′ disc continue spirit).
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
    history: list[dict[str, Any]] = []
    best_path = args.output_dir / "v7_phase_a_best.pt"
    best_gap = -1.0
    below_hold = 0

    # Pair roster: all OFF×ON
    pairs = [(o, n) for o in offs for n in ons]

    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(pairs)
        ep_losses: list[float] = []
        ep_basin: list[float] = []
        ep_disc: list[float] = []
        for off_prot, on_prot in pairs:
            optimizer.zero_grad(set_to_none=True)
            loss_off, out_off, _ = _physics_loss(
                model, off_prot, args.device, loss_coeffs=loss_coeffs
            )
            loss_on, out_on, _ = _physics_loss(
                model, on_prot, args.device, loss_coeffs=loss_coeffs
            )
            c = _curvature(out_off, model)
            idx_off = _graph_indices_for_rstar(off_prot, on_prot)
            idx_on = _graph_indices_for_rstar(on_prot, off_prot)
            z_off = structure_embedding_logmap0(
                out_off["x_hyp"],
                idx_off,
                curvature=c,
            )
            z_on = structure_embedding_logmap0(
                out_on["x_hyp"],
                idx_on,
                curvature=c,
            )
            l_basin = basin_contrastive_margin_loss(
                z_off,
                z_on,
                curvature=c,
                margin=float(args.basin_margin),
            )
            total = (
                loss_off["total"]
                + loss_on["total"]
                + float(args.basin_coeff) * l_basin
            )
            total.backward()
            optimizer.step()
            ep_losses.append(float(total.detach().cpu()))
            ep_basin.append(float(l_basin.detach().cpu()))
            ep_disc.append(_disc_r_mean(out_off))
            ep_disc.append(_disc_r_mean(out_on))

        disc_mean = float(np.nanmean(ep_disc)) if ep_disc else float("nan")
        row = {
            "epoch": epoch,
            "loss_mean": float(np.mean(ep_losses)) if ep_losses else float("nan"),
            "basin_loss_mean": float(np.mean(ep_basin)) if ep_basin else float("nan"),
            "disc_r_mean": disc_mean,
            "n_pairs": len(pairs),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss_mean']:.4f} "
            f"basin={row['basin_loss_mean']:.4f} disc_r={disc_mean:.4f}",
            flush=True,
        )

        # Save epoch snapshot
        snap = {
            "model_state_dict": model.state_dict(),
            "architecture": getattr(model, "architecture", {})
            if isinstance(getattr(model, "architecture", None), dict)
            else {"hyp_mp_primary": True},
            "training_config": {
                "phase": "v7_phase_a_nucleotide_basin",
                "basin_coeff": float(args.basin_coeff),
                "basin_margin": float(args.basin_margin),
                "resume": str(args.resume),
            },
            "epoch": epoch,
            "health": {"disc_r_mean": disc_mean},
        }
        # Prefer fat checkpoint format from sealed when possible
        try:
            donor = torch.load(args.resume, map_location="cpu", weights_only=False)
            if isinstance(donor, dict):
                for k in ("architecture", "training_config", "model_version"):
                    if k in donor and k not in snap:
                        snap[k] = donor[k]
                if "architecture" in donor and isinstance(donor["architecture"], dict):
                    arch = dict(donor["architecture"])
                    arch["hyp_mp_primary"] = True
                    snap["architecture"] = arch
        except Exception:  # noqa: BLE001
            pass
        torch.save(snap, args.output_dir / "epochs" / f"epoch_{epoch:03d}.pt")
        torch.save(snap, args.output_dir / "phase_12.pt")

        if np.isfinite(disc_mean) and disc_mean >= float(args.min_disc_r_mean_hold):
            below_hold = 0
            # Prefer higher disc among holders; also prefer lower basin loss
            score = disc_mean - 0.01 * row["basin_loss_mean"]
            if score > best_gap:
                best_gap = score
                torch.save(snap, best_path)
        else:
            below_hold += 1
            if below_hold >= 2:
                print(
                    f"ABORT: disc_r_mean < {args.min_disc_r_mean_hold} "
                    f"for {below_hold} consecutive epochs",
                    flush=True,
                )
                break

    meta = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "resume": str(args.resume),
        "manifest": str(args.manifest),
        "prereg": str(PREREG),
        "output_dir": str(args.output_dir),
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "history": history,
        "basin_coeff": float(args.basin_coeff),
        "basin_margin": float(args.basin_margin),
        "min_disc_r_mean_hold": float(args.min_disc_r_mean_hold),
    }
    (args.output_dir / "phase_a_train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in meta if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
