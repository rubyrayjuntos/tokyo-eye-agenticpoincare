#!/usr/bin/env python3
"""Disc audit: RAF1 PPI transfer run vs coverage_v2 ep164 baseline.

Compares shared anchor 4OBE and pathway nodes (RAF1/MEK/14-3-3/chaperone)
for mean radius, largest empty angular gap, and mid-disc occupancy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v6.visualization.interactive_viewer import disc_xy_from_model_output
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_STRUCTURES = [
    ("4OBE", "A", "KRAS"),
    ("3OMV", "A", "RAF1"),
    ("3EQI", "A", "MAP2K1"),
    ("2O02", "A", "YWHAZ"),
    ("3NMQ", "A", "HSP90AB1"),
    ("4MNE", "B", "BRAF"),
]


def _largest_angular_gap_deg(theta_deg: np.ndarray) -> float:
    if theta_deg.size < 2:
        return float("nan")
    t = np.sort(np.mod(theta_deg, 360.0))
    gaps = np.diff(t, append=t[0] + 360.0)
    return float(np.max(gaps))


def _disc_stats(xy: np.ndarray) -> dict[str, float]:
    r = np.linalg.norm(xy, axis=1)
    theta = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    mid = (r >= 0.15) & (r < 0.55)
    rim = r >= 0.55
    return {
        "n": float(len(r)),
        "r_mean": float(np.mean(r)),
        "r_p50": float(np.median(r)),
        "r_p90": float(np.percentile(r, 90)),
        "mid_frac": float(np.mean(mid)),
        "rim_frac": float(np.mean(rim)),
        "largest_angular_gap_deg": _largest_angular_gap_deg(theta),
        "rim_largest_angular_gap_deg": (
            _largest_angular_gap_deg(theta[r >= 0.20])
            if np.sum(r >= 0.20) >= 2
            else float("nan")
        ),
    }


@torch.inference_mode()
def _audit_one(
    model: torch.nn.Module,
    pdb_id: str,
    chain: str,
    gene: str,
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        return {"pdb_id": pdb_id, "chain": chain, "gene": gene, "error": "load_failed"}
    data = prepare_training_batch(model, prot, device, structural_disc_frozen=False)
    # Match node_emb width (feeler topology=3 vs legacy MASTER=4).
    in_f = int(model.node_emb.in_features)
    d = int(data.x.size(-1))
    if d > in_f:
        data.x = data.x[:, :in_f].contiguous()
    elif d < in_f:
        pad = torch.zeros(
            data.x.size(0),
            in_f - d,
            device=data.x.device,
            dtype=data.x.dtype,
        )
        data.x = torch.cat([data.x, pad], dim=-1)
    out = model(data)
    xy = disc_xy_from_model_output(out).detach().cpu().numpy()
    stats = _disc_stats(xy)
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "gene": gene,
        "n_residues": int(stats["n"]),
        **{k: v for k, v in stats.items() if k != "n"},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt"
        ),
    )
    p.add_argument(
        "--transfer",
        type=Path,
        default=Path("checkpoints/v66/runs/raf1_ppi_coverage_v1/v66_best_disc.pt"),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/raf1_ppi_disc_audit"),
    )
    args = p.parse_args(argv)

    if not args.transfer.is_file():
        # Fall back to latest epoch snapshot
        epoch_dir = args.transfer.parent / "epochs"
        if epoch_dir.is_dir():
            snaps = sorted(epoch_dir.glob("epoch_*.pt"))
            if snaps:
                args.transfer = snaps[-1]
                print(f"Using latest snapshot {args.transfer}")
        if not args.transfer.is_file():
            raise SystemExit(f"Transfer checkpoint missing: {args.transfer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    base = load_model_from_checkpoint(args.baseline, device)
    base.eval()
    xfer = load_model_from_checkpoint(args.transfer, device)
    xfer.eval()

    rows: list[dict[str, Any]] = []
    for pdb_id, chain, gene in DEFAULT_STRUCTURES:
        print(f"… {gene} {pdb_id}:{chain}", flush=True)
        b = _audit_one(base, pdb_id, chain, gene, args.pdb_dir, device)
        t = _audit_one(xfer, pdb_id, chain, gene, args.pdb_dir, device)
        delta: dict[str, Any] = {}
        if "error" not in b and "error" not in t:
            for k in (
                "r_mean",
                "r_p50",
                "mid_frac",
                "rim_frac",
                "largest_angular_gap_deg",
                "rim_largest_angular_gap_deg",
            ):
                delta[k] = float(t[k] - b[k]) if not (
                    isinstance(t[k], float) and math.isnan(t[k])
                ) and not (isinstance(b[k], float) and math.isnan(b[k])) else float("nan")
        rows.append({"gene": gene, "baseline": b, "transfer": t, "delta_transfer_minus_baseline": delta})
        if delta:
            print(
                f"  gap {b.get('largest_angular_gap_deg', float('nan')):.1f}→"
                f"{t.get('largest_angular_gap_deg', float('nan')):.1f} "
                f"(Δ{delta.get('largest_angular_gap_deg', float('nan')):+.1f}) "
                f"r̄ {b.get('r_mean', float('nan')):.3f}→{t.get('r_mean', float('nan')):.3f}",
                flush=True,
            )

    report = {
        "baseline_checkpoint": str(args.baseline),
        "transfer_checkpoint": str(args.transfer),
        "structures": rows,
        "summary": {
            "shared_anchor_4OBE_gap_delta": next(
                (
                    r["delta_transfer_minus_baseline"].get("largest_angular_gap_deg")
                    for r in rows
                    if r["gene"] == "KRAS"
                ),
                None,
            ),
            "shared_anchor_4OBE_r_mean_delta": next(
                (
                    r["delta_transfer_minus_baseline"].get("r_mean")
                    for r in rows
                    if r["gene"] == "KRAS"
                ),
                None,
            ),
        },
    }
    out = args.output_dir / "disc_delta.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
