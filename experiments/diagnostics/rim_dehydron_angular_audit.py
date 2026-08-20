"""Compare within-rim dehydron angular spread across checkpoints / SSOT.

Usage:
  python -m experiments.diagnostics.rim_dehydron_angular_audit \\
    --checkpoint checkpoints/v66/runs/feeler_expand_23_dbh_edges_v1/v66_phase3_23prot.pt \\
    --structures 4OBE 1R69 --learned-disc

  python -m experiments.diagnostics.rim_dehydron_angular_audit \\
    --checkpoint checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt \\
    --structures 4OBE 1R69 --structural-ssot

  make audit-rim-dehydron-angular CHECKPOINT=... STRUCTURES="4OBE 1R69"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.diagnostics.crescent_biology_projection import _load_biology_arrays
from experiments.diagnostics.embedding_occupancy_audit import load_audit_model
from science.dtie.v66.rim_dehydron_angular import compute_rim_dehydron_angular_stats


def _audit_structure(
    structure_id: str,
    *,
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    use_structural_disc_ssot: bool,
) -> dict:
    chain = "A"
    bio = _load_biology_arrays(
        structure_id,
        chain,
        model,
        pdb_dir,
        device,
        use_structural_disc_ssot=use_structural_disc_ssot,
    )
    stats = compute_rim_dehydron_angular_stats(
        structure_id=structure_id,
        disc_r=np.asarray(bio.disc_r),
        disc_theta_deg=np.asarray(bio.disc_theta_deg),
        disc_xy=np.asarray(bio.disc_xy),
        rho=np.asarray(bio.rho),
        dehydron=np.asarray(bio.dehydron),
    )
    row = stats.to_dict()
    row["disc_layout_source"] = bio.disc_layout_source
    row["chain"] = chain
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Within-rim dehydron angular audit")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint (.pt)",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["4OBE", "1R69"],
        help="PDB IDs to audit (default: spike-prone KRAS + SSOT control)",
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/rim_dehydron_angular"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--structural-ssot",
        action="store_true",
        help="Frozen Tier-1 disc (biology control)",
    )
    mode.add_argument(
        "--learned-disc",
        action="store_true",
        help="GNN-learned disc (feeler checkpoints)",
    )
    args = parser.parse_args(argv)

    use_ssot = bool(args.structural_ssot) or not args.learned_disc
    if args.learned_disc:
        use_ssot = False

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, _version = load_audit_model(str(args.checkpoint), device)
    model.eval()

    rows: list[dict] = []
    for sid in args.structures:
        try:
            rows.append(
                _audit_structure(
                    sid.upper(),
                    model=model,
                    pdb_dir=args.pdb_dir,
                    device=device,
                    use_structural_disc_ssot=use_ssot,
                )
            )
        except Exception as exc:
            print(f"WARN {sid}: {exc}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = args.checkpoint.stem
    layout = "structural_ssot" if use_ssot else "gnn_learned"
    out_path = args.output_dir / f"rim_dehydron_angular_{ckpt_stem}_{layout}.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "disc_mode": layout,
        "structures": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {out_path}\n")
    header = (
        f"{'sid':<6} {'n_rim':>5} {'θ_std':>7} {'θ_p5p95':>8} "
        f"{'max_Δθ':>7} {'eff_r':>6} {'ρ↔r':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['structure_id']:<6} "
            f"{row['n_rim_dehydron']:>5} "
            f"{row['within_rim_angular_std_deg']:>7.2f} "
            f"{row['within_rim_angular_p5_p95_span_deg']:>8.2f} "
            f"{row['max_pairwise_delta_theta_deg']:>7.1f} "
            f"{row['rim_xy_effective_rank']:>6.2f} "
            f"{row['corr_rho_disc_r']:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
