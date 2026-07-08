#!/usr/bin/env python3
"""Per-residue epistemic / aleatoric diagnostic — verify head output before trusting telemetry.

Exports one row per residue with ν_epi, ν_ale, ν_total, evidence params, and sanity
metrics (spread, decoupling, total = epi + ale, τ-boundary ale lift).

Usage:
  TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/residue_uncertainty_audit.py \\
    --checkpoint checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --structures 1MBN:A,4OBE:A \\
    --json-out checkpoints/v6/diagnostics/residue_uncertainty_1mbn_route.json \\
    --csv-out checkpoints/v6/diagnostics/residue_uncertainty_1mbn_route.csv \\
    --pdb-local
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v6.gnn.evidential import uncertainty_head_is_decoupled
from science.training.evidential_validation import assess_evidential_decomposition
from science.training.uncertainty_diagnostics import (
    audit_uncertainty_sanity,
    extract_residue_uncertainty_rows,
)

logger = logging.getLogger("residue_uncertainty_audit")


def _parse_structures(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pdb, chain = item.split(":", 1)
        else:
            pdb, chain = item, "A"
        out.append((pdb.upper(), chain))
    return out


def audit_structure(
    model: torch.nn.Module,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    *,
    structural_disc_frozen: bool,
) -> dict:
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Failed to load {pdb_id}:{chain}")

    with torch.no_grad():
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural_disc_frozen
        )
        out = model(data)

    rows = extract_residue_uncertainty_rows(out, prot)
    sanity = audit_uncertainty_sanity(rows)
    decomposition = assess_evidential_decomposition(
        rows,
        decoupled_head=bool(
            getattr(model, "decoupled_uncertainty_heads", False)
            or uncertainty_head_is_decoupled(model.state_dict())
        ),
    )
    return {
        "structure_id": pdb_id.lower(),
        "chain": chain,
        "n_residues": len(rows),
        "decoupled_uncertainty_head": bool(
            getattr(model, "decoupled_uncertainty_heads", False)
            or uncertainty_head_is_decoupled(model.state_dict())
        ),
        "uncertainty_from_backbone": bool(getattr(model, "uncertainty_from_backbone", False)),
        "sanity": sanity,
        "decomposition": decomposition,
        "residues": rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Per-residue uncertainty diagnostic")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structures", required=True, help="Comma PDB:chain list, e.g. 1MBN:A,4OBE:A")
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument(
        "--structural-disc-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pdb-local", action="store_true")
    parser.add_argument("--print-top", type=int, default=10, help="Print top-N by epistemic")
    parser.add_argument(
        "--validate-decomposition",
        action="store_true",
        help="Print P7–P11 decomposition validation summary per structure",
    )
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    model = load_v6_model(args.checkpoint, args.device)
    model.eval()
    learned_c = float(model.curvature.detach().cpu().item())

    structures_out = []
    for pdb_id, chain in _parse_structures(args.structures):
        block = audit_structure(
            model,
            pdb_id,
            chain,
            args.pdb_dir,
            args.device,
            structural_disc_frozen=args.structural_disc_frozen,
        )
        structures_out.append(block)
        s = block["sanity"]
        d = block.get("decomposition") or {}
        logger.info(
            "%s:%s ok=%s epi_std=%.4f ale_std=%.4f r_epi_ale=%.3f ale_tau_lift=%.3f",
            pdb_id,
            chain,
            s["ok"],
            s["epistemic_std"],
            s["aleatoric_std"],
            s["r_epi_ale"] if s["r_epi_ale"] == s["r_epi_ale"] else float("nan"),
            s.get("aleatoric_tau_lift", float("nan")),
        )
        if args.validate_decomposition and d:
            checks = d.get("checks") or {}
            logger.info(
                "  decomposition ok=%s checks=%s r_epi_ale=%.3f spars=%s",
                d.get("ok"),
                {k: v for k, v in checks.items()},
                d.get("r_epi_ale", float("nan")),
                d.get("sparsification_reason"),
            )

        if args.print_top > 0:
            ranked = sorted(block["residues"], key=lambda r: r["epistemic"], reverse=True)
            print(f"\n--- {pdb_id}:{chain} top {args.print_top} by ν_epi ---")
            print(f"{'residue_id':<12} {'rho':>6} {'depth':>6} {'ν_epi':>8} {'ν_ale':>8} {'ν_tot':>8} E")
            for row in ranked[: args.print_top]:
                print(
                    f"{row['residue_id']:<12} {row['rho']:6.1f} {row['cone_depth']:6.3f} "
                    f"{row['epistemic']:8.4f} {row['aleatoric']:8.4f} {row['total']:8.4f} "
                    f"{row['expert']}"
                )

    payload = {
        "version": "residue_uncertainty_audit_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "learned_curvature": learned_c,
        "structures": structures_out,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON → {args.json_out}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "structure_id",
            "chain",
            "residue_id",
            "index",
            "rho",
            "tau_flag",
            "cone_depth",
            "epistemic",
            "aleatoric",
            "total",
            "expert",
            "near_tau_boundary",
        ]
        with args.csv_out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for block in structures_out:
                for row in block["residues"]:
                    writer.writerow(
                        {
                            "structure_id": block["structure_id"],
                            "chain": block["chain"],
                            **row,
                        }
                    )
        print(f"Wrote CSV → {args.csv_out}")


if __name__ == "__main__":
    main()
