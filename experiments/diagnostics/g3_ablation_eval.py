#!/usr/bin/env python3
"""G3 A/B evaluation — decorr-only vs full epistemic decoupling circularity gate.

Runs Stage A corpus uncertainty audits on two Phase 4 checkpoints and emits
``g3_supervision_circularity_report()`` plus per-variant S6/P9/P11 summaries.

Usage:
  TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/g3_ablation_eval.py \\
    --decor-only checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_best.pt \\
    --full-supervision checkpoints/v6/runs/g3_p4_full_v1/v6_best.pt \\
    --json-out checkpoints/v6/diagnostics/g3_ablation_report.json
"""

from __future__ import annotations

import argparse
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
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.evidential_validation import (
    G3_VARIANT_DECORR_ONLY,
    G3_VARIANT_FULL_P4,
    OOD_PINNED_STRUCTURES,
    assess_evidential_decomposition,
    g3_supervision_circularity_report,
    out_of_corpus_epistemic_contrast,
    sparsification_curve,
    sparsification_error_monotone,
    uncertainty_s6_joint_pass,
)
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

logger = logging.getLogger("g3_ablation_eval")
STAGE_A_MANIFEST = ROOT / "manifests/v6_corpus_stage_a_small_v1.json"


def collect_corpus_rows(
    checkpoint: Path,
    pdb_dir: Path,
    *,
    device: str = "cpu",
    max_proteins: int | None = None,
    structural_disc_frozen: bool = True,
) -> list[dict]:
    proteins, failed = load_training_proteins(pdb_dir, STAGE_A_MANIFEST)
    if failed:
        logger.warning("Corpus load: %d proteins failed", failed)
    if max_proteins:
        proteins = proteins[:max_proteins]
    model = load_v6_model(checkpoint, device)
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_disc_frozen
            )
            out = model(data)
            rows.extend(extract_residue_uncertainty_rows(out, prot))
    return rows


def collect_ood_rows(
    checkpoint: Path,
    pdb_dir: Path,
    *,
    device: str = "cpu",
    structural_disc_frozen: bool = True,
) -> list[dict]:
    model = load_v6_model(checkpoint, device)
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for pdb_id, chain in OOD_PINNED_STRUCTURES:
            prot = load_protein_graph(pdb_id, chain, pdb_dir)
            if prot is None:
                raise RuntimeError(f"OOD structure {pdb_id}:{chain} failed to load")
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_disc_frozen
            )
            out = model(data)
            rows.extend(extract_residue_uncertainty_rows(out, prot))
    return rows


def audit_checkpoint(
    checkpoint: Path,
    pdb_dir: Path,
    *,
    variant: str,
    device: str = "cpu",
    max_proteins: int | None = None,
    structural_disc_frozen: bool = True,
) -> dict:
    rows = collect_corpus_rows(
        checkpoint,
        pdb_dir,
        device=device,
        max_proteins=max_proteins,
        structural_disc_frozen=structural_disc_frozen,
    )
    decomposition = assess_evidential_decomposition(rows, decoupled_head=True)
    curve = sparsification_curve(rows)
    p9_ok, p9_reason = sparsification_error_monotone(curve)
    in_rows = rows[: max(len(rows) // 2, 1)]
    ood_rows = collect_ood_rows(
        checkpoint, pdb_dir, device=device, structural_disc_frozen=structural_disc_frozen
    )
    p11 = out_of_corpus_epistemic_contrast(in_rows, ood_rows)
    s6_joint = uncertainty_s6_joint_pass(rows)
    return {
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "n_residues": len(rows),
        "decomposition": decomposition,
        "p9_ok": p9_ok,
        "p9_reason": p9_reason,
        "p11_ok": bool(p11.get("ok")),
        "p11": p11,
        "s6_joint": s6_joint,
        "ood_ok": bool(p11.get("ok")),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="G3 A/B epistemic supervision circularity eval")
    parser.add_argument(
        "--decor-only",
        type=Path,
        required=True,
        help="G3 variant A checkpoint (p4_head_decouple_decorr_only)",
    )
    parser.add_argument(
        "--full-supervision",
        type=Path,
        required=True,
        help="G3 variant B checkpoint (p4_head_decouple full)",
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--structural-disc-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--pdb-local", action="store_true")
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"
    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        logger.error("Set TRAINING_LOAD_FROM_PDB=1 or pass --pdb-local")
        sys.exit(2)

    for label, path in (("decor-only", args.decor_only), ("full", args.full_supervision)):
        if not path.is_file():
            logger.error("Missing %s checkpoint: %s", label, path)
            sys.exit(1)

    decor_audit = audit_checkpoint(
        args.decor_only,
        args.pdb_dir,
        variant=G3_VARIANT_DECORR_ONLY,
        device=args.device,
        max_proteins=args.max_proteins,
        structural_disc_frozen=args.structural_disc_frozen,
    )
    full_audit = audit_checkpoint(
        args.full_supervision,
        args.pdb_dir,
        variant=G3_VARIANT_FULL_P4,
        device=args.device,
        max_proteins=args.max_proteins,
        structural_disc_frozen=args.structural_disc_frozen,
    )
    g3 = g3_supervision_circularity_report(decor_audit, full_audit)

    payload = {
        "version": "g3_ablation_eval_v1",
        "decor_only": decor_audit,
        "full_supervision": full_audit,
        "g3_report": g3,
    }

    logger.info("G3 pass=%s — %s", g3["g3_pass"], g3["interpretation"])
    logger.info("decor_only flags=%s", g3["decorrelation_only"])
    logger.info("full flags=%s", g3["with_epistemic_decoupling"])
    logger.info("supervision_required=%s", g3["supervision_required_for_pass"])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.json_out)

    print(json.dumps(g3, indent=2))
    sys.exit(0 if g3["g3_pass"] else 1)


if __name__ == "__main__":
    main()
