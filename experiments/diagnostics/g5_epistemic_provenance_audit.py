#!/usr/bin/env python3
"""G5 + G5b — v3 teacher vs v6 student epistemic provenance audit (CPU, Stage A corpus).

Correlates student epistemic with SASA, ρ, frozen v3 teacher, and pinned OOD (1PGB).
Emits ``g5_epistemic_provenance_report()`` per checkpoint (includes G5b + bootstrap CI).

Usage:
  TRAINING_LOAD_FROM_PDB=1 python experiments/diagnostics/g5_epistemic_provenance_audit.py \\
    --checkpoints route_v1:checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \\
    --checkpoints g3_decorr:checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_phase4_12prot.pt \\
    --checkpoints g3_full:checkpoints/v6/runs/g3_p4_full_v1/v6_phase4_12prot.pt \\
    --json-out checkpoints/v6/diagnostics/g5_provenance_report.json \\
    --pdb-local
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diagnostics.g3_ablation_eval import collect_ood_rows
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features, prepare_training_batch
from experiments.training.v6.v2_teacher import (
    V2Teacher,
    resolve_default_v2_teacher_checkpoint,
)
from science.training.evidential_validation import (
    _pearson_r,
    g5_epistemic_provenance_report,
)
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

logger = logging.getLogger("g5_epistemic_provenance_audit")
STAGE_A_MANIFEST = ROOT / "manifests/v6_corpus_stage_a_small_v1.json"
DEFAULT_TEACHER = resolve_default_v2_teacher_checkpoint()


def _parse_checkpoints(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def audit_checkpoint(
    label: str,
    checkpoint: Path,
    teacher: V2Teacher,
    proteins: list[dict],
    *,
    pdb_dir: Path,
    device: str = "cpu",
    structural_disc_frozen: bool = True,
) -> dict:
    model = load_v6_model(checkpoint, device)
    model.eval()

    all_rows: list[dict] = []
    rows_by_structure: dict[str, list[dict]] = {}
    per_structure: dict[str, dict] = {}

    with torch.no_grad():
        for prot in proteins:
            pdb_id = str(prot.get("pdb_id", "?")).upper()
            data_student = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_disc_frozen
            )
            out_student = model(data_student)
            rows = extract_residue_uncertainty_rows(out_student, prot)

            data_teacher = attach_v6_features(prot["data"].clone().to(device))
            teacher_out = teacher.infer(data_teacher)
            t_epi = teacher_out["epistemic"].numpy().reshape(-1)
            if t_epi.shape[0] != len(rows):
                raise RuntimeError(
                    f"{pdb_id}: teacher/student residue count mismatch "
                    f"{t_epi.shape[0]} vs {len(rows)}"
                )
            for i, row in enumerate(rows):
                row["teacher_epistemic"] = float(t_epi[i])
                row["structure_id"] = pdb_id
            rows_by_structure[pdb_id] = list(rows)
            all_rows.extend(rows)

            stu = np.array([r["epistemic"] for r in rows], dtype=np.float64)
            tea = t_epi.astype(np.float64)
            sasa_arr = np.array(
                [float(r["sasa"]) for r in rows if r.get("sasa") is not None],
                dtype=np.float64,
            )
            stu_for_sasa = np.array(
                [float(r["epistemic"]) for r in rows if r.get("sasa") is not None],
                dtype=np.float64,
            )
            per_structure[pdb_id] = {
                "n_residues": len(rows),
                "epistemic_std": float(np.std(stu)),
                "teacher_epistemic_std": float(np.std(tea)),
                "r_student_teacher": _pearson_r(stu, tea),
                "r_student_sasa": _pearson_r(stu_for_sasa, sasa_arr)
                if sasa_arr.size >= 3
                else float("nan"),
            }
            logger.info(
                "%s %s: r(stu,tea)=%.3f epi_std=%.3f teacher_std=%.3f",
                label,
                pdb_id,
                per_structure[pdb_id]["r_student_teacher"],
                per_structure[pdb_id]["epistemic_std"],
                per_structure[pdb_id]["teacher_epistemic_std"],
            )

    ood_rows = collect_ood_rows(
        checkpoint,
        pdb_dir,
        device=device,
        structural_disc_frozen=structural_disc_frozen,
    )

    report = g5_epistemic_provenance_report(
        all_rows,
        per_structure=per_structure,
        ood_rows=ood_rows,
        rows_by_structure=rows_by_structure,
    )
    return {
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "g5_report": report,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="G5 + G5b epistemic provenance audit")
    parser.add_argument(
        "--checkpoints",
        action="append",
        required=True,
        help="label:path (repeatable)",
    )
    parser.add_argument("--corpus", type=Path, default=STAGE_A_MANIFEST)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cpu")
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

    if args.teacher is None or not Path(args.teacher).is_file():
        logger.error("Missing v3 teacher checkpoint: %s", args.teacher)
        sys.exit(1)

    specs = _parse_checkpoints(args.checkpoints)
    for _, path in specs:
        if not path.is_file():
            logger.error("Missing checkpoint: %s", path)
            sys.exit(1)

    proteins, failed = load_training_proteins(args.pdb_dir, args.corpus)
    if failed:
        logger.warning("Corpus load: %d proteins failed", failed)
    if not proteins:
        logger.error("No proteins loaded")
        sys.exit(1)

    teacher = V2Teacher(args.teacher, device=args.device)
    results = []
    for label, ckpt in specs:
        logger.info("=== G5 audit: %s ===", label)
        results.append(
            audit_checkpoint(
                label,
                ckpt,
                teacher,
                proteins,
                pdb_dir=args.pdb_dir,
                device=args.device,
                structural_disc_frozen=args.structural_disc_frozen,
            )
        )
        g5 = results[-1]["g5_report"]
        boot = g5.get("teacher_student_bootstrap") or {}
        g5b = g5.get("g5b") or {}
        logger.info(
            "%s distilled_sasa=%s rho_proxy=%s borderline=%s "
            "r(epi,sasa)=%.3f r(epi,ρ)=%.3f r(stu,tea)=%.3f boot_CI=[%.3f,%.3f]",
            label,
            g5.get("distilled_sasa_proxy"),
            g5.get("rho_feature_proxy"),
            g5.get("teacher_student_borderline"),
            g5.get("r_epi_sasa_marginal", float("nan")),
            g5.get("r_epi_rho_marginal", float("nan")),
            g5.get("r_student_teacher_corpus", float("nan")),
            boot.get("ci_low", float("nan")),
            boot.get("ci_high", float("nan")),
        )
        ood_res = (g5b.get("ood_contrast_rho_residual") or {})
        logger.info(
            "%s G5b OOD raw_ratio=%.3f residual_ratio=%.3f collapsed=%s",
            label,
            (g5b.get("ood_contrast_raw") or {}).get("epistemic_ood_ratio", float("nan")),
            ood_res.get("epistemic_ood_ratio_rho_residual", float("nan")),
            ood_res.get("ood_separation_collapsed_after_rho"),
        )

    payload = {
        "version": "g5_epistemic_provenance_audit_v2",
        "teacher_checkpoint": str(Path(args.teacher).resolve()),
        "n_proteins": len(proteins),
        "checkpoints": results,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.json_out)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
