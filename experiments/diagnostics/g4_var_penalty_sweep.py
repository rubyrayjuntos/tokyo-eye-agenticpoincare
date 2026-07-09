#!/usr/bin/env python3
"""G4 w_var_penalty coefficient sweep — short probes + frozen separation criteria."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diagnostics.g4_mask_audit import audit_protein
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.aleatoric_shaping_holdout import (
    G4_DEFAULT_MASK_SEED,
    corpus_protein_holdout_ids,
)
from science.training.evidential_validation import (
    G4_POPULATION_GAP_STD_MULT_MIN,
    G4_POPULATION_SEPARATION_WVP_WEIGHTS,
    aleatoric_population_separation_report,
)
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

logger = logging.getLogger("g4_var_penalty_sweep")
STAGE_A_MANIFEST = ROOT / "manifests/v6_corpus_stage_a_small_v1.json"
DEFAULT_RESUME = ROOT / "checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt"


def _run_id_for_weight(w: float) -> str:
    return f"g4_wvp_{str(w).replace('.', '_')}"


def _collect_rows(model: torch.nn.Module, proteins: list[dict], device: str) -> list[dict]:
    rows: list[dict] = []
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=True
            )
            out = model(data)
            rows.extend(extract_residue_uncertainty_rows(out, prot))
    return rows


def eval_checkpoint(
    ckpt: Path,
    proteins: list[dict],
    *,
    device: str,
    holdout_proteins: frozenset[str],
) -> dict:
    model = load_v6_model(ckpt, device)
    model.eval()
    rows = _collect_rows(model, proteins, device)
    separation = aleatoric_population_separation_report(rows)
    per_protein = [
        audit_protein(
            prot,
            model,
            device=device,
            holdout_proteins=holdout_proteins,
            structural_disc_frozen=True,
        )
        for prot in proteins
    ]
    return {
        "checkpoint": str(ckpt),
        "population_separation": separation,
        "per_protein_mask_audit": per_protein,
        "n_residues": len(rows),
    }


def train_probe(
    w_var_penalty: float,
    *,
    epochs: int,
    resume: Path,
    output_root: Path,
    device: str,
) -> Path:
    run_id = _run_id_for_weight(w_var_penalty)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.training.v6.launch_training",
        "--corpus",
        str(STAGE_A_MANIFEST),
        "--output-dir",
        str(out_dir),
        "--pdb-dir",
        "/tmp/dtie_pdb_cache",
        "--device",
        device,
        "--max-proteins",
        "12",
        "--max-residues",
        "650",
        "--mlflow-uri",
        f"file:{ROOT / 'mlruns'}",
        "--resume",
        str(resume),
        "--p4-v3-aleatoric-shaping",
        "--structural-disc-frozen",
        "--p4-epistemic-lr",
        "5e-5",
        "--epistemic-decoupling-holdouts",
        "1IVO,4MNE",
        "--epochs",
        str(epochs),
        "--w-var-penalty",
        str(w_var_penalty),
        "--no-corpus-cache",
    ]
    logger.info("Training probe w_var_penalty=%.3f epochs=%d → %s", w_var_penalty, epochs, out_dir)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("TRAINING_LOAD_FROM_PDB", "1")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    ckpt = out_dir / "v6_phase4_12prot.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Expected checkpoint missing after probe: {ckpt}")
    return ckpt


def sweep_verdict(probe_results: list[dict]) -> dict:
    any_sep = any(r.get("population_separation", {}).get("ok") for r in probe_results)
    gaps = [
        float(r.get("population_separation", {}).get("gap_over_pooled_std", float("nan")))
        for r in probe_results
    ]
    global_stds = [
        float(r.get("population_separation", {}).get("global_ale_std", float("nan")))
        for r in probe_results
    ]
    best_gap_idx = int(np.nanargmax(gaps)) if gaps else -1
    if any_sep:
        verdict = "magnitude_sufficient"
        interpretation = (
            "At least one probe weight met frozen separation criterion — tune weight "
            "before architectural changes"
        )
    elif best_gap_idx >= 0 and gaps[best_gap_idx] >= 0.5 * G4_POPULATION_GAP_STD_MULT_MIN:
        verdict = "partial_gap_at_low_weight"
        interpretation = (
            "Gap opens partially at lowest weight but frozen criterion not met — "
            "may need weight below grid or architectural calibration"
        )
    else:
        verdict = "structure_likely_required"
        interpretation = (
            "Gap stays near-zero across probe weights — shared-trunk calibration "
            "(per-population affine) is next minimal bet; not stop-gradient"
        )
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "best_gap_weight": probe_results[best_gap_idx].get("w_var_penalty")
        if best_gap_idx >= 0
        else None,
        "best_gap_over_pooled_std": gaps[best_gap_idx] if best_gap_idx >= 0 else float("nan"),
        "global_stds_by_weight": {
            str(r.get("w_var_penalty")): r.get("population_separation", {}).get("global_ale_std")
            for r in probe_results
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="G4 w_var_penalty coefficient sweep")
    parser.add_argument(
        "--weights",
        default=",".join(str(w) for w in G4_POPULATION_SEPARATION_WVP_WEIGHTS),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--output-root", type=Path, default=ROOT / "checkpoints/v6/runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=ROOT / "checkpoints/v6/diagnostics/g4_wvp_sweep_report.json")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing run dirs under output-root",
    )
    args = parser.parse_args()

    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
    proteins, _ = load_training_proteins(Path("/tmp/dtie_pdb_cache"), STAGE_A_MANIFEST)
    pdb_ids = [str(p.get("pdb_id", "")).upper() for p in proteins]
    holdout_proteins = corpus_protein_holdout_ids(
        pdb_ids, holdout_fraction=0.20, seed=G4_DEFAULT_MASK_SEED
    )

    probe_results: list[dict] = []
    for w in weights:
        run_dir = args.output_root / _run_id_for_weight(w)
        if args.eval_only:
            ckpt = run_dir / "v6_phase4_12prot.pt"
            if not ckpt.is_file():
                logger.warning("Skip w=%.3f — missing %s", w, ckpt)
                continue
        else:
            ckpt = train_probe(
                w,
                epochs=args.epochs,
                resume=args.resume,
                output_root=args.output_root,
                device=args.device,
            )
        result = eval_checkpoint(
            ckpt, proteins, device=args.device, holdout_proteins=holdout_proteins
        )
        result["w_var_penalty"] = w
        result["run_dir"] = str(run_dir)
        probe_results.append(result)
        sep = result["population_separation"]
        logger.info(
            "w=%.3f gap/σ=%.3f global_σ=%.4f separation_ok=%s",
            w,
            sep.get("gap_over_pooled_std", float("nan")),
            sep.get("global_ale_std", float("nan")),
            sep.get("ok"),
        )

    report = {
        "version": "g4_var_penalty_sweep_v1",
        "weights_frozen": list(weights),
        "epochs_per_probe": args.epochs,
        "resume": str(args.resume),
        "separation_criterion_frozen": {
            "gap_over_pooled_std_min": G4_POPULATION_GAP_STD_MULT_MIN,
            "global_ale_std_min": 0.05,
            "rule": "|μ_dehyd−μ_regular|/σ_pooled ≥ gap_min AND global σ ≥ 0.05",
        },
        "rejected_fix": {
            "stop_gradient_dehydron": (
                "Does not apply — var_penalty already excludes dehydron residues; "
                "coupling is via shared trunk parameters, not activation flow from dehydron"
            ),
        },
        "architectural_fallback_if_structure_required": (
            "Per-population affine scale/shift on aleatoric output conditioned on dehydron "
            "flag — not duplicate ale_trunk"
        ),
        "probes": probe_results,
        "sweep_verdict": sweep_verdict(probe_results),
    }
    text = json.dumps(report, indent=2)
    print(text)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(text)


if __name__ == "__main__":
    main()
