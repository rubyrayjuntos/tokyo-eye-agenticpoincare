#!/usr/bin/env python3
"""G4 mask audit — target_dehydron distribution + loss vs forward field alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v6.loss import v3_aleatoric_shaping_loss
from science.training.aleatoric_shaping_holdout import (
    G4_DEFAULT_HOLDOUT_FRACTION,
    G4_DEFAULT_MASK_SEED,
    aleatoric_shaping_holdout_masks_torch,
    corpus_protein_holdout_ids,
)

STAGE_A_MANIFEST = ROOT / "manifests/v6_corpus_stage_a_small_v1.json"


def _value_summary(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    uniq = np.unique(np.round(arr, 6))
    return {
        "dtype": str(arr.dtype),
        "n": int(arr.size),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "unique_count": int(uniq.size),
        "unique_values": uniq[:20].tolist(),
        "is_binary_01": bool(
            arr.size > 0
            and np.all((arr == 0.0) | (arr == 1.0))
            and set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0})
        ),
        "dehydron_fraction": float(np.mean(arr >= 0.5)) if arr.size else float("nan"),
    }


def audit_protein(
    prot: dict,
    model: torch.nn.Module,
    *,
    device: str,
    holdout_proteins: frozenset[str],
    structural_disc_frozen: bool,
) -> dict:
    pdb_id = str(prot.get("pdb_id", "?")).upper()
    target_dehydron = prot.get("target_dehydron")
    if target_dehydron is None:
        return {"pdb_id": pdb_id, "error": "missing target_dehydron"}

    td_np = target_dehydron.detach().cpu().numpy().reshape(-1)
    data = prepare_training_batch(
        model, prot, device, structural_disc_frozen=structural_disc_frozen
    )
    tau_from_x = data.x[:, 1].detach().cpu().numpy().reshape(-1)

    mismatch = np.abs(td_np - tau_from_x) > 1e-6
    train_mask, holdout_mask = aleatoric_shaping_holdout_masks_torch(
        target_dehydron.to(device),
        list(prot.get("residue_ids") or []),
        pdb_id=pdb_id,
        holdout_fraction=G4_DEFAULT_HOLDOUT_FRACTION,
        device=device,
        holdout_mode="protein",
        corpus_holdout_proteins=holdout_proteins,
    )
    regularity = (1.0 - target_dehydron.to(device).squeeze(-1)).clamp(0.0, 1.0)
    shaping_mask = train_mask.float() * regularity
    shaping_frac = float(shaping_mask.sum().item() / max(shaping_mask.numel(), 1))

    with torch.no_grad():
        out = model(data)
        ale = out["uncertainty"]["aleatoric"].squeeze(-1)
        dummy_loss = v3_aleatoric_shaping_loss(
            ale, target_dehydron.to(device), train_mask
        )

    dehyd_mask = td_np >= 0.5
    regular_mask = ~dehyd_mask
    ale_np = ale.detach().cpu().numpy()

    return {
        "pdb_id": pdb_id,
        "n_residues": int(td_np.size),
        "target_dehydron": _value_summary(td_np),
        "data_x_tau_flag": _value_summary(tau_from_x),
        "target_vs_x_mismatch_n": int(mismatch.sum()),
        "target_vs_x_max_abs_delta": float(np.max(np.abs(td_np - tau_from_x)))
        if td_np.size
        else float("nan"),
        "holdout_protein": pdb_id in holdout_proteins,
        "train_mask_fraction": float(train_mask.float().mean().item()),
        "holdout_mask_fraction": float(holdout_mask.float().mean().item()),
        "regularity_mean": float(regularity.mean().item()),
        "shaping_mask_fraction": shaping_frac,
        "var_penalty": float(dummy_loss["var_penalty"].item()),
        "aleatoric_hinge": float(dummy_loss["aleatoric_hinge"].item()),
        "ale_std_all": float(np.std(ale_np)),
        "ale_std_dehydron": float(np.std(ale_np[dehyd_mask])) if dehyd_mask.any() else float("nan"),
        "ale_std_regular": float(np.std(ale_np[regular_mask]))
        if regular_mask.any()
        else float("nan"),
        "ale_mean_shaping_mask": float(ale[shaping_mask > 0].mean().item())
        if shaping_mask.sum() > 0
        else float("nan"),
        "uncertainty_head_reads_dehydron": False,
        "loss_reads_field": "prot[target_dehydron]",
        "gate_reads_field": "data.x[:, 1]",
        "g4_eval_reads_field": "data.x[:, 1] (tau_flag in uncertainty rows)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G4 dehydron mask audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--structural-disc-frozen", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    proteins, failed = load_training_proteins(args.pdb_dir, STAGE_A_MANIFEST)
    pdb_ids = [str(p.get("pdb_id", "")).upper() for p in proteins]
    holdout_proteins = corpus_protein_holdout_ids(
        pdb_ids, holdout_fraction=G4_DEFAULT_HOLDOUT_FRACTION, seed=G4_DEFAULT_MASK_SEED
    )

    model = load_v6_model(args.checkpoint, args.device)
    model.eval()

    per_protein = [
        audit_protein(
            prot,
            model,
            device=args.device,
            holdout_proteins=holdout_proteins,
            structural_disc_frozen=args.structural_disc_frozen,
        )
        for prot in proteins
    ]

    all_td: list[float] = []
    all_tau_x: list[float] = []
    for prot in proteins:
        td = prot["target_dehydron"].detach().cpu().numpy().reshape(-1)
        tx = prot["data"].x[:, 1].detach().cpu().numpy().reshape(-1)
        all_td.extend(td.tolist())
        all_tau_x.extend(tx.tolist())

    corpus_td = np.array(all_td, dtype=np.float64)

    report = {
        "checkpoint": str(args.checkpoint),
        "holdout_proteins": sorted(holdout_proteins),
        "holdout_seed": G4_DEFAULT_MASK_SEED,
        "corpus_target_dehydron": _value_summary(np.array(all_td)),
        "corpus_data_x_tau_flag": _value_summary(np.array(all_tau_x)),
        "corpus_field_agreement": {
            "mismatch_residues": int(np.sum(np.abs(np.array(all_td) - np.array(all_tau_x)) > 1e-6)),
            "max_abs_delta": float(np.max(np.abs(np.array(all_td) - np.array(all_tau_x))))
            if all_td
            else float("nan"),
        },
        "field_consumers": {
            "v3_aleatoric_shaping_loss": "prot[target_dehydron]",
            "gosp_loss_v6_shaping": "prot[target_dehydron] via train_loop",
            "hyperbolic_gate": "data.x[:, 1] at model forward",
            "DecoupledEvidentialHead": "no dehydron input (backbone+routed tangent, depth, cone_width)",
            "g4_eval_stratification": "data.x[:, 1] as tau_flag",
        },
        "per_protein": per_protein,
        "interpretation": None,
    }

    td_bin = report["corpus_target_dehydron"]["is_binary_01"]
    agree = report["corpus_field_agreement"]["mismatch_residues"] == 0
    if not td_bin:
        report["interpretation"] = (
            "target_dehydron is NOT binary {0,1} — regularity=(1-dehyd) is a continuous "
            "weight, not discrete gating; inspect unique_values"
        )
    elif not agree:
        report["interpretation"] = (
            "target_dehydron and data.x[:,1] disagree — loss mask and gate/eval read "
            "different dehydron fields"
        )
    else:
        report["interpretation"] = (
            "target_dehydron is binary and matches data.x[:,1]; symmetric collapse is "
            "NOT explained by soft-mask or field split — investigate global aleatoric "
            "suppression (coeff magnitude, head clamp, or shaping applied beyond mask)"
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)


if __name__ == "__main__":
    main()
