"""
Phase 4 Stage 1 smoke gate — optimizer only.

Answers one question: do uncertainty_head weights move under AdamW (no AMP)?

Three assertions (all required):
  1. Every uncertainty_head parameter delta > 1e-6
  2. loss_epoch1 < loss_epoch0
  3. r_epi_bf_resid_epoch1 != 0.040 (null-run stuck probe)

Direction guard (epoch-1 regression):
  delta_r_epi_bf = r1 - r0 must be > -0.010 (λ₂ dominating shows up immediately)

Does not change corpus, logging, or p2_bridge — uses disc_target manifest as-is.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
DEFAULT_MANIFEST = Path("manifests/v6_corpus_disc_target.json")
NULL_RUN_R_EPI_BF = 0.040
MIN_PARAM_DELTA = 1e-6
MAX_R_EPI_BF_DROP = -0.010  # epoch-1: allow small dip, catch λ₂-dominated regression


def _snapshot_uncertainty_head(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.uncertainty_head.parameters()]


def _snapshot_frozen(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "uncertainty_head" not in name
    }


def _full_p4_coeffs(phase_cfg) -> dict[str, float]:
    """Full λ₁/λ₂ for smoke — skip ramp so loss descent is unambiguous."""
    coeffs = phase_cfg.coeffs.model_dump()
    if phase_cfg.epistemic_bf_align_coeff_final is not None:
        coeffs["epistemic_bf_align_coeff"] = phase_cfg.epistemic_bf_align_coeff_final
    if phase_cfg.epistemic_sasa_pen_coeff_final is not None:
        coeffs["epistemic_sasa_pen_coeff"] = phase_cfg.epistemic_sasa_pen_coeff_final
    return coeffs


def run_smoke(
    *,
    checkpoint: Path,
    pdb_dir: Path,
    manifest: Path,
    device: str,
    lr: float = 1e-4,
) -> dict[str, float]:
    from experiments.diagnostics.embedding_occupancy_audit import load_audit_model
    from experiments.training.v6.corpus import load_training_proteins
    from experiments.training.v6.train_loop import (
        build_p4_optimizer,
        set_p4_uncertainty_only_freeze,
        train_epoch,
    )
    from science.training.config import p4_epistemic_decoupling_phase_config

    proteins, failed = load_training_proteins(
        pdb_dir,
        manifest,
        max_proteins=None,
        use_cache=True,
    )
    if failed or len(proteins) < 2:
        raise RuntimeError(f"Need ≥2 proteins for smoke test (loaded {len(proteins)}, failed {failed})")

    model, version = load_audit_model(checkpoint, device)
    if version != "v6":
        raise RuntimeError(f"Expected v6 checkpoint, got {version}")

    phase_cfg = p4_epistemic_decoupling_phase_config(lr=lr, epochs=2)
    coeffs = _full_p4_coeffs(phase_cfg)
    holdouts = frozenset({"1IVO", "4MNE"})

    set_p4_uncertainty_only_freeze(model)
    optimizer = build_p4_optimizer(model, lr=lr)
    params_before = _snapshot_uncertainty_head(model)
    frozen_before = _snapshot_frozen(model)

    logger.info("Smoke epoch 0 (baseline loss)...")
    losses0 = train_epoch(
        model,
        optimizer,
        proteins,
        coeffs,
        device=device,
        use_amp=False,
        epistemic_decoupling_holdouts=holdouts,
        epistemic_uncertainty_only_train=True,
    )

    logger.info("Smoke epoch 1...")
    losses1 = train_epoch(
        model,
        optimizer,
        proteins,
        coeffs,
        device=device,
        use_amp=False,
        epistemic_decoupling_holdouts=holdouts,
        epistemic_uncertainty_only_train=True,
    )

    params_after = _snapshot_uncertainty_head(model)
    deltas = [(after - before).abs().max().item() for before, after in zip(params_before, params_after)]
    min_delta = min(deltas)
    max_delta = max(deltas)

    frozen_ok = True
    for name, param in model.named_parameters():
        if "uncertainty_head" in name:
            continue
        diff = (param.detach() - frozen_before[name]).abs().max().item()
        if diff > 1e-7:
            frozen_ok = False
            logger.error("Frozen param moved: %s max_diff=%.2e", name, diff)

    r0 = float(losses0.get("r_epi_bf_resid", 0.0))
    r1 = float(losses1.get("r_epi_bf_resid", 0.0))
    delta_r_epi_bf = r1 - r0
    partial0 = float(losses0.get("partial_epi_sasa_given_depth", 0.0))
    partial1 = float(losses1.get("partial_epi_sasa_given_depth", 0.0))
    total0 = float(losses0["total"])
    total1 = float(losses1["total"])

    logger.info(
        "Smoke results: min_param_delta=%.2e max_param_delta=%.2e "
        "loss %.4f→%.4f r_epi_bf_resid %.4f→%.4f (Δ=%.4f) "
        "partial(epi,sasa|depth) %.3f→%.3f frozen_ok=%s",
        min_delta,
        max_delta,
        total0,
        total1,
        r0,
        r1,
        delta_r_epi_bf,
        partial0,
        partial1,
        frozen_ok,
    )

    assert all(d > MIN_PARAM_DELTA for d in deltas), (
        f"uncertainty_head param delta min={min_delta:.2e} — expected all > {MIN_PARAM_DELTA}"
    )
    assert total1 < total0, f"loss did not descend: {total0:.4f} → {total1:.4f}"
    assert abs(r1 - NULL_RUN_R_EPI_BF) > 1e-4, (
        f"r_epi_bf_resid stuck at null-run value {NULL_RUN_R_EPI_BF}: got {r1:.4f}"
    )
    assert delta_r_epi_bf > MAX_R_EPI_BF_DROP, (
        f"r_epi_bf_resid dropped {delta_r_epi_bf:.4f} — λ₂ likely dominating "
        f"(threshold {MAX_R_EPI_BF_DROP})"
    )
    assert frozen_ok, "non-uncertainty_head parameters changed during smoke test"

    return {
        "min_param_delta": min_delta,
        "max_param_delta": max_delta,
        "loss_epoch0": total0,
        "loss_epoch1": total1,
        "r_epi_bf_resid_epoch0": r0,
        "r_epi_bf_resid_epoch1": r1,
        "delta_r_epi_bf_resid": delta_r_epi_bf,
        "partial_epi_sasa_epoch0": partial0,
        "partial_epi_sasa_epoch1": partial1,
        "frozen_ok": float(frozen_ok),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    try:
        results = run_smoke(
            checkpoint=args.checkpoint,
            pdb_dir=args.pdb_dir,
            manifest=args.manifest,
            device=args.device,
            lr=args.lr,
        )
    except AssertionError as exc:
        logger.error("SMOKE FAILED: %s", exc)
        return 1

    logger.info("SMOKE PASSED: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
