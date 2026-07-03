"""
spike_hypercore_optimizer.py — Compare geoopt RiemannianAdam vs HyperCore optimizer (Phase 1 spike).

Requires optional: pip install hypcore

Usage:
    python -m experiments.training.v6.spike_hypercore_optimizer \\
        --corpus manifests/v6_corpus_120.json \\
        --max-proteins 5 \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.training.v6.corpus import load_training_proteins
from science.training.config import TrainingConfig
from experiments.training.v6.launch_training import build_model
from experiments.training.v6.train_loop import measure_geometry_health, set_expert_dropout, train_epoch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("spike_hypercore")

PHASE1_COEFFS = {
    "evidential_coeff": 0.005,
    "balance_coeff": 0.1,
    "cone_coeff": 0.30,
    "neighborhood_coeff": 0.10,
    "angular_coeff": 0.0,
    "domain_sep_2d_coeff": 0.0,
    "domain_sep_3d_coeff": 0.0,
}


def build_geoopt_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    from science.dtie.v5.gnn.model import build_optimizer

    return build_optimizer(model, lr=lr)


def build_hypercore_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer | None:
    """Attempt HyperCore Riemannian optimizer; return None if hypcore unavailable."""
    try:
        import hypercore.optim as hc_optim  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("hypercore (hypcore) not installed — HyperCore track skipped")
        return None

    # HyperCore API may expose RiemannianAdam; fall back to torch Adam on all params
    if hasattr(hc_optim, "RiemannianAdam"):
        return hc_optim.RiemannianAdam(model.parameters(), lr=lr)
    logger.warning("hypercore.optim.RiemannianAdam not found — using torch.Adam for spike")
    return torch.optim.Adam(model.parameters(), lr=lr)


def run_track(
    name: str,
    optimizer_builder,
    proteins: list[dict[str, Any]],
    *,
    device: str,
    epochs: int,
    lr: float,
) -> dict[str, Any] | None:
    model = build_model(
        TrainingConfig(
            hidden=128,
            num_layers=6,
            num_experts=4,
            capacity_threshold=0.4,
            min_usage=0.05,
        )
    )
    model.to(device)
    set_expert_dropout(model, 0.0)
    optimizer = optimizer_builder(model, lr)
    if optimizer is None:
        return None

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        losses = train_epoch(
            model,
            optimizer,
            proteins,
            PHASE1_COEFFS,
            freeze_radial=False,
            freeze_angular=True,
            freeze_backbone=False,
            freeze_gate=False,
            device=device,
        )
        health = measure_geometry_health(model, proteins[:2], device)
        row = {**losses, **health}
        history.append(row)
        logger.info("[%s] epoch %d loss=%.4f route_H=%.3f proj=%.3f", name, epoch + 1, row["total"], row.get("routing_entropy", 0), row.get("proj_frac_mean", -1))

    return {"track": name, "epochs": epochs, "history": history, "final": history[-1] if history else {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperCore vs geoopt optimizer spike")
    parser.add_argument("--corpus", type=Path, default=Path("manifests/v6_corpus_120.json"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-proteins", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=Path("docs/training/spike_hypercore_results.json"))
    args = parser.parse_args()

    proteins, failed = load_training_proteins(args.pdb_dir, args.corpus, max_proteins=args.max_proteins)
    if not proteins:
        logger.error("No proteins loaded (%d failed)", failed)
        sys.exit(1)

    results: dict[str, Any] = {"protein_count": len(proteins), "tracks": []}
    for name, builder in (
        ("geoopt_riemannian_adam", build_geoopt_optimizer),
        ("hypercore_optimizer", build_hypercore_optimizer),
    ):
        track = run_track(name, builder, proteins, device=args.device, epochs=args.epochs, lr=args.lr)
        if track is not None:
            results["tracks"].append(track)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote spike results to %s", args.output)


if __name__ == "__main__":
    main()
