"""Seal the Fix-1 healthy trunk into a self-describing fat checkpoint.

Writes ``v66_healthy_sealed.pt`` next to the locked run artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.training.v66.healthy_fix1 import HEALTHY_FIX1_CKPT, HEALTHY_FIX1_RUN_DIR


def seal(*, run_dir: Path, out: Path) -> Path:
    phase = torch.load(run_dir / "phase_12.pt", map_location="cpu", weights_only=False)
    epoch = torch.load(run_dir / "epochs" / "epoch_030.pt", map_location="cpu", weights_only=False)
    best_disc = torch.load(run_dir / "v66_best_disc.pt", map_location="cpu", weights_only=False)

    sd_p = phase["model_state_dict"]
    sd_e = epoch["model_state_dict"]
    if set(sd_p.keys()) != set(sd_e.keys()):
        raise RuntimeError("phase_12 vs epoch_030 state_dict key mismatch")
    for k in sd_p:
        if not torch.equal(sd_p[k], sd_e[k]):
            raise RuntimeError(f"phase_12 vs epoch_030 tensor mismatch: {k}")

    arch = dict(best_disc["architecture"])
    arch["version"] = "v6.6"
    sealed = {
        "global_epoch": 30,
        "phase": 12,
        "phase_name": phase.get("phase_name") or best_disc.get("phase_name") or "phase_12",
        "model_state_dict": sd_p,
        "training_config": dict(epoch["training_config"]),
        "architecture": arch,
        "gnn_lineage": "v6.6",
        "sealed_from": run_dir.name,
        "source_weights": "phase_12.pt",
        "source_training_config": "epochs/epoch_030.pt",
        "source_architecture": (
            "v66_best_disc.pt (architecture block only; ep7 metrics discarded)"
        ),
        "note": (
            "Self-describing healthy Fix-1+S4 trunk. Prefer this over bare phase_12.pt."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sealed, out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HEALTHY_FIX1_RUN_DIR)
    p.add_argument("--out", type=Path, default=HEALTHY_FIX1_CKPT)
    args = p.parse_args()
    path = seal(run_dir=args.run_dir, out=args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
