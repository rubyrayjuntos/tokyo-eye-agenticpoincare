"""Seal the v7 B′ disc-health bank into a self-describing fat checkpoint.

Writes ``v7_healthy_sealed.pt`` from ``epochs/epoch_041.pt`` (last eligible +
disc_r Pass). Prefer this over bare phase_12 / late ineligible epochs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.training.v7.healthy_bprime import (
    HEALTHY_V7_CKPT,
    HEALTHY_V7_EPOCH,
    HEALTHY_V7_EPOCH_CKPT,
    HEALTHY_V7_HEALTH_CLOSEOUT,
    HEALTHY_V7_RUN_DIR,
    HEALTHY_V7_SEAL_GATE,
)


def seal(*, run_dir: Path, epoch_ckpt: Path, out: Path, epoch: int) -> Path:
    if not epoch_ckpt.is_file():
        raise FileNotFoundError(f"missing health bank epoch: {epoch_ckpt}")
    ep = torch.load(epoch_ckpt, map_location="cpu", weights_only=False)
    if int(ep.get("global_epoch", -1)) != int(epoch):
        raise RuntimeError(
            f"epoch ckpt global_epoch={ep.get('global_epoch')} != expected {epoch}"
        )

    # Architecture block: prefer disc saver (has full arch); fall back to phase_12.
    arch_src = run_dir / "v7_best_disc.pt"
    if not arch_src.is_file():
        arch_src = run_dir / "phase_12.pt"
    if not arch_src.is_file():
        raise FileNotFoundError(f"missing architecture donor under {run_dir}")
    donor = torch.load(arch_src, map_location="cpu", weights_only=False)
    arch = dict(donor.get("architecture") or {})
    arch["version"] = "v7"
    arch["class"] = "TokyoEye"
    arch["hyp_mp_primary"] = True
    arch["se3_aux"] = False
    arch["bprime_healthy_sealed"] = True
    arch["sealed_epoch"] = int(epoch)

    tc = dict(ep.get("training_config") or donor.get("training_config") or {})
    tc["gnn_lineage"] = "v7"
    tc["hyp_mp_primary"] = True
    tc["se3_aux"] = False
    tc["hyperbolic_mp_graph"] = False

    sealed = {
        "global_epoch": int(epoch),
        "phase": int(ep.get("phase") or 12),
        "phase_name": donor.get("phase_name")
        or ep.get("phase_name")
        or "Phase 12: v6.6 feeler (geometric angular prior)",
        "model_state_dict": ep["model_state_dict"],
        "training_config": tc,
        "architecture": arch,
        "gnn_lineage": "v7",
        "model_version": "TokyoEye-v7",
        "sealed_from": run_dir.name,
        "source_weights": f"epochs/epoch_{epoch:03d}.pt",
        "source_architecture": arch_src.name,
        "health_closeout": (
            "data/gates/tokyo_eye_v7_bprime_core_radial_floor_closeout.json"
            if HEALTHY_V7_HEALTH_CLOSEOUT.exists()
            else None
        ),
        "note": (
            "Self-describing v7 B′ disc-health trunk (core radial floor Pass). "
            "Uncertainty (ale/epi) not yet recovered — use heads-only continue."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sealed, out)
    return out


def write_gate_stamp(*, out: Path, epoch: int, run_dir: Path) -> Path:
    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_bprime_healthy_sealed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checkpoint": str(out),
        "source_epoch": int(epoch),
        "source_run": run_dir.name,
        "health": {
            "disc_r_mean_min": 0.25,
            "pass": True,
            "closeout": "data/gates/tokyo_eye_v7_bprime_core_radial_floor_closeout.json",
        },
        "locks": {
            "hyp_mp_primary": True,
            "se3_aux": False,
            "hyperbolic_mp_graph": False,
        },
        "forbidden_claims": [
            "biology_pass_vs_fix1",
            "uncertainty_pass",
            "promote",
            "kras_hub_migration",
        ],
        "note": "Disc-health sealed; ale/epi recovery is a separate gate.",
    }
    HEALTHY_V7_SEAL_GATE.parent.mkdir(parents=True, exist_ok=True)
    HEALTHY_V7_SEAL_GATE.write_text(json.dumps(stamp, indent=2) + "\n")
    return HEALTHY_V7_SEAL_GATE


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=HEALTHY_V7_RUN_DIR)
    p.add_argument("--epoch-ckpt", type=Path, default=HEALTHY_V7_EPOCH_CKPT)
    p.add_argument("--epoch", type=int, default=HEALTHY_V7_EPOCH)
    p.add_argument("--out", type=Path, default=HEALTHY_V7_CKPT)
    args = p.parse_args()
    path = seal(
        run_dir=args.run_dir,
        epoch_ckpt=args.epoch_ckpt,
        out=args.out,
        epoch=args.epoch,
    )
    gate = write_gate_stamp(out=path, epoch=args.epoch, run_dir=args.run_dir)
    print(f"wrote {path}")
    print(f"gate {gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
