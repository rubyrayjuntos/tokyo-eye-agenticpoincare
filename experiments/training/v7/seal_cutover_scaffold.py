"""Seal a cutover-scaffold TokyoEye checkpoint for contract production.

This is **not** a biology champion — init weights that load cleanly for ingest /
health until a trained ``TokyoEye-v7`` champion is promoted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.training.v7 import LINEAGE_ID, PRODUCTION_MODULE, V7_HYP_SPACE_NAME
from science.tokyo_eye.TokyoEye import TokyoEye

DEFAULT_OUT = Path("checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt")
STAMP = Path("data/gates/tokyo_eye_v7_cutover_scaffold.json")


def seal_scaffold(*, out: Path = DEFAULT_OUT, node_dim: int = 4, hidden: int = 128) -> dict:
    model = TokyoEye(
        node_dim=node_dim,
        hidden=hidden,
        hyp_mp_primary=True,
        se3_aux=False,
    )
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "class": "TokyoEye",
            "hyp_mp_primary": True,
            "se3_aux": False,
            "hidden": hidden,
            "node_dim": node_dim,
        },
        "training_config": {
            "gnn_lineage": LINEAGE_ID,
            "model_version": "TokyoEye-v7",
            "hyp_mp_primary": True,
            "se3_aux": False,
        },
        "curvature": float(model.curvature.detach().cpu().item()),
        "meta": {
            "kind": "cutover_scaffold",
            "biology_graded": False,
            "production_module": PRODUCTION_MODULE,
            "space_name": V7_HYP_SPACE_NAME,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    torch.save(payload, out)
    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_cutover_scaffold",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(out),
        "biology_graded": False,
        "model_version": "TokyoEye-v7",
        "curvature_init": payload["curvature"],
        "note": "Init scaffold for contract cutover; replace via promote when champion trains.",
    }
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    stamp = seal_scaffold(out=args.out)
    print(json.dumps(stamp, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
