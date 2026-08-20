"""Seal TokyoEye B′ surgical warmstart from Fix-1 sparsity champion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.training.v66.healthy_fix1 import FIX1_SPARSITY_CHAMPION_CKPT
from experiments.training.v7 import V7_CHECKPOINT_ROOT
from science.tokyo_eye.surgical_warmstart import surgical_warmstart_tokyo_eye

DEFAULT_OUT = Path("checkpoints/v7/tokyo_eye_v7_bprime_warmstart.pt")
STAMP = Path("data/gates/tokyo_eye_v7_bprime_warmstart_sealed.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor", type=Path, default=FIX1_SPARSITY_CHAMPION_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    if not args.donor.is_file():
        raise SystemExit(f"missing donor: {args.donor}")

    report = surgical_warmstart_tokyo_eye(args.donor, out_path=args.out)
    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_bprime_warmstart_sealed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "donor": report.donor,
        "out": report.out,
        "transferred_keys": report.transferred_keys,
        "denied_keys": report.denied_keys,
        "missing_after_load": report.missing_after_load,
        "unexpected_after_load": report.unexpected_after_load,
        "hyp_mp_primary": report.hyp_mp_primary,
        "se3_aux": report.se3_aux,
        "denied_prefixes": list(report.denied_prefixes),
        "spec": "docs/specs/tokyo-eye-v7/bprime-health-warmstart-prereg.md",
        "checkpoint_root": str(V7_CHECKPOINT_ROOT),
    }
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n")
    print(json.dumps(stamp, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {STAMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
