#!/usr/bin/env python3
"""Validation gate: compare SSOT-on vs learned (SSOT-off) inference policy for a checkpoint.

Does not run full GNN by default — resolves policy and prints the decision.
Pass --forward only when a science container + DB graph are available.

Usage:
  PYTHONPATH=. python -m experiments.diagnostics.validate_learned_ssot_gate \\
    --checkpoint checkpoints/v65/runs/master_cold_v1/v65_best.pt

  # Force A/B policy print for production lever_a:
  PYTHONPATH=. python -m experiments.diagnostics.validate_learned_ssot_gate \\
    --checkpoint checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = parser.parse_args()

    from science.dtie.common.structural_disc_policy import (
        read_checkpoint_structural_disc_frozen,
        resolve_structural_disc_frozen,
    )

    ckpt = args.checkpoint
    if not ckpt.is_file():
        print(f"FAIL: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    from_ckpt = read_checkpoint_structural_disc_frozen(ckpt)
    frozen_default, reason_default = resolve_structural_disc_frozen(ckpt)
    frozen_on, _ = resolve_structural_disc_frozen(
        ckpt, job_params={"structural_disc_frozen": True}
    )
    frozen_off, _ = resolve_structural_disc_frozen(
        ckpt, job_params={"structural_disc_frozen": False}
    )

    report = {
        "checkpoint": str(ckpt),
        "training_config_structural_disc_frozen": from_ckpt,
        "resolved_default": {"frozen": frozen_default, "reason": reason_default},
        "override_on": frozen_on,
        "override_off": frozen_off,
        "gate": {
            "learned_path_eligible": from_ckpt is False,
            "slim_or_ssot_checkpoint": from_ckpt is True,
            "unknown_uses_legacy_ssot": from_ckpt is None and frozen_default is True,
            "note": (
                "Production flip: only promote learned-path ingest after Stage A / 4OBE "
                "forward compare (job_params structural_disc_frozen true vs false)."
            ),
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
