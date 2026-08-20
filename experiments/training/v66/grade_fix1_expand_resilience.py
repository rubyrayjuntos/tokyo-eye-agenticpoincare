"""Grade Fix-1 expand continue resilience gates from metrics.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _last_health(
    metrics_path: Path,
    *,
    continue_epochs: int | None,
    run_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = json.loads(metrics_path.read_text())
    if not rows:
        raise ValueError(f"empty metrics: {metrics_path}")
    last = rows[-1]
    # Prefer epoch snapshot count when present — early-stop runs append less
    # than --continue-epochs, and naive [-N:] would leak parent-run history.
    n_snap = 0
    if run_dir is not None:
        epochs_dir = run_dir / "epochs"
        if epochs_dir.is_dir():
            n_snap = len(list(epochs_dir.glob("epoch_*.pt")))
    window = None
    if n_snap > 0:
        window = n_snap
    elif continue_epochs is not None and continue_epochs > 0:
        window = continue_epochs
    # Continue resumes append onto prior history in metrics.json. Grade the
    # first epoch of *this* continue window, not cold ge1 origin occupancy.
    if window is not None and window > 0 and len(rows) >= window:
        first = rows[-window]
    else:
        # Auto: first mature row in the trailing contiguous block ending at last.
        first = rows[0]
        for i in range(len(rows) - 1, -1, -1):
            d = (rows[i].get("health") or {}).get("disc_r_mean")
            if d is None or float(d) < 0.20:
                if i + 1 < len(rows):
                    first = rows[i + 1]
                break
            first = rows[i]
    return first, last


def grade(
    *,
    run_dir: Path,
    disc_r_min: float = 0.25,
    sigma_min: float = 0.80,
    r_proj_min: float = 0.95,
    r_tau_min: float = 0.45,
    max_expert_hard_share: float = 0.50,
    continue_epochs: int | None = 15,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    first, last = _last_health(
        metrics_path, continue_epochs=continue_epochs, run_dir=run_dir
    )
    fh, lh = first.get("health") or {}, last.get("health") or {}

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, value: Any, threshold: Any, note: str = "") -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(ok),
                "value": value,
                "threshold": threshold,
                "note": note,
            }
        )

    # First-epoch of continue window (mature-band)
    d0 = fh.get("disc_r_mean")
    add(
        "ep1_disc_r_mean",
        d0 is not None and float(d0) >= disc_r_min,
        d0,
        f">= {disc_r_min}",
        f"continue window start ge={first.get('global_epoch')} (not cold origin)",
    )
    # Final epoch
    dN = lh.get("disc_r_mean")
    add(
        "final_disc_r_mean",
        dN is not None and float(dN) >= disc_r_min,
        dN,
        f">= {disc_r_min}",
    )
    sN = lh.get("disc_sigma2_sigma1_mean")
    add(
        "final_sigma2_sigma1",
        sN is not None and float(sN) >= sigma_min,
        sN,
        f">= {sigma_min}",
    )
    rp = lh.get("probe_r_proj_depth")
    add(
        "final_probe_r_proj_depth",
        rp is not None and float(rp) >= r_proj_min,
        rp,
        f">= {r_proj_min}",
    )
    rt = lh.get("probe_r_depth_tau")
    add(
        "final_probe_r_depth_tau",
        rt is not None and float(rt) >= r_tau_min,
        rt,
        f">= {r_tau_min}",
    )

    # Expert monopoly from inference_routing if present
    ir = last.get("inference_routing") or {}
    hard = ir.get("hard_frac") or ir.get("expert_load") or []
    max_hard = max(hard) if hard else None
    add(
        "expert_hard_monopoly",
        max_hard is None or float(max_hard) < max_expert_hard_share,
        max_hard,
        f"< {max_expert_hard_share}",
    )

    passed = all(c["passed"] for c in checks)
    report = {
        "gate": "FIX1_EXPAND_RESILIENCE",
        "passed": passed,
        "run_dir": str(run_dir),
        "global_epoch_first": first.get("global_epoch"),
        "global_epoch_last": last.get("global_epoch"),
        "checks": checks,
        "graded_at": datetime.now(timezone.utc).isoformat(),
    }
    out = run_dir / "resilience_gate.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--disc-r-min", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.80)
    p.add_argument("--r-proj-min", type=float, default=0.95)
    p.add_argument("--r-tau-min", type=float, default=0.45)
    p.add_argument(
        "--profile",
        choices=("expand", "transfer"),
        default="expand",
        help="expand: P1/P2 volume bars; transfer: P3 pathway bars (σ≥0.75, probe≥0.85)",
    )
    p.add_argument(
        "--continue-epochs",
        type=int,
        default=15,
        help="Grade ep1 as the first of the last N epochs (resume window)",
    )
    args = p.parse_args()
    sigma_min = args.sigma_min
    r_proj_min = args.r_proj_min
    if args.profile == "transfer":
        if args.sigma_min == 0.80:
            sigma_min = 0.75
        if args.r_proj_min == 0.95:
            r_proj_min = 0.85
    report = grade(
        run_dir=args.run_dir,
        disc_r_min=args.disc_r_min,
        sigma_min=sigma_min,
        r_proj_min=r_proj_min,
        r_tau_min=args.r_tau_min,
        continue_epochs=args.continue_epochs,
    )
    report["profile"] = args.profile
    (args.run_dir / "resilience_gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
