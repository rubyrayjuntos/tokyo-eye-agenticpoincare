"""Grade v7 B′ uncertainty-heads rematch-0 against prereg bars."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_METRICS = (
    REPO
    / "checkpoints"
    / "v7"
    / "runs"
    / "tokyo_eye_v7_bprime_uncertainty_heads_v1"
    / "metrics.json"
)
DEFAULT_OUT = REPO / "data" / "gates" / "tokyo_eye_v7_bprime_uncertainty_heads_closeout.json"
PREREG = REPO / "data" / "gates" / "tokyo_eye_v7_bprime_uncertainty_heads_prereg.json"


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(float(x))


def grade(metrics_path: Path) -> dict:
    rows = json.loads(metrics_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"empty metrics: {metrics_path}")
    unc = [
        e
        for e in rows
        if "uncertainty heads" in str(e.get("phase_name") or "").lower()
    ]
    rematch1 = [
        e for e in unc if "rematch-1" in str(e.get("phase_name") or "").lower()
    ]
    scored = rematch1 if rematch1 else (unc if unc else rows[-24:])
    last = scored[-1]
    is_rematch1 = "rematch-1" in str(last.get("phase_name") or "").lower()
    health = dict(last.get("health") or {})
    losses = dict(last.get("losses") or {})

    ale = health.get("aleatoric_std_mean")
    epi = health.get("epistemic_std_mean")
    nu_cv = health.get("evidence_nu_cv_mean")
    r_ea = health.get("probe_r_epi_ale")
    tau_flag = health.get("uncertainty_tau_ale_elevated")
    tau_lift = health.get("node_aleatoric_tau_lift_relative")
    disc_r = health.get("disc_r_mean")

    disc_vals = [
        float((e.get("health") or {}).get("disc_r_mean"))
        for e in scored
        if _finite((e.get("health") or {}).get("disc_r_mean"))
    ]
    disc_min = min(disc_vals) if disc_vals else None
    disc_final = float(disc_r) if _finite(disc_r) else None

    bars = {
        "aleatoric_std_mean": {
            "threshold": 0.02,
            "observed": ale,
            "pass": _finite(ale) and float(ale) >= 0.02,
        },
        "epistemic_std_mean": {
            "threshold": 0.01,
            "observed": epi,
            "pass": _finite(epi) and float(epi) >= 0.01,
        },
        "evidence_nu_cv_mean": {
            "threshold": 0.02,
            "observed": nu_cv,
            "pass": _finite(nu_cv) and float(nu_cv) >= 0.02,
        },
        "probe_r_epi_ale_abs": {
            "threshold": 0.70,
            "observed": r_ea,
            "pass": _finite(r_ea) and abs(float(r_ea)) <= 0.70,
        },
        "tau_ale": {
            "elevated": tau_flag,
            "relative_lift": tau_lift,
            "pass": (
                (tau_flag is not None and float(tau_flag) >= 1.0)
                or (_finite(tau_lift) and float(tau_lift) >= 0.20)
            ),
        },
        "disc_r_mean_hold": {
            "threshold": 0.25,
            "observed_final": disc_final,
            "observed_min": disc_min,
            "pass": _finite(disc_final) and float(disc_final) >= 0.25,
        },
        "no_nan_loss": {
            "pass": all(
                (not isinstance(v, float))
                or math.isfinite(v)
                or k.endswith("_index")  # telemetry monitors may be nan when undefined
                for k, v in losses.items()
            )
        },
    }
    passed = all(bool(b.get("pass")) for b in bars.values())
    flat_disc_held = (
        (not bars["aleatoric_std_mean"]["pass"] or not bars["epistemic_std_mean"]["pass"])
        and bars["disc_r_mean_hold"]["pass"]
    )
    # Only rematch-0 authorizes rematch-1; rematch-1 exhausts the pre-auth budget.
    rematch_authorized = bool(flat_disc_held and not passed and not is_rematch1)
    if passed:
        note = "Uncertainty heads closeout Pass. No biology / promote claim."
    elif is_rematch1:
        note = (
            "Fail after rematch-1 (pre-auth exhausted). Disc held; ale/epi improved "
            "but bars not cleared. Restore sealed health for biology deferral; "
            "do not chase further without new prereg."
        )
    elif flat_disc_held:
        note = (
            "Fail rematch-0: ale/epi flat, disc held — pre-authorized rematch-1 "
            "(raise anticollapse/decorrelation; trunk stays frozen)."
        )
    else:
        note = (
            "Fail — disc broke or other bars; restore sealed health; no biology chase."
        )
    return {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_bprime_uncertainty_heads_closeout",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "prereg": str(PREREG.relative_to(REPO)) if PREREG.exists() else None,
        "run_id": metrics_path.parent.name,
        "metrics": str(metrics_path),
        "final_global_epoch": last.get("global_epoch"),
        "phase_name": last.get("phase_name"),
        "n_scored_epochs": len(scored),
        "bars": bars,
        "rematch_authorized": rematch_authorized,
        "artifacts": {
            "phase_ckpt": str(metrics_path.parent / "v7_phase4_12prot.pt"),
            "v7_best": str(metrics_path.parent / "v7_best.pt"),
            "metrics": str(metrics_path),
            "healthy_sealed": "checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt",
        },
        "note": note,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if not args.metrics.is_file():
        raise SystemExit(f"missing metrics: {args.metrics}")
    report = grade(args.metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
