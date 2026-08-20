#!/usr/bin/env python3
"""Preflight gate for manifold-synchronized euc-reach cold train (v66).

Runs Part 0 construction checks on grade_ssot (shared manifold), verifies
frozen site lists, baseline checkpoint, gate stamp, and disk headroom.

Exit 0 + writes TRAIN_READY.json → safe to launch:
  make train-v66-chem-mvp-euc-reach-manifold RUN_ID=chem_mvp_euc_reach_manifold_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

from experiments.training.v66.manifold_ssot import (
    DEFAULT_BASELINE_RUN_ID,
    DEFAULT_TREATMENT_RUN_ID,
    EUCLIDEAN_REACH_SUITE,
    MANIFOLD_LOADER,
    MANIFOLD_RHO_SOURCE,
    load_suite_prot,
)

GATE_STAMP = REPO / "data/gates/p_feature_01_passed.json"
BASELINE_CKPT = REPO / f"checkpoints/v66/runs/{DEFAULT_BASELINE_RUN_ID}/v66_best.pt"
MANIFEST = REPO / "manifests/v6_corpus_stage_a_small_v1.json"
SITE_DIR = REPO / "checkpoints/v66/diagnostics/euclidean_reach/site_lists"
DIAG_OUT = REPO / "checkpoints/v66/diagnostics/euclidean_reach"
MIN_DISK_GB = 12.0


def _check(name: str, ok: bool, detail: str, *, hard: bool = True) -> dict[str, Any]:
    return {"name": name, "pass": ok, "hard": hard, "detail": detail}


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _site_lists_frozen() -> tuple[bool, str]:
    missing = []
    not_frozen = []
    for spec in EUCLIDEAN_REACH_SUITE:
        pid = spec["pdb_id"].lower()
        p = SITE_DIR / f"{pid}_pathway_residues.json"
        if not p.is_file():
            missing.append(str(p))
            continue
        blob = json.loads(p.read_text())
        if not blob.get("frozen"):
            not_frozen.append(str(p))
    if missing:
        return False, f"missing site lists: {missing}"
    if not_frozen:
        return False, f"not frozen: {not_frozen}"
    return True, f"frozen: {[s['pdb_id'] for s in EUCLIDEAN_REACH_SUITE]}"


def _run_part0_grade_ssot() -> tuple[bool, str, Path | None]:
    """Invoke Part0 with grade_ssot loader; return (ok, msg, summary_path)."""
    cmd = [
        sys.executable,
        str(REPO / "experiments/diagnostics/euclidean_reach_part0.py"),
        "--loader",
        "grade_ssot",
    ]
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    summary = DIAG_OUT / "part0" / "summary.json"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return False, f"Part0 exit {proc.returncode}: {tail}", summary
    if not summary.is_file():
        return False, "Part0 finished but summary.json missing", None
    blob = json.loads(summary.read_text())
    status = blob.get("status") or blob.get("overall_status")
    if status and "FAIL" in str(status).upper():
        return False, f"Part0 status={status}", summary
    return True, f"Part0 grade_ssot OK (status={status})", summary


def _manifold_smoke() -> tuple[bool, str, list[dict[str, Any]]]:
    """Load each suite member via shared SSOT; report ρ stats."""
    rows: list[dict[str, Any]] = []
    for spec in EUCLIDEAN_REACH_SUITE:
        pdb_id = spec["pdb_id"]
        chains = spec["chains"]
        prot = load_suite_prot(pdb_id, chains)
        x = prot["data"].x.detach().cpu().numpy()
        rho = x[:, 0]
        const = bool((rho == rho[0]).all()) if rho.size else False
        rows.append(
            {
                "pdb_id": pdb_id,
                "chains": list(chains),
                "N": int(rho.size),
                "rho_min": float(rho.min()) if rho.size else None,
                "rho_max": float(rho.max()) if rho.size else None,
                "rho_mean": float(rho.mean()) if rho.size else None,
                "rho_constant": const,
                "loader": prot.get("loader", MANIFOLD_LOADER),
                "rho_source": prot.get("rho_source", MANIFOLD_RHO_SOURCE),
            }
        )
        if const and pdb_id != "1BE9":  # tiny proteins may look flat — still flag
            if rho.size > 20 and abs(float(rho[0]) - 12.0) < 1e-6:
                return False, f"{pdb_id}: constant ρ≡12 detected (legacy leak)", rows
    return True, "shared manifold loads real wrapping ρ on suite", rows


def run_preflight(
    *,
    run_id: str,
    skip_part0: bool = False,
    min_disk_gb: float = MIN_DISK_GB,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_dir = REPO / "checkpoints/v66/runs" / run_id

    checks.append(
        _check(
            "gate_stamp",
            GATE_STAMP.is_file(),
            str(GATE_STAMP) if GATE_STAMP.is_file() else "missing — run make gate-p-feature-01",
        )
    )
    checks.append(
        _check(
            "baseline_ckpt",
            BASELINE_CKPT.is_file(),
            str(BASELINE_CKPT),
        )
    )
    checks.append(
        _check(
            "corpus_manifest",
            MANIFEST.is_file(),
            str(MANIFEST),
        )
    )
    ok_sites, msg_sites = _site_lists_frozen()
    checks.append(_check("site_lists_frozen", ok_sites, msg_sites))

    free_gb = _disk_free_gb(REPO)
    checks.append(
        _check(
            "disk_headroom",
            free_gb >= min_disk_gb,
            f"{free_gb:.1f} GB free (need ≥{min_disk_gb})",
        )
    )

    if run_dir.exists() and any(run_dir.iterdir()):
        checks.append(
            _check(
                "run_dir_clean",
                False,
                f"{run_dir} exists and is non-empty — use fresh RUN_ID or remove",
                hard=False,
            )
        )
    else:
        checks.append(
            _check("run_dir_clean", True, f"{run_dir} absent or empty (OK for new run)")
        )

    ok_man, msg_man, rho_rows = _manifold_smoke()
    checks.append(_check("manifold_smoke", ok_man, msg_man))

    part0_summary: Path | None = None
    if skip_part0:
        checks.append(
            _check("part0_grade_ssot", True, "skipped (--skip-part0)", hard=False)
        )
    else:
        ok_p0, msg_p0, part0_summary = _run_part0_grade_ssot()
        checks.append(_check("part0_grade_ssot", ok_p0, msg_p0))

    hard_fail = [c for c in checks if c["hard"] and not c["pass"]]
    soft_warn = [c for c in checks if not c["hard"] and not c["pass"]]
    ready = len(hard_fail) == 0

    payload: dict[str, Any] = {
        "schema_version": 1,
        "preflight": "euclidean_reach_manifold_sync",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "manifold_loader": MANIFOLD_LOADER,
        "manifold_rho_source": MANIFOLD_RHO_SOURCE,
        "baseline_run_id": DEFAULT_BASELINE_RUN_ID,
        "treatment_run_id": run_id,
        "invalidated_prior_run_id": "chem_mvp_euc_reach_v1",
        "checks": checks,
        "hard_failures": [c["name"] for c in hard_fail],
        "warnings": [c["name"] for c in soft_warn],
        "manifold_smoke": rho_rows,
        "part0_summary": str(part0_summary) if part0_summary else None,
        "launch_command": (
            f"make train-v66-chem-mvp-euc-reach-manifold RUN_ID={run_id}"
        ),
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-id",
        default=DEFAULT_TREATMENT_RUN_ID,
        help=f"Treatment run id (default: {DEFAULT_TREATMENT_RUN_ID})",
    )
    ap.add_argument("--skip-part0", action="store_true")
    ap.add_argument("--min-disk-gb", type=float, default=MIN_DISK_GB)
    ap.add_argument(
        "--out",
        type=Path,
        default=DIAG_OUT / "preflight_manifold_sync.json",
    )
    args = ap.parse_args()

    payload = run_preflight(
        run_id=args.run_id,
        skip_part0=args.skip_part0,
        min_disk_gb=args.min_disk_gb,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    run_dir = REPO / "checkpoints/v66/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ready_path = run_dir / "TRAIN_READY.json"
    ready_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    if not payload["ready"]:
        print("\nPREFLIGHT FAIL — do not train.", file=sys.stderr)
        sys.exit(1)
    print(f"\nPREFLIGHT PASS — training ready. Wrote {args.out} and {ready_path}")
    print(f"Launch: {payload['launch_command']}")


if __name__ == "__main__":
    main()
