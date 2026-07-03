#!/usr/bin/env python3
"""Apply human review dispositions to TM-align redundancy report → locked Stage A manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]

# Review dispositions (TM-align draft review, 2026-07-03)
TRAIN_TO_HOLDOUT: dict[str, str] = {
    "1SUP:A": "CROSS_FOLD_HIGH_TM vs 4OBE:A (3.40.50.200 sibling); SOD filler yields to GTPase centrality",
    "2ABD:A": "CROSS_FOLD_HIGH_TM hub (9 pairs); acyl-carrier filler mimics multiple fold trains",
}

HOLDOUT_TO_TRAIN: dict[str, str] = {
    "3CON:A": "GTPase cap-2 biological-centrality override: structurally farthest RAS paralog from 4OBE (TM=0.894)",
}

FOLD_CAP_OVERRIDES: dict[str, dict[str, Any]] = {
    "3.40.50.300": {
        "max_train": 2,
        "rule": "biological_centrality_override",
        "rationale": "KRAS/NRAS/HRAS platform evidence base; cap-2 for paralog conformational diversity",
    },
}


def apply_lock(report: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(report)
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []

    struct_by_key = {s["structure_key"]: deepcopy(s) for s in out["structures"]}
    draft_train = {r["structure_key"]: r for r in out["stage_a_draft"]["train"]}
    draft_holdout = {r["structure_key"]: r for r in out["stage_a_draft"]["eval_holdout"]}

    for key, row in draft_train.items():
        base = {**struct_by_key[key], **row}
        if key in TRAIN_TO_HOLDOUT:
            base["role"] = "eval_holdout"
            base["disposition_rule"] = TRAIN_TO_HOLDOUT[key]
            base.setdefault("flags", []).append("REVIEW_DEMOTED_FROM_TRAIN")
            holdout_rows.append(base)
        else:
            base["role"] = "train"
            train_rows.append(base)

    for key, row in draft_holdout.items():
        base = {**struct_by_key[key], **row}
        if key in HOLDOUT_TO_TRAIN:
            base["role"] = "train"
            base["disposition_rule"] = HOLDOUT_TO_TRAIN[key]
            base["enabled"] = True
            flags = [f for f in base.get("flags", []) if f != "WITHIN_FOLD_HIGH_IDENTITY_DROPPED"]
            base["flags"] = flags + ["REVIEW_PROMOTED_TO_TRAIN", "GTPASE_CAP2_OVERRIDE"]
            train_rows.append(base)
        elif key not in TRAIN_TO_HOLDOUT:
            base["role"] = "eval_holdout"
            holdout_rows.append(base)

    train_keys = {r["structure_key"] for r in train_rows}
    violations: list[str] = []
    for p in out["pairs"]:
        if p["decision"] != "CROSS_FOLD_HIGH_TM":
            continue
        if p["a"] in train_keys and p["b"] in train_keys:
            violations.append(f"CROSS_FOLD_HIGH_TM_TRAIN:{p['a']}:{p['b']}:{p['tm_score']}")

    fold_train_counts: dict[str, int] = {}
    for r in train_rows:
        fid = r.get("fold_id") or "UNVERIFIED"
        fold_train_counts[fid] = fold_train_counts.get(fid, 0) + 1
        cap = FOLD_CAP_OVERRIDES.get(fid, {}).get("max_train", out.get("max_train_per_fold_id", 2))
        if fold_train_counts[fid] > cap:
            violations.append(f"FOLD_TRAIN_CAP_EXCEEDED:{fid}")

    locked = {
        "train_count": len(train_rows),
        "eval_holdout_count": len(holdout_rows),
        "fold_count": len({r.get("fold_id") for r in train_rows}),
        "violations": violations,
        "train": [{k: v for k, v in r.items() if k != "flags"} for r in train_rows],
        "eval_holdout": [{k: v for k, v in r.items() if k != "flags"} for r in holdout_rows],
    }

    out["review_dispositions"] = {
        "train_to_holdout": TRAIN_TO_HOLDOUT,
        "holdout_to_train": HOLDOUT_TO_TRAIN,
        "fold_cap_overrides": FOLD_CAP_OVERRIDES,
    }
    out["stage_a_locked"] = locked
    out["status"] = "locked"
    out["locking_artifact"] = True
    out["superseded_by"] = None
    out["pass"] = len(violations) == 0 and locked["train_count"] > 0
    return out


def write_locked_manifest(report: dict[str, Any], out_path: Path) -> None:
    proteins = []
    for row in report["stage_a_locked"]["train"] + report["stage_a_locked"]["eval_holdout"]:
        proteins.append(
            {
                "pdb_id": row["pdb_id"],
                "chain": row["chain"],
                "gene": row.get("gene"),
                "fold_id": row.get("fold_id"),
                "fold_id_tier": row.get("fold_id_tier"),
                "fold_id_source": row.get("fold_id_source"),
                "role": row["role"],
                "disposition_rule": row.get("disposition_rule"),
                "enabled": row["role"] == "train",
                "stage0": row.get("stage0", False),
            }
        )
    payload = {
        "version": "1.0",
        "description": "Stage A locked corpus — TM-align redundancy + review dispositions",
        "source_report": "corpus_redundancy_report.json",
        "structural_metric": report.get("structural_metric"),
        "max_sequence_identity_pct": report.get("sequence_identity_threshold_pct"),
        "max_train_per_fold_id": report.get("max_train_per_fold_id"),
        "proteins": proteins,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock Stage A corpus from TM-align report")
    parser.add_argument(
        "--report-in",
        type=Path,
        default=_REPO / "manifests" / "corpus_redundancy_report_tmalign.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=_REPO / "manifests" / "corpus_redundancy_report.json",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=_REPO / "manifests" / "v6_corpus_stage_a.json",
    )
    args = parser.parse_args()

    report = json.loads(args.report_in.read_text())
    if report.get("structural_metric") == "biotite_ca_proxy":
        raise SystemExit("Refusing to lock a biotite_ca_proxy report — run TM-align first")

    locked_report = apply_lock(report)
    args.report_out.write_text(json.dumps(locked_report, indent=2) + "\n")
    write_locked_manifest(locked_report, args.manifest_out)

    s = locked_report["stage_a_locked"]
    print(f"Locked train: {s['train_count']} | holdout: {s['eval_holdout_count']}")
    print(f"Cross-fold violations: {len(s['violations'])}")
    print(f"Pass: {locked_report['pass']}")
    print(f"Report: {args.report_out}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
