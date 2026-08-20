#!/usr/bin/env python3
"""Build v8 PDB-Bind Refined@30% cluster holdout manifests (Sprint 10).

Hard wall: any cluster / ID touching Core is purged from train/val.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from science.tokyo_eye.v8.pdbbind_loader import DEFAULT_LP_CSV, load_lp_table, refined_core_frames
from science.tokyo_eye.v8.seq_cluster import (
    assert_no_core_leak,
    cluster_sequences_exact,
    cluster_sequences_mmseqs,
    ids_hitting_core,
    mmseqs_available,
    split_reps_train_val,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v8 PDB-Bind cluster30 splits")
    p.add_argument("--csv", type=Path, default=DEFAULT_LP_CSV)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/v8_pdbbind_refined_cluster30_v1.json"),
    )
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--identity-floor", type=float, default=0.30)
    p.add_argument("--kmer-jaccard-floor", type=float, default=0.45)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_lp_table(args.csv)
    refined, core = refined_core_frames(df)

    core_seqs = {str(i): str(row["seq"]) for i, row in core.iterrows()}
    ref_seqs = {str(i): str(row["seq"]) for i, row in refined.iterrows()}
    core_ids = set(core_seqs)

    # 1) Purge refined IDs hitting Core (exact / mmseqs / k-mer wall)
    purged = ids_hitting_core(
        ref_seqs,
        core_seqs,
        identity_floor=float(args.identity_floor),
        kmer_jaccard_floor=float(args.kmer_jaccard_floor),
    )
    kept_ids = [i for i in refined.index.astype(str) if i not in purged]
    if not kept_ids:
        raise SystemExit("no refined complexes survived Core purge")

    kept_seqs = {i: ref_seqs[i] for i in kept_ids}

    # 2) Cluster kept refined (exact seq; mmseqs if available for finer groups)
    if mmseqs_available():
        member_to_rep = cluster_sequences_mmseqs(
            kept_seqs, min_seq_id=float(args.identity_floor)
        )
        backend = "mmseqs"
    else:
        member_to_rep = cluster_sequences_exact(kept_seqs)
        backend = "exact_seq+kmer_core_purge"

    # Extra safety: if any rep's members include a purged id somehow — already filtered
    reps = sorted(set(member_to_rep.values()))
    train_reps, val_reps = split_reps_train_val(
        reps, val_fraction=float(args.val_fraction), seed=int(args.seed)
    )
    train_rep_set, val_rep_set = set(train_reps), set(val_reps)

    def _rows(rep_set: set[str]) -> list[dict]:
        out = []
        for pid in kept_ids:
            rep = member_to_rep[pid]
            if rep not in rep_set:
                continue
            row = refined.loc[pid]
            out.append(
                {
                    "pdb_id": str(pid).upper(),
                    "chain": "A",
                    "affinity": float(row["value"]),
                    "label_unit": "-log10(K)_M",
                    "kd_ki_raw": str(row.get("kd/ki", "")),
                    "cluster_rep": str(rep).upper(),
                    "category": "refined",
                }
            )
        return out

    train_rows = _rows(train_rep_set)
    val_rows = _rows(val_rep_set)
    core_rows = []
    for pid, row in core.iterrows():
        core_rows.append(
            {
                "pdb_id": str(pid).upper(),
                "chain": "A",
                "affinity": float(row["value"]),
                "label_unit": "-log10(K)_M",
                "kd_ki_raw": str(row.get("kd/ki", "")),
                "cluster_rep": str(pid).upper(),
                "category": "core",
            }
        )

    assert_no_core_leak(
        [r["pdb_id"] for r in train_rows],
        [r["pdb_id"] for r in val_rows],
        [r["pdb_id"] for r in core_rows],
    )
    # Also assert no exact sequence leak
    core_seq_set = {str(s) for s in core_seqs.values()}
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        for r in rows:
            seq = ref_seqs[r["pdb_id"].lower()]
            if seq in core_seq_set:
                raise AssertionError(
                    f"sequence leak in {split_name}: {r['pdb_id']} shares Core sequence"
                )

    manifest = {
        "version": "1.0",
        "description": "TokyoEye-v8 PDB-Bind Refined cluster holdout @30% vs Core",
        "spec": "docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-affinity-regression-design.md",
        "source_csv": str(args.csv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cluster_backend": backend,
        "identity_floor": float(args.identity_floor),
        "kmer_jaccard_floor": float(args.kmer_jaccard_floor),
        "val_fraction": float(args.val_fraction),
        "seed": int(args.seed),
        "n_refined_total": int(len(refined)),
        "n_purged_core_hit": int(len(purged)),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_core_test": len(core_rows),
        "train": train_rows,
        "val": val_rows,
        "core_test": core_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "backend": backend,
                "n_purged": len(purged),
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "n_core_test": len(core_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
