#!/usr/bin/env python3
"""Orthogonality gate for the first dehydron-barcode scalar arm.

This is a pure-input diagnostic: it reads the saved Stage A-12 raw barcode
lists, recomputes only residue/dehydron geometry (not GUDHI), and compares the
candidate H1 scalar block against ``[rho, tau, ss]`` and against itself.

The gate uses the representation that would enter corpus z-normalization:
``log1p`` for persistence mass/count features, raw max/fraction. Z-scoring
does not alter Pearson or Spearman correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from experiments.training.v6 import _data
from experiments.training.v6.corpus import iter_corpus_entries
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    LONG_LIVED_PERSISTENCE_ANGSTROM,
    _load_chain_structure_atoms,
    extract_dehydron_midpoints,
)
from science.dtie.common.residue_features import FeatureMode, build_from_pdb_chain

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json"
_DEFAULT_RAW_BARS = (
    _REPO_ROOT
    / "checkpoints"
    / "v65"
    / "diagnostics"
    / "dehydron_bar_length_v1"
    / "bars_raw"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "checkpoints"
    / "v65"
    / "diagnostics"
    / "dehydron_scalar_orthogonality_v1"
)

DEFAULT_REDUNDANCY_THRESHOLD = 0.70
CANDIDATE_NAMES = (
    "total_persistence_h1",
    "max_persistence_h1",
    "num_h1_bars",
    "fraction_long_lived_h1",
    "n_dehydrons_touching",
)
BASELINE_NAMES = ("rho", "tau", "ss")


def _bar_value(bar: Any, key: str) -> Any:
    return bar[key] if isinstance(bar, Mapping) else getattr(bar, key)


def build_candidate_matrix(
    bars: Sequence[Any],
    touching_counts: np.ndarray,
    *,
    long_lived_threshold: float,
) -> dict[str, np.ndarray]:
    """Build raw H1-only candidates with globals broadcast to touched residues."""
    touching = np.asarray(touching_counts, dtype=np.float64).reshape(-1)
    touched = touching > 0
    h1 = [
        float(_bar_value(bar, "persistence"))
        for bar in bars
        if int(_bar_value(bar, "dim")) == 1
    ]
    total = float(sum(h1))
    maximum = float(max(h1)) if h1 else 0.0
    count = float(len(h1))
    fraction = (
        float(sum(p >= long_lived_threshold for p in h1) / len(h1))
        if h1
        else 0.0
    )

    def broadcast(value: float) -> np.ndarray:
        out = np.zeros(touching.shape[0], dtype=np.float64)
        out[touched] = value
        return out

    return {
        "total_persistence_h1": broadcast(total),
        "max_persistence_h1": broadcast(maximum),
        "num_h1_bars": broadcast(count),
        "fraction_long_lived_h1": broadcast(fraction),
        "n_dehydrons_touching": touching.copy(),
    }


def model_pre_z_matrix(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply fixed pre-z transforms matching the intended model representation."""
    return {
        "total_persistence_h1": np.log1p(raw["total_persistence_h1"]),
        "max_persistence_h1": np.asarray(
            raw["max_persistence_h1"], dtype=np.float64
        ),
        "num_h1_bars": np.log1p(raw["num_h1_bars"]),
        "fraction_long_lived_h1": np.asarray(
            raw["fraction_long_lived_h1"], dtype=np.float64
        ),
        "n_dehydrons_touching": np.log1p(raw["n_dehydrons_touching"]),
    }


def _correlation_pair(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    n = int(x_arr.size)
    if n < 3:
        return {
            "n": n,
            "pearson": None,
            "spearman": None,
            "reason": "n_lt_3",
        }
    if np.ptp(x_arr) == 0.0 or np.ptp(y_arr) == 0.0:
        return {
            "n": n,
            "pearson": None,
            "spearman": None,
            "reason": "constant",
        }
    return {
        "n": n,
        "pearson": float(pearsonr(x_arr, y_arr).statistic),
        "spearman": float(spearmanr(x_arr, y_arr).statistic),
        "reason": None,
    }


def _ss_key(value: float) -> str:
    return f"{float(value):g}"


def correlation_report(
    x: np.ndarray,
    y: np.ndarray,
    ss: np.ndarray,
) -> dict[str, Any]:
    """Marginal and within-secondary-structure correlation report."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    ss_arr = np.asarray(ss, dtype=np.float64).reshape(-1)
    if not (x_arr.shape == y_arr.shape == ss_arr.shape):
        raise ValueError(
            f"shape mismatch x={x_arr.shape} y={y_arr.shape} ss={ss_arr.shape}"
        )
    within = {}
    for value in sorted(float(v) for v in np.unique(ss_arr[np.isfinite(ss_arr)])):
        mask = ss_arr == value
        within[_ss_key(value)] = _correlation_pair(x_arr[mask], y_arr[mask])
    return {
        "marginal": _correlation_pair(x_arr, y_arr),
        "within_ss": within,
    }


def _walk_correlations(node: Any, prefix: str = ""):
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in ("pearson", "spearman") and isinstance(value, (float, int)):
            if math.isfinite(float(value)):
                yield path, float(value)
        elif isinstance(value, Mapping):
            yield from _walk_correlations(value, path)


def redundancy_verdict(
    reports: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> dict[str, Any]:
    """Fail when any valid Pearson/Spearman magnitude reaches the threshold."""
    values = list(_walk_correlations(reports))
    if not values:
        return {
            "passes": False,
            "threshold": threshold,
            "max_abs_correlation": None,
            "worst_path": None,
            "reason": "no_scorable_correlations",
        }
    worst_path, worst_value = max(values, key=lambda item: abs(item[1]))
    max_abs = abs(worst_value)
    return {
        "passes": max_abs < threshold,
        "threshold": threshold,
        "max_abs_correlation": max_abs,
        "signed_correlation": worst_value,
        "worst_path": worst_path,
        "reason": None,
    }


def payload_baseline_gate_reports(reports: Mapping[str, Any]) -> dict[str, Any]:
    """Gate payload values, retaining all-residue support alignment as telemetry.

    Barcode globals are undefined off their dehydron-touching support and are
    zero-filled with a missing mask there. Correlation introduced solely by
    that support is expected; payload orthogonality is conditional on valid
    rows, plus the independent structure-level check.
    """
    gated: dict[str, Any] = {}
    for name, report in reports.items():
        if name == "structure_level":
            gated[name] = report
        elif isinstance(report, Mapping) and "touching_only" in report:
            gated[name] = {"touching_only": report["touching_only"]}
    return gated


def _load_raw_bars(raw_dir: Path, pdb_id: str, chain: str) -> dict[str, Any]:
    path = raw_dir / f"{pdb_id.upper()}_{chain}_bars.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing raw bars {path}; run make diagnose-dehydron-bar-length"
        )
    payload = json.loads(path.read_text())
    payload["_path"] = str(path)
    return payload


def align_touching_counts(
    midpoints: Sequence[Any],
    residue_index_map: Mapping[tuple[str, int], int],
    feature_residue_indices: Sequence[int],
) -> np.ndarray:
    """Align midpoint endpoint counts onto filtered SSOT feature rows by resseq."""
    index_to_resseq = {
        int(midpoint_index): int(resseq)
        for (_chain, resseq), midpoint_index in residue_index_map.items()
    }
    by_resseq: dict[int, float] = {}
    for midpoint in midpoints:
        for midpoint_index in (int(midpoint.donor_idx), int(midpoint.acceptor_idx)):
            resseq = index_to_resseq[midpoint_index]
            by_resseq[resseq] = by_resseq.get(resseq, 0.0) + 1.0
    return np.asarray(
        [by_resseq.get(int(resseq), 0.0) for resseq in feature_residue_indices],
        dtype=np.float64,
    )


def _touching_counts(
    pdb_path: Path,
    chain: str,
    feature_residue_indices: Sequence[int],
) -> tuple[np.ndarray, int]:
    atoms, residue_index_map = _load_chain_structure_atoms(pdb_path, chain)
    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)
    counts = align_touching_counts(
        midpoints,
        residue_index_map,
        feature_residue_indices,
    )
    return counts, len(midpoints)


def _scope_report(
    x: np.ndarray,
    y: np.ndarray,
    ss: np.ndarray,
    touched: np.ndarray,
) -> dict[str, Any]:
    touched_mask = np.asarray(touched, dtype=bool)
    return {
        "all_residues": correlation_report(x, y, ss),
        "touching_only": correlation_report(
            x[touched_mask], y[touched_mask], ss[touched_mask]
        ),
    }


def _structure_level_report(
    candidate_rows: list[dict[str, float]],
    baseline_rows: list[dict[str, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES[:-1]:
        x = np.asarray([row[candidate] for row in candidate_rows], dtype=np.float64)
        result[candidate] = {}
        for baseline in ("mean_rho", "tau_fraction", "ss_mean"):
            y = np.asarray(
                [row[baseline] for row in baseline_rows], dtype=np.float64
            )
            result[candidate][baseline] = _correlation_pair(x, y)
    return result


def _greedy_independent_subset(
    candidate_reports: Mapping[str, Any],
    peer_reports: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Deterministic first-pass subset in design priority order."""
    selected: list[str] = []
    rejected: dict[str, str] = {}
    for candidate in CANDIDATE_NAMES:
        baseline_verdict = redundancy_verdict(
            candidate_reports[candidate], threshold=threshold
        )
        if not baseline_verdict["passes"]:
            rejected[candidate] = (
                f"baseline redundancy at {baseline_verdict['worst_path']}="
                f"{baseline_verdict['max_abs_correlation']}"
            )
            continue
        conflict = None
        for prior in selected:
            pair_key = "::".join(sorted((candidate, prior)))
            verdict = redundancy_verdict(peer_reports[pair_key], threshold=threshold)
            if not verdict["passes"]:
                conflict = (
                    f"peer redundancy with {prior} at {verdict['worst_path']}="
                    f"{verdict['max_abs_correlation']}"
                )
                break
        if conflict:
            rejected[candidate] = conflict
        else:
            selected.append(candidate)
    return {
        "order": list(CANDIDATE_NAMES),
        "selected": selected,
        "rejected": rejected,
        "note": (
            "Mechanical screen only; inspect report before locking the shipping subset."
        ),
    }


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dehydron scalar orthogonality diagnostic",
        "",
        f"- Structures: **{report['succeeded']}** / {report['attempted']}",
        f"- Residues: **{report['n_residues']}** "
        f"(touching: {report['n_touching_residues']})",
        f"- Long-lived H1 threshold: **{report['long_lived_threshold_angstrom']} Å**",
        f"- Redundancy cut: **|r| / |ρₛ| ≥ {report['redundancy_threshold']}**",
        f"- Barcode feature version: `{report['barcode_feature_version']}`",
        "",
        "## Mechanical screen",
        "",
        "| Candidate | Baseline pass | Worst baseline correlation |",
        "|-----------|---------------|----------------------------|",
    ]
    for name in CANDIDATE_NAMES:
        verdict = report["candidate_baseline_verdicts"][name]
        lines.append(
            f"| `{name}` | **{verdict['passes']}** | "
            f"{verdict['worst_path']} = {verdict['max_abs_correlation']} |"
        )
    subset = report["greedy_independent_subset"]
    lines.extend(
        [
            "",
            "## Greedy independent subset (inspect; do not auto-promote)",
            "",
            f"- Selected: {', '.join(f'`{v}`' for v in subset['selected']) or 'none'}",
        ]
    )
    for name, reason in subset["rejected"].items():
        lines.append(f"- Rejected `{name}`: {reason}")
    lines.extend(
        [
            "",
            "## Interpretation guards",
            "",
            "- Correlations are reported for all residues and touching-only residues.",
            "- **Gate scope:** touching-only payload values + structure-level checks. "
            "All-residue correlation is support-mask telemetry: off-support rows are "
            "zero-filled and explicitly missing, so their rho/tau alignment is expected.",
            "- Every scope includes marginal and within-SS-class results.",
            "- Structure-level correlations test whether broadcast globals merely encode "
            "cross-protein rho/tau/SS composition.",
            "- Constant vectors are `unscorable`, never treated as zero correlation.",
            "- Full numeric detail is in `summary.json`; raw residue rows are in "
            "`residue_rows.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    raw_bars_dir = Path(args.raw_bars_dir)
    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = iter_corpus_entries(manifest, max_proteins=args.max_proteins)
    all_baseline = {name: [] for name in BASELINE_NAMES}
    all_candidates = {name: [] for name in CANDIDATE_NAMES}
    all_raw_candidates = {name: [] for name in CANDIDATE_NAMES}
    all_ss: list[np.ndarray] = []
    all_touched: list[np.ndarray] = []
    residue_rows: list[dict[str, Any]] = []
    structure_candidates: list[dict[str, float]] = []
    structure_baselines: list[dict[str, float]] = []
    structures: list[dict[str, Any]] = []
    failures: list[str] = []

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        key = f"{pdb_id}:{chain}"
        try:
            raw_bars = _load_raw_bars(raw_bars_dir, pdb_id, chain)
            pdb_path = _data._download_pdb(pdb_id, pdb_dir)
            features = build_from_pdb_chain(pdb_path, chain, mode=FeatureMode.MASTER)
            touching, n_midpoints = _touching_counts(
                pdb_path,
                chain,
                [feature.residue_index for feature in features],
            )
            if int(raw_bars["n_midpoints"]) != n_midpoints:
                raise ValueError(
                    f"raw/current midpoint mismatch {raw_bars['n_midpoints']} != "
                    f"{n_midpoints}; raw bar artifact is stale"
                )

            baseline = {
                "rho": np.asarray([f.rho for f in features], dtype=np.float64),
                "tau": np.asarray([f.tau_flag for f in features], dtype=np.float64),
                "ss": np.asarray([f.ss_type for f in features], dtype=np.float64),
            }
            raw_candidate = build_candidate_matrix(
                raw_bars["bars"],
                touching,
                long_lived_threshold=args.long_lived_threshold,
            )
            candidate = model_pre_z_matrix(raw_candidate)
            touched = touching > 0

            for name in BASELINE_NAMES:
                all_baseline[name].append(baseline[name])
            for name in CANDIDATE_NAMES:
                all_candidates[name].append(candidate[name])
                all_raw_candidates[name].append(raw_candidate[name])
            all_ss.append(baseline["ss"])
            all_touched.append(touched)

            h1 = [
                float(bar["persistence"])
                for bar in raw_bars["bars"]
                if int(bar["dim"]) == 1
            ]
            structure_candidates.append(
                {
                    "total_persistence_h1": math.log1p(sum(h1)),
                    "max_persistence_h1": max(h1) if h1 else 0.0,
                    "num_h1_bars": math.log1p(len(h1)),
                    "fraction_long_lived_h1": (
                        sum(p >= args.long_lived_threshold for p in h1) / len(h1)
                        if h1
                        else 0.0
                    ),
                }
            )
            structure_baselines.append(
                {
                    "mean_rho": float(baseline["rho"].mean()),
                    "tau_fraction": float(baseline["tau"].mean()),
                    "ss_mean": float(baseline["ss"].mean()),
                }
            )

            for idx, feature in enumerate(features):
                row: dict[str, Any] = {
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "residue_index": feature.residue_index,
                    "rho": baseline["rho"][idx],
                    "tau": baseline["tau"][idx],
                    "ss": baseline["ss"][idx],
                    "touching": int(touched[idx]),
                }
                for name in CANDIDATE_NAMES:
                    row[f"raw_{name}"] = raw_candidate[name][idx]
                    row[name] = candidate[name][idx]
                residue_rows.append(row)
            structures.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "n_residues": len(features),
                    "n_touching": int(touched.sum()),
                    "n_midpoints": n_midpoints,
                    "n_h1_bars": len(h1),
                    "raw_bars_path": raw_bars["_path"],
                }
            )
            logger.info(
                "OK %s residues=%d touching=%d midpoints=%d H1=%d",
                key,
                len(features),
                touched.sum(),
                n_midpoints,
                len(h1),
            )
        except Exception as exc:
            failures.append(f"{key}: {exc}")
            logger.exception("FAIL %s", key)
            if args.fail_fast:
                break

    if not structures:
        return 1

    baseline_pool = {
        name: np.concatenate(parts) for name, parts in all_baseline.items()
    }
    candidate_pool = {
        name: np.concatenate(parts) for name, parts in all_candidates.items()
    }
    ss_pool = np.concatenate(all_ss)
    touched_pool = np.concatenate(all_touched)

    candidate_baseline: dict[str, Any] = {}
    candidate_payload_gate: dict[str, Any] = {}
    candidate_verdicts: dict[str, Any] = {}
    for candidate_name in CANDIDATE_NAMES:
        reports: dict[str, Any] = {}
        for baseline_name in BASELINE_NAMES:
            reports[baseline_name] = _scope_report(
                candidate_pool[candidate_name],
                baseline_pool[baseline_name],
                ss_pool,
                touched_pool,
            )
        candidate_baseline[candidate_name] = reports
        candidate_payload_gate[candidate_name] = payload_baseline_gate_reports(
            reports
        )
        candidate_verdicts[candidate_name] = redundancy_verdict(
            candidate_payload_gate[candidate_name],
            threshold=args.redundancy_threshold,
        )

    peer_reports: dict[str, Any] = {}
    peer_payload_gate: dict[str, Any] = {}
    for i, left in enumerate(CANDIDATE_NAMES):
        for right in CANDIDATE_NAMES[i + 1 :]:
            key = "::".join(sorted((left, right)))
            peer_reports[key] = _scope_report(
                candidate_pool[left],
                candidate_pool[right],
                ss_pool,
                touched_pool,
            )
            peer_payload_gate[key] = {
                "touching_only": peer_reports[key]["touching_only"]
            }

    structure_level = _structure_level_report(
        structure_candidates, structure_baselines
    )
    for name in CANDIDATE_NAMES[:-1]:
        candidate_baseline[name]["structure_level"] = structure_level[name]
        candidate_payload_gate[name] = payload_baseline_gate_reports(
            candidate_baseline[name]
        )
        candidate_verdicts[name] = redundancy_verdict(
            candidate_payload_gate[name],
            threshold=args.redundancy_threshold,
        )

    subset = _greedy_independent_subset(
        candidate_payload_gate,
        peer_payload_gate,
        threshold=args.redundancy_threshold,
    )

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    rows_path = out_dir / "residue_rows.csv"
    report = {
        "manifest": str(manifest),
        "raw_bars_dir": str(raw_bars_dir),
        "attempted": len(entries),
        "succeeded": len(structures),
        "failed": len(failures),
        "failures": failures,
        "n_residues": int(ss_pool.size),
        "n_touching_residues": int(touched_pool.sum()),
        "long_lived_threshold_angstrom": args.long_lived_threshold,
        "redundancy_threshold": args.redundancy_threshold,
        "barcode_feature_version": BARCODE_FEATURE_VERSION,
        "candidate_order": list(CANDIDATE_NAMES),
        "pre_z_transforms": {
            "total_persistence_h1": "log1p",
            "max_persistence_h1": "identity",
            "num_h1_bars": "log1p",
            "fraction_long_lived_h1": "identity",
            "n_dehydrons_touching": "log1p",
        },
        "candidate_vs_baseline": candidate_baseline,
        "candidate_payload_gate_reports": candidate_payload_gate,
        "candidate_baseline_verdicts": candidate_verdicts,
        "candidate_vs_candidate": peer_reports,
        "candidate_peer_payload_gate_reports": peer_payload_gate,
        "structure_level": structure_level,
        "greedy_independent_subset": subset,
        "structures": structures,
        "artifacts": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "residue_rows_csv": str(rows_path),
        },
    }
    _write_rows_csv(residue_rows, rows_path)
    summary_path.write_text(json.dumps(report, indent=2))
    _write_report_md(report, report_path)
    print(
        json.dumps(
            {
                "succeeded": report["succeeded"],
                "failed": report["failed"],
                "selected": subset["selected"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orthogonality gate for first dehydron barcode scalar arm"
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--raw-bars-dir", type=Path, default=_DEFAULT_RAW_BARS)
    parser.add_argument("--pdb-dir", type=Path, default=_REPO_ROOT / "pdb_cache")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--long-lived-threshold",
        type=float,
        default=LONG_LIVED_PERSISTENCE_ANGSTROM,
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=DEFAULT_REDUNDANCY_THRESHOLD,
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    sys.exit(run(args))


if __name__ == "__main__":
    main()
