#!/usr/bin/env python3
"""Stage A-12 dehydron barcode bar-length diagnostic (pure TDA — no GNN).

Computes Euclidean dehydron-midpoint witness persistence for each enabled
manifest structure, then reports:

* pooled and **per-structure** H0 / H1 bar-length distributions
* dehydron midpoint count per structure (feeds min-dehydron guard)
* suggested long-lived thresholds (percentiles + largest-gap heuristic)
* dominance check so one protein cannot silently own a pooled percentile

Raw per-structure bar lists are written to disk so the threshold can be
revisited later without re-running GUDHI.

Example::

  PYTHONPATH=. python -m experiments.diagnostics.dehydron_bar_length_threshold \\
    --manifest manifests/v6_corpus_stage_a_small_v1.json \\
    --pdb-dir pdb_cache \\
    --out-dir checkpoints/v65/diagnostics/dehydron_bar_length_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.training.v6 import _data
from experiments.training.v6.corpus import iter_corpus_entries
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    PersistenceBar,
    _load_chain_structure_atoms,
    compute_witness_persistence,
    extract_dehydron_midpoints,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json"
_DEFAULT_OUT = _REPO_ROOT / "checkpoints" / "v65" / "diagnostics" / "dehydron_bar_length_v1"

# Production noise floor (bars below this never enter feature aggregation).
DEFAULT_NOISE_FLOOR_A = 0.1
# Soft flag: one structure owns more than this fraction of pooled H1 bars.
DOMINANCE_FRAC = 0.25

PERCENTILES = (10, 25, 50, 75, 90, 95, 99)

MappingLike = Any


@dataclass(frozen=True)
class StructureBarcodeRaw:
    pdb_id: str
    chain: str
    n_midpoints: int
    n_residues: int
    bars: list[dict[str, float | int]]
    params: dict[str, Any]


def bar_to_dict(bar: PersistenceBar) -> dict[str, float | int]:
    return {
        "dim": int(bar.dim),
        "birth": float(bar.birth),
        "death": float(bar.death),
        "persistence": float(bar.persistence),
    }


def _get(bar: MappingLike, key: str) -> Any:
    if isinstance(bar, dict):
        return bar[key]
    return getattr(bar, key)


def persistences_for_dim(
    bars: Sequence[MappingLike],
    dim: int,
    *,
    min_persistence: float = 0.0,
) -> np.ndarray:
    """Extract persistence lengths for one homology dimension."""
    vals: list[float] = []
    for bar in bars:
        if int(_get(bar, "dim")) != int(dim):
            continue
        p = float(_get(bar, "persistence"))
        if p >= float(min_persistence):
            vals.append(p)
    return np.asarray(vals, dtype=np.float64)


def summarize_lengths(values: np.ndarray) -> dict[str, Any]:
    """Scalar summary of a 1-D persistence-length array."""
    if values.size == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {str(p): None for p in PERCENTILES},
        }
    pct = {str(p): float(np.percentile(values, p)) for p in PERCENTILES}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "percentiles": pct,
    }


def largest_gap_threshold(
    values: np.ndarray,
    *,
    search_from_percentile: float = 50.0,
    min_tail_count: int = 3,
) -> dict[str, Any] | None:
    """Largest consecutive gap in the upper half of sorted lengths.

    Returns the midpoint of that gap as a candidate long-lived cut, or None
    if the sample is too small / no usable gap.
    """
    if values.size < max(min_tail_count + 1, 4):
        return None
    sorted_v = np.sort(values)
    floor = float(np.percentile(sorted_v, search_from_percentile))
    idxs = np.where(sorted_v >= floor)[0]
    if idxs.size < 2:
        return None
    best: dict[str, Any] | None = None
    for i in range(int(idxs[0]), len(sorted_v) - 1):
        lo = float(sorted_v[i])
        hi = float(sorted_v[i + 1])
        gap = hi - lo
        n_above = int((sorted_v > lo).sum())
        if n_above < min_tail_count:
            continue
        if best is None or gap > float(best["gap"]):
            best = {
                "gap": gap,
                "lo": lo,
                "hi": hi,
                "threshold_midpoint": 0.5 * (lo + hi),
                "n_bars_strictly_above_lo": n_above,
                "search_from_percentile": search_from_percentile,
            }
    return best


def dominance_by_structure(
    per_structure_counts: dict[str, int],
) -> dict[str, Any]:
    """Flag whether pooled stats are dominated by one structure's bar count."""
    total = int(sum(per_structure_counts.values()))
    if total <= 0:
        return {
            "total_bars": 0,
            "max_frac": None,
            "max_structure": None,
            "dominated": False,
            "per_structure_frac": {},
        }
    fracs = {k: v / total for k, v in per_structure_counts.items()}
    max_key = max(fracs, key=fracs.get)
    max_frac = float(fracs[max_key])
    return {
        "total_bars": total,
        "max_frac": max_frac,
        "max_structure": max_key,
        "dominated": max_frac >= DOMINANCE_FRAC,
        "dominance_threshold": DOMINANCE_FRAC,
        "per_structure_frac": {k: float(v) for k, v in sorted(fracs.items())},
    }


def histogram_counts(
    values: np.ndarray,
    *,
    bin_width: float = 0.25,
    max_edge: float | None = None,
) -> dict[str, Any]:
    if values.size == 0:
        return {"bin_width": bin_width, "edges": [], "counts": []}
    hi = float(values.max()) if max_edge is None else float(max_edge)
    hi = max(hi, bin_width)
    n_bins = max(1, int(math.ceil(hi / bin_width)))
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "bin_width": bin_width,
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def build_dim_report(
    *,
    pooled_bars: list[dict[str, float | int]],
    per_structure_bars: dict[str, list[dict[str, float | int]]],
    dim: int,
    noise_floor: float,
) -> dict[str, Any]:
    """Pooled + per-structure summaries for one homology dimension."""
    pooled_all = persistences_for_dim(pooled_bars, dim, min_persistence=0.0)
    pooled_feat = persistences_for_dim(
        pooled_bars, dim, min_persistence=noise_floor
    )

    per_struct: dict[str, Any] = {}
    feat_counts: dict[str, int] = {}
    for key, bars in sorted(per_structure_bars.items()):
        all_v = persistences_for_dim(bars, dim, min_persistence=0.0)
        feat_v = persistences_for_dim(bars, dim, min_persistence=noise_floor)
        feat_counts[key] = int(feat_v.size)
        per_struct[key] = {
            "all_bars": summarize_lengths(all_v),
            "above_noise_floor": summarize_lengths(feat_v),
            "n_above_noise": int(feat_v.size),
        }

    gap = largest_gap_threshold(pooled_feat)
    dominance = dominance_by_structure(feat_counts)
    pct = summarize_lengths(pooled_feat)["percentiles"]

    suggestions = {
        "p75": pct.get("75"),
        "p90": pct.get("90"),
        "largest_gap_midpoint": None if gap is None else gap["threshold_midpoint"],
        "note": (
            "Prefer an empirical cut on the noise-filtered H1 distribution. "
            "If dominance.dominated is true, inspect per-structure views before "
            "trusting a pooled percentile."
        ),
    }

    return {
        "dim": dim,
        "noise_floor_angstrom": noise_floor,
        "pooled": {
            "all_bars": summarize_lengths(pooled_all),
            "above_noise_floor": summarize_lengths(pooled_feat),
            "histogram_above_noise": histogram_counts(pooled_feat),
        },
        "per_structure": per_struct,
        "dominance_above_noise": dominance,
        "largest_gap_above_noise": gap,
        "suggested_long_lived_candidates": suggestions,
    }


def compute_structure_raw(
    pdb_path: Path,
    pdb_id: str,
    chain: str,
    *,
    max_alpha_angstrom: float = 20.0,
    n_landmarks: int = 30,
    random_state: int = 42,
) -> StructureBarcodeRaw:
    """Run midpoint extraction + witness persistence with noise filter off.

    Bars are saved raw (min_persistence=0); summaries apply the noise floor.
    """
    structure_atoms, residue_index_map = _load_chain_structure_atoms(pdb_path, chain)
    midpoints = extract_dehydron_midpoints(structure_atoms, residue_index_map)
    bars = compute_witness_persistence(
        midpoints,
        max_alpha_angstrom=max_alpha_angstrom,
        min_persistence_angstrom=0.0,
        n_landmarks=n_landmarks,
        random_state=random_state,
    )
    return StructureBarcodeRaw(
        pdb_id=pdb_id.upper(),
        chain=chain,
        n_midpoints=len(midpoints),
        n_residues=len(residue_index_map),
        bars=[bar_to_dict(b) for b in bars],
        params={
            "max_alpha_angstrom": max_alpha_angstrom,
            "min_persistence_angstrom_at_compute": 0.0,
            "n_landmarks": n_landmarks,
            "random_state": random_state,
            "barcode_feature_version": BARCODE_FEATURE_VERSION,
            "landmark_method": "kmeans_v3_phase1",
        },
    )


def recommend_threshold(h1_report: dict[str, Any]) -> dict[str, Any]:
    """Pick a default recommendation with an explicit rationale string."""
    suggestions = h1_report["suggested_long_lived_candidates"]
    dominance = h1_report["dominance_above_noise"]
    gap = h1_report.get("largest_gap_above_noise")
    p75 = suggestions.get("p75")
    p90 = suggestions.get("p90")

    if p75 is None:
        return {
            "recommended_angstrom": None,
            "method": None,
            "rationale": "No H1 bars above noise floor — cannot set long-lived threshold.",
            "needs_manual_review": True,
        }

    if gap is not None and float(gap["gap"]) >= 0.25 * max(float(p75), 1e-6):
        method = "largest_gap_midpoint"
        value = float(gap["threshold_midpoint"])
        rationale = (
            f"Largest upper-half gap ({gap['gap']:.3f} Å between "
            f"{gap['lo']:.3f} and {gap['hi']:.3f}); midpoint used as cut."
        )
    else:
        method = "pooled_percentile_75"
        value = float(p75)
        rationale = (
            "No strong upper-half gap; defaulting to pooled H1 75th percentile "
            f"({value:.3f} Å) of noise-filtered bars."
        )

    needs_review = bool(dominance.get("dominated"))
    if needs_review:
        rationale += (
            f" MANUAL REVIEW: {dominance['max_structure']} owns "
            f"{100.0 * float(dominance['max_frac']):.1f}% of pooled H1 bars "
            f"(≥{100.0 * DOMINANCE_FRAC:.0f}% dominance flag)."
        )

    return {
        "recommended_angstrom": value,
        "method": method,
        "rationale": rationale,
        "needs_manual_review": needs_review,
        "alternates": {"p75": p75, "p90": p90},
    }


def write_report_md(summary: dict[str, Any], path: Path) -> None:
    rec = summary["recommendation"]
    h1 = summary["h1"]
    h0 = summary["h0"]
    counts = summary["dehydron_midpoint_counts"]
    lines = [
        "# Dehydron bar-length threshold diagnostic",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Structures succeeded: **{summary['succeeded']}** / {summary['attempted']}",
        f"- Noise floor (feature-entering): **{summary['noise_floor_angstrom']} Å**",
        f"- Barcode SSOT version: `{summary['barcode_feature_version']}`",
        "",
        "## Recommendation",
        "",
        f"- **Recommended long-lived threshold:** "
        f"{rec['recommended_angstrom']} Å (`{rec['method']}`)",
        f"- Needs manual review: **{rec['needs_manual_review']}**",
        f"- Rationale: {rec['rationale']}",
        "",
        "## Dehydron midpoint counts (min-dehydron guard input)",
        "",
        "| Structure | n_midpoints | n_residues |",
        "|-----------|-------------|------------|",
    ]
    for row in counts["per_structure"]:
        lines.append(
            f"| {row['pdb_id']}:{row['chain']} | {row['n_midpoints']} | "
            f"{row['n_residues']} |"
        )
    lines.extend(
        [
            "",
            f"- min / median / max midpoints: "
            f"{counts['summary']['min']} / {counts['summary']['median']} / "
            f"{counts['summary']['max']}",
            "",
            "## H1 (primary) — pooled above noise",
            "",
            f"- n={h1['pooled']['above_noise_floor']['n']}, "
            f"p50={h1['pooled']['above_noise_floor']['percentiles']['50']}, "
            f"p75={h1['pooled']['above_noise_floor']['percentiles']['75']}, "
            f"p90={h1['pooled']['above_noise_floor']['percentiles']['90']}",
            f"- Dominance: max_frac="
            f"{h1['dominance_above_noise']['max_frac']} "
            f"({h1['dominance_above_noise']['max_structure']}); "
            f"dominated={h1['dominance_above_noise']['dominated']}",
            "",
            "## H0 — pooled above noise (context)",
            "",
            f"- n={h0['pooled']['above_noise_floor']['n']}, "
            f"p50={h0['pooled']['above_noise_floor']['percentiles']['50']}, "
            f"p75={h0['pooled']['above_noise_floor']['percentiles']['75']}, "
            f"p90={h0['pooled']['above_noise_floor']['percentiles']['90']}",
            f"- Dominance: max_frac="
            f"{h0['dominance_above_noise']['max_frac']} "
            f"({h0['dominance_above_noise']['max_structure']}); "
            f"dominated={h0['dominance_above_noise']['dominated']}",
            "",
            "## Artifacts",
            "",
            f"- Raw bars: `{summary['artifacts']['bars_raw_dir']}`",
            f"- Machine summary: `{summary['artifacts']['summary_json']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    bars_dir = out_dir / "bars_raw"
    bars_dir.mkdir(parents=True, exist_ok=True)

    entries = iter_corpus_entries(manifest, max_proteins=args.max_proteins)
    if not entries:
        logger.error("No enabled proteins in manifest %s", manifest)
        return 1

    noise_floor = float(args.noise_floor)
    per_structure_bars: dict[str, list[dict[str, float | int]]] = {}
    pooled_bars: list[dict[str, float | int]] = []
    midpoint_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    structure_payloads: list[dict[str, Any]] = []

    logger.info(
        "Bar-length diagnostic: %d structures from %s -> %s",
        len(entries),
        manifest,
        out_dir,
    )

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        key = f"{pdb_id}:{chain}"
        try:
            pdb_path = _data._download_pdb(pdb_id, pdb_dir)
            raw = compute_structure_raw(
                pdb_path,
                pdb_id,
                chain,
                max_alpha_angstrom=args.max_alpha,
                n_landmarks=args.n_landmarks,
                random_state=args.random_state,
            )
            raw_path = bars_dir / f"{pdb_id}_{chain}_bars.json"
            payload = {
                "pdb_id": raw.pdb_id,
                "chain": raw.chain,
                "n_midpoints": raw.n_midpoints,
                "n_residues": raw.n_residues,
                "params": raw.params,
                "bars": raw.bars,
            }
            raw_path.write_text(json.dumps(payload, indent=2))
            structure_payloads.append({**payload, "raw_path": str(raw_path)})

            per_structure_bars[key] = raw.bars
            pooled_bars.extend(raw.bars)
            midpoint_rows.append(
                {
                    "pdb_id": raw.pdb_id,
                    "chain": raw.chain,
                    "n_midpoints": raw.n_midpoints,
                    "n_residues": raw.n_residues,
                    "n_bars_total": len(raw.bars),
                    "n_h0": int(persistences_for_dim(raw.bars, 0).size),
                    "n_h1": int(persistences_for_dim(raw.bars, 1).size),
                }
            )
            logger.info(
                "OK %s — midpoints=%d bars=%d (H0=%d H1=%d)",
                key,
                raw.n_midpoints,
                len(raw.bars),
                midpoint_rows[-1]["n_h0"],
                midpoint_rows[-1]["n_h1"],
            )
        except Exception as exc:
            msg = f"{key}: {exc}"
            failures.append(msg)
            logger.error("FAIL %s", msg)
            if args.fail_fast:
                break

    if not midpoint_rows:
        logger.error("All structures failed")
        return 1

    counts_arr = np.asarray([r["n_midpoints"] for r in midpoint_rows], dtype=np.float64)
    dehydron_counts = {
        "per_structure": midpoint_rows,
        "summary": {
            "n_structures": len(midpoint_rows),
            "min": int(counts_arr.min()),
            "max": int(counts_arr.max()),
            "mean": float(counts_arr.mean()),
            "median": float(np.median(counts_arr)),
            "histogram": {
                str(int(v)): int((counts_arr == v).sum())
                for v in sorted(set(int(x) for x in counts_arr))
            },
        },
        "guard_hint": (
            "Set min-dehydron-count at or below the smallest non-degenerate "
            "cluster you are willing to trust for landmark k-means; structures "
            "below that route to missing-mask (design §4.3)."
        ),
    }

    h0 = build_dim_report(
        pooled_bars=pooled_bars,
        per_structure_bars=per_structure_bars,
        dim=0,
        noise_floor=noise_floor,
    )
    h1 = build_dim_report(
        pooled_bars=pooled_bars,
        per_structure_bars=per_structure_bars,
        dim=1,
        noise_floor=noise_floor,
    )
    recommendation = recommend_threshold(h1)

    h0_p75 = h0["pooled"]["above_noise_floor"]["percentiles"].get("75")
    h1_p75 = h1["pooled"]["above_noise_floor"]["percentiles"].get("75")
    homology_note: dict[str, Any] = {
        "h0_p75": h0_p75,
        "h1_p75": h1_p75,
        "relative_abs_diff": None,
        "flag_suspiciously_similar": False,
    }
    if h0_p75 is not None and h1_p75 is not None and max(h0_p75, h1_p75) > 0:
        rel = abs(float(h0_p75) - float(h1_p75)) / max(float(h0_p75), float(h1_p75))
        homology_note["relative_abs_diff"] = rel
        homology_note["flag_suspiciously_similar"] = rel < 0.05

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    counts_path = out_dir / "dehydron_midpoint_counts.json"

    summary = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "attempted": len(entries),
        "succeeded": len(midpoint_rows),
        "failed": len(failures),
        "failures": failures,
        "noise_floor_angstrom": noise_floor,
        "barcode_feature_version": BARCODE_FEATURE_VERSION,
        "params": {
            "max_alpha_angstrom": args.max_alpha,
            "n_landmarks": args.n_landmarks,
            "random_state": args.random_state,
            "min_persistence_at_compute": 0.0,
        },
        "dehydron_midpoint_counts": dehydron_counts,
        "h0": h0,
        "h1": h1,
        "homology_similarity_check": homology_note,
        "recommendation": recommendation,
        "artifacts": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "dehydron_counts_json": str(counts_path),
            "bars_raw_dir": str(bars_dir),
        },
        "structures": [
            {
                "pdb_id": p["pdb_id"],
                "chain": p["chain"],
                "n_midpoints": p["n_midpoints"],
                "n_bars": len(p["bars"]),
                "raw_path": p["raw_path"],
            }
            for p in structure_payloads
        ],
    }

    counts_path.write_text(json.dumps(dehydron_counts, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    write_report_md(summary, report_path)

    print(
        json.dumps(
            {
                "succeeded": summary["succeeded"],
                "failed": summary["failed"],
                "recommended_long_lived_angstrom": recommendation["recommended_angstrom"],
                "method": recommendation["method"],
                "needs_manual_review": recommendation["needs_manual_review"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pure-TDA dehydron bar-length diagnostic for long-lived threshold "
            "+ min-dehydron guard inputs"
        )
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--pdb-dir", type=Path, default=_REPO_ROOT / "pdb_cache")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=DEFAULT_NOISE_FLOOR_A,
        help="Persistence floor matching feature aggregation (default 0.1 Å)",
    )
    parser.add_argument("--max-alpha", type=float, default=20.0)
    parser.add_argument("--n-landmarks", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
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
