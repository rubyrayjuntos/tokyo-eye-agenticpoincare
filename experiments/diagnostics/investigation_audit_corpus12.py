#!/usr/bin/env python3
"""Score a run's corpus viewers using the trusted physics Investigation channel.

The legacy ``investigation`` field is evidential (aleatoric × (1 − epistemic))
and is intentionally never accepted as a fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SCORING_PATH = "physics_investigation"
FORBIDDEN_SCORING_PATH = "investigation"
DEFAULT_EXPECTED_STRUCTURES = 12
DEFAULT_MIN_RIM_ENRICHED = 11

KRAS_MOTIFS: dict[str, tuple[int, ...]] = {
    "switch_ii_60_63": tuple(range(60, 64)),
    "switch_i_s39": (39,),
    "siip_74_78": (74, 78),
    "alpha3_105_107": tuple(range(105, 108)),
    "e98": (98,),
    "cterm_164_169": tuple(range(164, 170)),
}


def load_residue_metrics(html_path: Path) -> list[dict[str, Any]]:
    """Extract ``RESIDUE_METRICS`` from an exported interactive viewer."""
    match = re.search(
        r"RESIDUE_METRICS\s*=\s*(\[.*?\]);",
        html_path.read_text(),
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"RESIDUE_METRICS not found in {html_path}")
    rows = json.loads(match.group(1))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"RESIDUE_METRICS must be a non-empty list in {html_path}")
    return rows


def _physics_score(row: dict[str, Any]) -> float:
    if SCORING_PATH not in row:
        raise ValueError(
            f"row {row.get('key', '<unknown>')} lacks {SCORING_PATH}; "
            f"refusing evidential-only {FORBIDDEN_SCORING_PATH} scoring"
        )
    return float(row[SCORING_PATH])


def _quartile_partition(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores = sorted(_physics_score(row) for row in rows)
    high_cut = scores[int(0.75 * (len(scores) - 1))]
    low_cut = scores[int(0.25 * (len(scores) - 1))]
    high = [row for row in rows if _physics_score(row) >= high_cut]
    low = [row for row in rows if _physics_score(row) <= low_cut]
    return high, low


def _motif_summary(
    rows: Sequence[dict[str, Any]],
    motifs: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    by_residue: dict[int, dict[str, Any]] = {}
    ranked = sorted(rows, key=_physics_score, reverse=True)
    rank = {str(row["key"]): index for index, row in enumerate(ranked, start=1)}
    for row in rows:
        try:
            residue = int(str(row["key"]).split(":")[1])
        except (IndexError, ValueError):
            continue
        by_residue[residue] = row

    result: dict[str, Any] = {}
    for name, residue_numbers in motifs.items():
        present = [by_residue[number] for number in residue_numbers if number in by_residue]
        result[name] = {
            "present": bool(present),
            "mean_physics_investigation": (
                statistics.mean(_physics_score(row) for row in present)
                if present
                else None
            ),
            "ranks": [rank[str(row["key"])] for row in present],
        }
    return result


def summarize_structure(
    rows: Sequence[dict[str, Any]],
    structure_id: str,
) -> dict[str, Any]:
    """Summarize one structure; hard-fail if physics scores are absent."""
    if not rows:
        raise ValueError(f"{structure_id}: no residue rows")
    for row in rows:
        _physics_score(row)

    high, low = _quartile_partition(rows)
    ranked = sorted(rows, key=_physics_score, reverse=True)
    expert_counts = Counter(int(row.get("expert", -1)) for row in rows)
    n_rows = len(rows)
    summary: dict[str, Any] = {
        "structure_id": structure_id.upper(),
        "scoring_path": SCORING_PATH,
        "n_residues": n_rows,
        "mean_depth_high_investigation": statistics.mean(
            float(row.get("cone_depth", 0.0)) for row in high
        ),
        "mean_depth_low_investigation": statistics.mean(
            float(row.get("cone_depth", 0.0)) for row in low
        ),
        "hard_expert_share": {
            str(expert): count / n_rows for expert, count in sorted(expert_counts.items())
        },
        "top_investigation": [
            {
                "key": str(row["key"]),
                SCORING_PATH: _physics_score(row),
                "cone_depth": float(row.get("cone_depth", 0.0)),
                "expert": int(row.get("expert", -1)),
            }
            for row in ranked[:10]
        ],
    }
    if structure_id.lower() == "4obe":
        summary["motifs"] = _motif_summary(rows, KRAS_MOTIFS)
    return summary


def audit_run(
    run_dir: Path,
    *,
    expected_structures: int = DEFAULT_EXPECTED_STRUCTURES,
    min_rim_enriched: int = DEFAULT_MIN_RIM_ENRICHED,
) -> dict[str, Any]:
    """Audit every exported structure viewer under ``run_dir``."""
    viewers_dir = run_dir / "viewers"
    html_paths = sorted(viewers_dir.glob("*/*_interactive.html"))
    structures: dict[str, Any] = {}
    for html_path in html_paths:
        structure_id = html_path.parent.name
        structures[structure_id.lower()] = summarize_structure(
            load_residue_metrics(html_path),
            structure_id,
        )

    n_rim_enriched = sum(
        summary["mean_depth_high_investigation"]
        > summary["mean_depth_low_investigation"]
        for summary in structures.values()
    )
    complete = len(structures) == expected_structures
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "scoring_path": SCORING_PATH,
        "forbidden_fallback": FORBIDDEN_SCORING_PATH,
        "expected_structures": expected_structures,
        "structures_scored": len(structures),
        "complete_corpus": complete,
        "n_rim_enriched": n_rim_enriched,
        "min_rim_enriched": min_rim_enriched,
        "passes_rim_enrichment": complete and n_rim_enriched >= min_rim_enriched,
        "structures": structures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physics-only corpus Investigation audit"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--expected-structures",
        type=int,
        default=DEFAULT_EXPECTED_STRUCTURES,
    )
    parser.add_argument(
        "--min-rim-enriched",
        type=int,
        default=DEFAULT_MIN_RIM_ENRICHED,
    )
    args = parser.parse_args()

    report = audit_run(
        args.run_dir,
        expected_structures=args.expected_structures,
        min_rim_enriched=args.min_rim_enriched,
    )
    out_path = args.out or args.run_dir / "investigation_audit_corpus12.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "structures"}))
    return 0 if report["complete_corpus"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
