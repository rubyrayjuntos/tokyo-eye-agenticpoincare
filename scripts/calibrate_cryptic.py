"""Heuristic v1 Calibration for cryptic site classification.

Evaluates the site_type classification heuristic against a benchmark dataset
with ground truth labels. Computes precision, recall, and F1 per site_type,
flags types below the minimum precision threshold (default 0.7), and outputs
an updated calibration report.

Usage:
    PYTHONPATH=. python scripts/calibrate_cryptic.py \
        --benchmark data/calibration/benchmark_cryptic_sites.json \
        --output data/calibration/heuristic_calibration_report.json

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.tools.cryptic.scan_phase import HEURISTIC_VERSION, classify_site_type
from agent.tools.cryptic.seed_generator import CandidateCluster

logger = logging.getLogger(__name__)

# Minimum precision threshold — site_types below this are flagged as unreliable
DEFAULT_MINIMUM_PRECISION = 0.7

# Valid site types produced by the heuristic
VALID_SITE_TYPES = [
    "cryptic_wedge",
    "structural_stent",
    "dynamic_lid",
    "allosteric_clamp",
    "strain_relief_insert",
    "surface_pocket",
]


@dataclass
class BenchmarkSiteEntry:
    """A benchmark site with ground truth site_type for heuristic evaluation."""

    pdb_id: str
    chain: str
    residue_ids: list[str]
    ground_truth_type: str  # The true site_type label
    ground_truth_label: str  # "true_positive" or "true_negative"
    composite_score: float = 0.5
    graph_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SiteTypeMetrics:
    """Precision, recall, and F1 for a single site_type."""

    site_type: str
    precision: float
    recall: float
    f1: float
    n_samples: int  # total ground truth samples of this type
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class HeuristicCalibrationResult:
    """Full result of heuristic evaluation against benchmark."""

    heuristic_version: str
    per_site_type: dict[str, SiteTypeMetrics]
    flagged_types: list[str]
    minimum_precision_threshold: float
    total_sites_evaluated: int
    overall_accuracy: float
    predictions: list[dict[str, str]]  # [{pdb_id, predicted, actual}, ...]


def load_benchmark_for_heuristic(benchmark_path: str) -> list[BenchmarkSiteEntry]:
    """Load benchmark sites from JSON file for heuristic evaluation.

    Expected format: list of objects with pdb_id, chain, residue_ids,
    ground_truth_type, and optionally composite_score and graph_metrics.

    Raises:
        FileNotFoundError: If benchmark file does not exist.
        ValueError: If benchmark format is invalid.
    """
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with path.open() as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Benchmark file must contain a JSON array of site objects")

    sites: list[BenchmarkSiteEntry] = []
    required_fields = {"pdb_id", "chain", "residue_ids", "ground_truth_type"}

    for i, entry in enumerate(raw):
        missing = required_fields - set(entry.keys())
        if missing:
            logger.warning("Benchmark entry %d missing fields %s — skipping", i, missing)
            continue

        sites.append(
            BenchmarkSiteEntry(
                pdb_id=entry["pdb_id"],
                chain=entry["chain"],
                residue_ids=entry["residue_ids"],
                ground_truth_type=entry["ground_truth_type"],
                ground_truth_label=entry.get("ground_truth_label", "true_positive"),
                composite_score=entry.get("composite_score", 0.5),
                graph_metrics=entry.get("graph_metrics", {}),
                notes=entry.get("notes", ""),
            )
        )

    if not sites:
        raise ValueError("No valid benchmark sites found in file")

    return sites


def _build_candidate_cluster(site: BenchmarkSiteEntry) -> CandidateCluster:
    """Build a CandidateCluster from a benchmark site entry for classification.

    Uses the benchmark's composite_score and residue_ids to create a cluster
    that can be passed to classify_site_type.
    """
    # Compute a synthetic centroid (not needed for classification, but required by dataclass)
    return CandidateCluster(
        cluster_id=0,
        residue_ids=site.residue_ids,
        centroid_xyz=(0.0, 0.0, 0.0),
        composite_score=site.composite_score,
        member_count=len(site.residue_ids),
    )


def run_heuristic_evaluation(
    sites: list[BenchmarkSiteEntry],
    minimum_precision: float = DEFAULT_MINIMUM_PRECISION,
) -> HeuristicCalibrationResult:
    """Run all benchmark sites through the mapper's classify_site_type heuristic.

    For each benchmark site:
    1. Construct a CandidateCluster from the benchmark data
    2. Call classify_site_type with available graph_metrics
    3. Compare predicted site_type against ground_truth_type

    Then compute precision/recall/F1 per site_type and flag unreliable types.

    Args:
        sites: List of benchmark sites with ground truth labels.
        minimum_precision: Threshold below which a site_type is flagged.

    Returns:
        HeuristicCalibrationResult with per-type metrics and flagged types.

    Requirements: 7.1, 7.2, 7.3, 7.4
    """
    predictions: list[dict[str, str]] = []

    for site in sites:
        cluster = _build_candidate_cluster(site)
        graph_metrics = site.graph_metrics if site.graph_metrics else None
        predicted = classify_site_type(cluster, graph_metrics)
        predictions.append({
            "pdb_id": site.pdb_id,
            "chain": site.chain,
            "predicted": predicted,
            "actual": site.ground_truth_type,
        })

    # Compute per-type metrics
    per_site_type = compute_per_type_metrics(predictions)

    # Flag types below minimum precision
    flagged_types = [
        st for st, metrics in per_site_type.items()
        if metrics.precision < minimum_precision and metrics.n_samples > 0
    ]

    # Overall accuracy
    correct = sum(1 for p in predictions if p["predicted"] == p["actual"])
    overall_accuracy = correct / len(predictions) if predictions else 0.0

    return HeuristicCalibrationResult(
        heuristic_version=HEURISTIC_VERSION,
        per_site_type=per_site_type,
        flagged_types=sorted(flagged_types),
        minimum_precision_threshold=minimum_precision,
        total_sites_evaluated=len(predictions),
        overall_accuracy=overall_accuracy,
        predictions=predictions,
    )


def compute_per_type_metrics(
    predictions: list[dict[str, str]],
) -> dict[str, SiteTypeMetrics]:
    """Compute precision, recall, and F1 per site_type from predictions.

    For each site_type T:
    - TP = predicted T and actual T
    - FP = predicted T but actual != T
    - FN = actual T but predicted != T
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * P * R / (P + R)

    Args:
        predictions: List of {pdb_id, predicted, actual} dicts.

    Returns:
        Dict mapping site_type → SiteTypeMetrics.
    """
    # Collect all site_types from both predictions and actuals
    all_types: set[str] = set()
    for p in predictions:
        all_types.add(p["predicted"])
        all_types.add(p["actual"])

    results: dict[str, SiteTypeMetrics] = {}

    for site_type in sorted(all_types):
        tp = sum(
            1 for p in predictions
            if p["predicted"] == site_type and p["actual"] == site_type
        )
        fp = sum(
            1 for p in predictions
            if p["predicted"] == site_type and p["actual"] != site_type
        )
        fn = sum(
            1 for p in predictions
            if p["actual"] == site_type and p["predicted"] != site_type
        )

        n_samples = tp + fn  # total ground truth instances of this type

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[site_type] = SiteTypeMetrics(
            site_type=site_type,
            precision=precision,
            recall=recall,
            f1=f1,
            n_samples=n_samples,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    return results


def generate_heuristic_report(
    result: HeuristicCalibrationResult,
    output_path: str,
    previous_report_path: str | None = None,
) -> dict[str, Any]:
    """Generate a structured calibration report and write to JSON.

    Output format matches the Calibration Report Schema from the design doc.

    Args:
        result: The evaluation result.
        output_path: Path to write the JSON report.
        previous_report_path: Optional path to a previous report for regression detection.

    Returns:
        The report as a dict.

    Requirements: 7.3, 7.4
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    per_site_type_dict: dict[str, dict[str, Any]] = {}
    for st, metrics in result.per_site_type.items():
        per_site_type_dict[st] = {
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "n_samples": metrics.n_samples,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
        }

    report: dict[str, Any] = {
        "version": "1.0",
        "calibration_run_id": run_id,
        "timestamp": timestamp,
        "heuristic_metrics": {
            "overall_accuracy": round(result.overall_accuracy, 4),
            "per_site_type": per_site_type_dict,
            "heuristic_version": result.heuristic_version,
            "minimum_precision_threshold": result.minimum_precision_threshold,
            "flagged_types": result.flagged_types,
            "total_sites_evaluated": result.total_sites_evaluated,
        },
        "precision_by_site_type": {
            st: round(metrics.precision, 4)
            for st, metrics in result.per_site_type.items()
        },
        "flagged_types": result.flagged_types,
        "predictions": result.predictions,
    }

    # Detect regressions against previous report
    regression_flags: list[str] = []
    if previous_report_path:
        prev_path = Path(previous_report_path)
        if prev_path.exists():
            with prev_path.open() as f:
                prev_report = json.load(f)
            regression_flags = _detect_regressions(report, prev_report)

    report["regression_flags"] = regression_flags

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.info("Heuristic calibration report written to: %s", output_path)
    return report


def _detect_regressions(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> list[str]:
    """Detect regressions between current and previous calibration reports.

    A regression is flagged when:
    - A site_type's precision drops by more than 0.05
    - A site_type that was above threshold is now below threshold
    - Overall accuracy drops by more than 0.05

    Returns list of regression description strings.
    """
    flags: list[str] = []

    prev_per_type = previous.get("heuristic_metrics", {}).get("per_site_type", {})
    curr_per_type = current.get("heuristic_metrics", {}).get("per_site_type", {})
    threshold = current.get("heuristic_metrics", {}).get(
        "minimum_precision_threshold", DEFAULT_MINIMUM_PRECISION
    )

    for st, curr_metrics in curr_per_type.items():
        if st in prev_per_type:
            prev_precision = prev_per_type[st].get("precision", 0.0)
            curr_precision = curr_metrics.get("precision", 0.0)
            drop = prev_precision - curr_precision

            if drop > 0.05:
                flags.append(
                    f"{st}: precision dropped {prev_precision:.3f} → {curr_precision:.3f} "
                    f"(Δ={-drop:.3f})"
                )

            if prev_precision >= threshold and curr_precision < threshold:
                flags.append(
                    f"{st}: precision fell below threshold "
                    f"({prev_precision:.3f} → {curr_precision:.3f}, threshold={threshold})"
                )

    # Overall accuracy regression
    prev_accuracy = previous.get("heuristic_metrics", {}).get("overall_accuracy", 0.0)
    curr_accuracy = current.get("heuristic_metrics", {}).get("overall_accuracy", 0.0)
    if prev_accuracy - curr_accuracy > 0.05:
        flags.append(
            f"overall_accuracy dropped {prev_accuracy:.3f} → {curr_accuracy:.3f}"
        )

    return flags


def main() -> None:
    """CLI entry point for heuristic calibration.

    Usage:
        PYTHONPATH=. python scripts/calibrate_cryptic.py \
            --benchmark data/calibration/benchmark_cryptic_sites.json \
            --output data/calibration/heuristic_calibration_report.json
    """
    parser = argparse.ArgumentParser(
        description="Run heuristic v1 calibration against benchmark sites"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark_cryptic_sites.json",
    )
    parser.add_argument(
        "--output",
        default="data/calibration/heuristic_calibration_report.json",
        help="Path to write calibration report JSON",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MINIMUM_PRECISION,
        help=f"Minimum precision threshold for flagging (default: {DEFAULT_MINIMUM_PRECISION})",
    )
    parser.add_argument(
        "--previous-report",
        default=None,
        help="Path to previous calibration report for regression detection",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting heuristic v1 calibration")
    logger.info("  Benchmark: %s", args.benchmark)
    logger.info("  Output: %s", args.output)
    logger.info("  Min precision threshold: %.2f", args.min_precision)

    # Load benchmark
    sites = load_benchmark_for_heuristic(args.benchmark)
    logger.info("  Loaded %d benchmark sites", len(sites))

    # Run evaluation
    result = run_heuristic_evaluation(sites, minimum_precision=args.min_precision)

    # Log summary
    logger.info("  Overall accuracy: %.3f", result.overall_accuracy)
    logger.info("  Sites evaluated: %d", result.total_sites_evaluated)
    for st, metrics in result.per_site_type.items():
        logger.info(
            "    %s: P=%.3f R=%.3f F1=%.3f (n=%d)",
            st, metrics.precision, metrics.recall, metrics.f1, metrics.n_samples,
        )
    if result.flagged_types:
        logger.warning("  FLAGGED types (precision < %.2f): %s",
                       args.min_precision, result.flagged_types)
    else:
        logger.info("  No types flagged — all above precision threshold")

    # Generate report
    generate_heuristic_report(
        result,
        output_path=args.output,
        previous_report_path=args.previous_report,
    )

    # Exit with warning code if any types are flagged
    if result.flagged_types:
        logger.warning(
            "%d site_type(s) flagged as unreliable: %s",
            len(result.flagged_types),
            result.flagged_types,
        )
        sys.exit(0)  # Not a hard failure — just a warning

    logger.info("Heuristic calibration complete — all types above threshold")


if __name__ == "__main__":
    main()
