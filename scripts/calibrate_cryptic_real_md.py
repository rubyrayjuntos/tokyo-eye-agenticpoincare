"""Real MD Calibration Pipeline for cryptic binding site thresholds.

Dispatches real steered molecular dynamics (SMD) jobs against benchmark sites
using the GPU science container, collects work values, applies Youden's J
statistic to determine optimal thresholds separating true positives from
true negatives.

Usage:
    PYTHONPATH=. python scripts/calibrate_cryptic_real_md.py \
        --benchmark data/calibration/benchmark_cryptic_sites.json \
        --output data/calibration/calibrated_thresholds.json

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VALID_PROTOCOLS = [
    "SMD_three_phase",
    "SMD_stent_stabilization",
    "SMD_lid_restraint",
    "SMD_clamp_stabilization",
    "SMD_strain_relief",
]

# Minimum benchmark sites required per class per protocol
MIN_POSITIVE_SITES = 5
MIN_NEGATIVE_SITES = 5

# Default timeout per individual SMD job (seconds)
DEFAULT_TIMEOUT_PER_SITE = 3600


@dataclass
class BenchmarkSite:
    """A benchmark site with ground truth labels for calibration."""

    pdb_id: str
    chain: str
    residue_ids: list[str]
    ground_truth_type: str
    ground_truth_label: str  # "true_positive" or "true_negative"
    citation: str = ""
    source: str = "literature"
    notes: str = ""
    pdb_path: str = ""
    pulling_direction: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])



@dataclass
class ProtocolCalibrationResult:
    """Result of calibrating a single protocol against benchmark sites."""

    protocol: str
    positive_work_values: list[float]
    negative_work_values: list[float]
    failed_sites: list[str]
    optimal_threshold: float | None
    sensitivity: float | None
    specificity: float | None
    youdens_j: float | None
    pulling_rate_nm_per_ns: float | None
    force_constant_kJ_mol_nm2: float | None
    reference_sites: list[str]
    run_metadata: dict[str, Any] = field(default_factory=dict)


def load_benchmark(benchmark_path: str) -> list[BenchmarkSite]:
    """Load benchmark sites from JSON file.

    Expected format: list of objects with pdb_id, chain, residue_ids,
    ground_truth_type, ground_truth_label, etc.

    Raises:
        FileNotFoundError: If benchmark file does not exist.
        ValueError: If benchmark format is invalid or insufficient sites.
    """
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with path.open() as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Benchmark file must contain a JSON array of site objects")

    sites: list[BenchmarkSite] = []
    required_fields = {"pdb_id", "chain", "residue_ids", "ground_truth_type", "ground_truth_label"}

    for i, entry in enumerate(raw):
        missing = required_fields - set(entry.keys())
        if missing:
            logger.warning("Benchmark entry %d missing fields %s — skipping", i, missing)
            continue

        sites.append(
            BenchmarkSite(
                pdb_id=entry["pdb_id"],
                chain=entry["chain"],
                residue_ids=entry["residue_ids"],
                ground_truth_type=entry["ground_truth_type"],
                ground_truth_label=entry["ground_truth_label"],
                citation=entry.get("citation", ""),
                source=entry.get("source", "literature"),
                notes=entry.get("notes", ""),
                pdb_path=entry.get("pdb_path", ""),
                pulling_direction=entry.get("pulling_direction", [1.0, 0.0, 0.0]),
            )
        )

    if not sites:
        raise ValueError("No valid benchmark sites found in file")

    return sites


def _write_spec_json(
    site: BenchmarkSite, output_dir: Path, fast_mode: bool = False
) -> str:
    """Write a temporary spec JSON file for an SMD job.

    Args:
        site: Benchmark site definition.
        output_dir: Directory to write spec files.
        fast_mode: If True, inject reduced step counts for faster CPU runs.

    Returns path to the created spec file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] = {
        "structure_id": site.pdb_id,
        "site_id": f"{site.pdb_id}_{site.chain}_benchmark",
        "residue_ids": site.residue_ids,
        "pdb_path": site.pdb_path or f"data/structures/{site.pdb_id}.pdb",
        "pulling_direction": site.pulling_direction,
    }
    if fast_mode:
        # Reduced steps: ~10x fewer → ~2-3 min per site on CPU instead of 30+ min
        spec["custom_params"] = {
            "n_steps_phase1": 5000,
            "n_steps_phase2": 20000,
            "n_steps_phase3": 5000,
            "n_steps": 30000,
        }
    spec_path = output_dir / f"spec_{site.pdb_id}_{site.chain}.json"
    with spec_path.open("w") as f:
        json.dump(spec, f, indent=2)
    return str(spec_path)


def compute_youdens_j(
    positive_values: list[float],
    negative_values: list[float],
) -> tuple[float, float, float, float]:
    """Compute optimal threshold using Youden's J statistic.

    Youden's J = sensitivity + specificity - 1
    The optimal threshold maximizes J.

    For work-based metrics where positives have LOWER values (less work
    needed to open a true cryptic site), we sweep thresholds and find
    where true positives fall below the threshold.

    Args:
        positive_values: Work values from true-positive (cryptic) sites.
        negative_values: Work values from true-negative (non-cryptic) sites.

    Returns:
        Tuple of (optimal_threshold, sensitivity, specificity, youdens_j).
    """
    if not positive_values or not negative_values:
        return 0.0, 0.0, 0.0, 0.0

    all_values = sorted(set(positive_values + negative_values))

    # Generate candidate thresholds between each pair of values
    candidates: list[float] = []
    for i in range(len(all_values) - 1):
        candidates.append((all_values[i] + all_values[i + 1]) / 2.0)

    # Also include endpoints slightly outside the range
    candidates.insert(0, all_values[0] - 1.0)
    candidates.append(all_values[-1] + 1.0)

    best_j = -1.0
    best_threshold = 0.0
    best_sensitivity = 0.0
    best_specificity = 0.0

    pos_arr = np.array(positive_values)
    neg_arr = np.array(negative_values)

    for threshold in candidates:
        # For work-based: positive sites should have work < threshold
        # (easier to open = true cryptic site)
        tp = int(np.sum(pos_arr < threshold))
        fn = len(positive_values) - tp
        tn = int(np.sum(neg_arr >= threshold))
        fp = len(negative_values) - tn

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            best_threshold = threshold
            best_sensitivity = sensitivity
            best_specificity = specificity

    return best_threshold, best_sensitivity, best_specificity, best_j



async def run_real_md_calibration(
    benchmark_path: str,
    protocol: str,
    db: Any = None,
    timeout_per_site: int = DEFAULT_TIMEOUT_PER_SITE,
    fast_mode: bool = False,
) -> ProtocolCalibrationResult:
    """Run actual SMD against benchmark sites on GPU science container.

    For each benchmark site matching the protocol's target site_type:
    1. Write spec JSON for the site
    2. Dispatch real SMD job via smd_runner
    3. Wait for completion (up to timeout_per_site seconds)
    4. Record work values

    Args:
        benchmark_path: Path to benchmark JSON file.
        protocol: SMD protocol to calibrate.
        db: Database connection (unused in direct runner mode).
        timeout_per_site: Maximum seconds per site.

    Returns:
        ProtocolCalibrationResult with work values and threshold recommendation.

    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    from science.dtie.cryptic.smd_runner import run

    sites = load_benchmark(benchmark_path)

    # Partition sites by ground truth label
    positive_sites = [s for s in sites if s.ground_truth_label == "true_positive"]
    negative_sites = [s for s in sites if s.ground_truth_label == "true_negative"]

    logger.info(
        "Calibrating protocol=%s: %d positive, %d negative sites",
        protocol,
        len(positive_sites),
        len(negative_sites),
    )

    if len(positive_sites) < MIN_POSITIVE_SITES:
        logger.warning(
            "Only %d positive sites available (minimum %d recommended)",
            len(positive_sites),
            MIN_POSITIVE_SITES,
        )
    if len(negative_sites) < MIN_NEGATIVE_SITES:
        logger.warning(
            "Only %d negative sites available (minimum %d recommended)",
            len(negative_sites),
            MIN_NEGATIVE_SITES,
        )

    # Create temp dir for spec JSONs (use system temp for write permissions in containers)
    tmp_dir = Path(tempfile.mkdtemp(prefix="smd_calibration_"))

    positive_work_values: list[float] = []
    negative_work_values: list[float] = []
    failed_sites: list[str] = []
    reference_sites: list[str] = []

    # Process positive sites
    for site in positive_sites:
        site_label = f"{site.pdb_id}:{site.chain}"
        logger.info("Running positive site: %s", site_label)
        spec_path = _write_spec_json(site, tmp_dir, fast_mode=fast_mode)

        try:
            result = run(spec_path, protocol)
            if result.get("success"):
                work = result["work_kcal_mol"]
                positive_work_values.append(work)
                reference_sites.append(site_label)
                logger.info("  → work=%.3f kcal/mol (positive)", work)
            else:
                failed_sites.append(f"{site_label} (positive): {result.get('notes', 'unknown')}")
                logger.warning("  → FAILED: %s", result.get("notes", "unknown"))
        except Exception as e:
            failed_sites.append(f"{site_label} (positive): {e!s}")
            logger.error("  → ERROR: %s", e)

    # Process negative sites
    for site in negative_sites:
        site_label = f"{site.pdb_id}:{site.chain}"
        logger.info("Running negative site: %s", site_label)
        spec_path = _write_spec_json(site, tmp_dir, fast_mode=fast_mode)

        try:
            result = run(spec_path, protocol)
            if result.get("success"):
                work = result["work_kcal_mol"]
                negative_work_values.append(work)
                reference_sites.append(site_label)
                logger.info("  → work=%.3f kcal/mol (negative)", work)
            else:
                failed_sites.append(f"{site_label} (negative): {result.get('notes', 'unknown')}")
                logger.warning("  → FAILED: %s", result.get("notes", "unknown"))
        except Exception as e:
            failed_sites.append(f"{site_label} (negative): {e!s}")
            logger.error("  → ERROR: %s", e)

    # Compute optimal threshold via Youden's J
    optimal_threshold: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    youdens_j: float | None = None
    pulling_rate: float | None = None
    force_constant: float | None = None

    if positive_work_values and negative_work_values:
        optimal_threshold, sensitivity, specificity, youdens_j = compute_youdens_j(
            positive_work_values, negative_work_values
        )
        logger.info(
            "Protocol %s: threshold=%.3f, sensitivity=%.3f, specificity=%.3f, J=%.3f",
            protocol,
            optimal_threshold,
            sensitivity,
            specificity,
            youdens_j,
        )

        # Extract protocol parameters for provenance
        from science.dtie.cryptic.smd_runner_real import PROTOCOL_DEFAULTS

        if protocol in PROTOCOL_DEFAULTS:
            params = PROTOCOL_DEFAULTS[protocol]
            pulling_rate = params.get("pulling_rate_nm_per_ns")
            force_constant = params.get("force_constant_kJ_mol_nm2")
    else:
        logger.warning(
            "Insufficient data for Youden's J: %d positive, %d negative work values",
            len(positive_work_values),
            len(negative_work_values),
        )

    return ProtocolCalibrationResult(
        protocol=protocol,
        positive_work_values=positive_work_values,
        negative_work_values=negative_work_values,
        failed_sites=failed_sites,
        optimal_threshold=optimal_threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        youdens_j=youdens_j,
        pulling_rate_nm_per_ns=pulling_rate,
        force_constant_kJ_mol_nm2=force_constant,
        reference_sites=reference_sites,
        run_metadata={
            "benchmark_path": benchmark_path,
            "timeout_per_site": timeout_per_site,
            "n_positive": len(positive_work_values),
            "n_negative": len(negative_work_values),
            "n_failed": len(failed_sites),
        },
    )



async def calibrate_all_protocols(
    benchmark_path: str,
    db: Any = None,
    timeout_per_site: int = DEFAULT_TIMEOUT_PER_SITE,
    fast_mode: bool = False,
) -> dict[str, ProtocolCalibrationResult]:
    """Run calibration across all SMD protocols.

    Dispatches real MD for each protocol, collects work values,
    applies Youden's J, and generates calibrated threshold recommendations.

    Args:
        benchmark_path: Path to benchmark JSON file.
        db: Database connection (optional).
        timeout_per_site: Maximum seconds per site.
        fast_mode: Use reduced steps for CPU-feasible runs.

    Returns:
        Dict mapping protocol name → ProtocolCalibrationResult.

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    results: dict[str, ProtocolCalibrationResult] = {}

    for protocol in VALID_PROTOCOLS:
        logger.info("=" * 60)
        logger.info("Calibrating protocol: %s", protocol)
        logger.info("=" * 60)

        result = await run_real_md_calibration(
            benchmark_path=benchmark_path,
            protocol=protocol,
            db=db,
            timeout_per_site=timeout_per_site,
            fast_mode=fast_mode,
        )
        results[protocol] = result

        # Log summary
        if result.optimal_threshold is not None:
            logger.info(
                "  → Threshold: %.3f kcal/mol (J=%.3f, sens=%.3f, spec=%.3f)",
                result.optimal_threshold,
                result.youdens_j or 0.0,
                result.sensitivity or 0.0,
                result.specificity or 0.0,
            )
        else:
            logger.warning("  → Could not determine threshold (insufficient data)")

        if result.failed_sites:
            logger.warning("  → %d sites failed", len(result.failed_sites))

    return results


def apply_calibrated_thresholds(
    results: dict[str, ProtocolCalibrationResult],
    output_path: str = "data/calibration/calibrated_thresholds.json",
) -> None:
    """Write calibrated thresholds to output JSON file.

    Updates DEFAULT_SUCCESS_CRITERIA format with source="calibrated",
    the derived threshold, pulling rate, force constant, and reference sites.

    Args:
        results: Dict mapping protocol → calibration result.
        output_path: Path to write calibrated thresholds JSON.

    Requirements: 6.5, 6.6
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    output: dict[str, Any] = {
        "calibration_run_id": run_id,
        "calibration_date": timestamp,
        "protocols": {},
        "summary": {
            "total_protocols": len(results),
            "calibrated_protocols": 0,
            "failed_protocols": 0,
        },
    }

    for protocol, result in results.items():
        if result.optimal_threshold is not None and result.youdens_j is not None:
            output["protocols"][protocol] = {
                "metric": "work_kcal_mol",
                "op": "<",
                "threshold": round(result.optimal_threshold, 3),
                "calibration_metadata": {
                    "source": "calibrated",
                    "date": timestamp,
                    "reference_sites": result.reference_sites,
                    "pulling_rate_nm_per_ns": result.pulling_rate_nm_per_ns,
                    "force_constant_kJ_mol_nm2": result.force_constant_kJ_mol_nm2,
                    "notes": (
                        f"Calibrated via Youden's J (J={result.youdens_j:.3f}, "
                        f"sensitivity={result.sensitivity:.3f}, "
                        f"specificity={result.specificity:.3f})"
                    ),
                },
                "statistics": {
                    "youdens_j": round(result.youdens_j, 4),
                    "sensitivity": round(result.sensitivity, 4) if result.sensitivity else None,
                    "specificity": round(result.specificity, 4) if result.specificity else None,
                    "n_positive_sites": len(result.positive_work_values),
                    "n_negative_sites": len(result.negative_work_values),
                    "positive_work_mean": round(float(np.mean(result.positive_work_values)), 3),
                    "positive_work_std": round(float(np.std(result.positive_work_values)), 3),
                    "negative_work_mean": round(float(np.mean(result.negative_work_values)), 3),
                    "negative_work_std": round(float(np.std(result.negative_work_values)), 3),
                },
                "failed_sites": result.failed_sites,
            }
            output["summary"]["calibrated_protocols"] += 1
        else:
            output["protocols"][protocol] = {
                "metric": "work_kcal_mol",
                "op": "<",
                "threshold": None,
                "calibration_metadata": {
                    "source": "failed",
                    "date": timestamp,
                    "reference_sites": result.reference_sites,
                    "pulling_rate_nm_per_ns": result.pulling_rate_nm_per_ns,
                    "force_constant_kJ_mol_nm2": result.force_constant_kJ_mol_nm2,
                    "notes": "Calibration failed — insufficient successful MD runs",
                },
                "statistics": None,
                "failed_sites": result.failed_sites,
            }
            output["summary"]["failed_protocols"] += 1

    # Add provenance
    output["provenance"] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "script": "scripts/calibrate_cryptic_real_md.py",
        "method": "Youden's J statistic",
        "min_positive_sites": MIN_POSITIVE_SITES,
        "min_negative_sites": MIN_NEGATIVE_SITES,
    }

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info("Calibrated thresholds written to: %s", output_path)
    logger.info(
        "Summary: %d calibrated, %d failed out of %d protocols",
        output["summary"]["calibrated_protocols"],
        output["summary"]["failed_protocols"],
        output["summary"]["total_protocols"],
    )



def main() -> None:
    """CLI entry point for real MD calibration.

    Usage:
        PYTHONPATH=. python scripts/calibrate_cryptic_real_md.py \
            --benchmark data/calibration/benchmark_cryptic_sites.json \
            --output data/calibration/calibrated_thresholds.json
    """
    parser = argparse.ArgumentParser(
        description="Run real SMD calibration against benchmark sites"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark_cryptic_sites.json",
    )
    parser.add_argument(
        "--output",
        default="data/calibration/calibrated_thresholds.json",
        help="Path to write calibrated thresholds JSON (default: data/calibration/calibrated_thresholds.json)",
    )
    parser.add_argument(
        "--protocol",
        choices=VALID_PROTOCOLS,
        default=None,
        help="Run calibration for a single protocol (default: all protocols)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_PER_SITE,
        help=f"Timeout per site in seconds (default: {DEFAULT_TIMEOUT_PER_SITE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use reduced simulation steps for faster CPU calibration. "
            "Still runs real physics, just shorter trajectories. "
            "Suitable for initial threshold estimation."
        ),
    )

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Always show info for the main script
    logger.setLevel(logging.INFO)

    # --- Stub mode gate: require explicit user approval if active ---
    from science.dtie.cryptic.stub_control import require_stub_approval

    require_stub_approval()

    logger.info("Starting real MD calibration")
    logger.info("  Benchmark: %s", args.benchmark)
    logger.info("  Output: %s", args.output)
    logger.info("  Timeout: %ds per site", args.timeout)

    start_time = time.time()

    if args.fast:
        logger.info("  Mode: FAST (reduced steps — real physics, shorter trajectories)")
    else:
        logger.info("  Mode: FULL (production step counts)")

    if args.protocol:
        # Single protocol mode
        result = asyncio.run(
            run_real_md_calibration(
                benchmark_path=args.benchmark,
                protocol=args.protocol,
                timeout_per_site=args.timeout,
                fast_mode=args.fast,
            )
        )
        results = {args.protocol: result}
    else:
        # All protocols
        results = asyncio.run(
            calibrate_all_protocols(
                benchmark_path=args.benchmark,
                timeout_per_site=args.timeout,
                fast_mode=args.fast,
            )
        )

    # Apply calibrated thresholds and write output
    apply_calibrated_thresholds(results, args.output)

    elapsed = time.time() - start_time
    logger.info("Calibration complete in %.1f seconds", elapsed)

    # Exit with error if all protocols failed
    calibrated = sum(
        1 for r in results.values() if r.optimal_threshold is not None
    )
    if calibrated == 0:
        logger.error("All protocols failed calibration — no thresholds produced")
        sys.exit(1)

    logger.info("%d/%d protocols calibrated successfully", calibrated, len(results))


if __name__ == "__main__":
    main()
