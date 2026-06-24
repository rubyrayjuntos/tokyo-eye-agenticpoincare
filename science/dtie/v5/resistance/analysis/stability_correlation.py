"""Orthogonal Validation: Stability Correlation Analysis.

Correlates Resistance Profiler classification results with thermodynamic
stability predictions (FoldX ΔΔG or Rosetta ddG) to flag potential
misfolding artifacts.

A mutation classified as "Type_II_Allosteric" that also shows extreme
destabilization (ΔΔG > 2.0 kcal/mol) is flagged as a "Misfolding Candidate"
rather than a true resistance mechanism. This ensures the pipeline isn't
just detecting protein-breaking mutations.

Usage:
    from science.dtie.v5.resistance.analysis.stability_correlation import (
        analyze_stability_correlation,
        flag_misfolding_candidates,
    )

    # After running benchmark and FoldX
    candidates = analyze_stability_correlation(
        resistance_json_path="tests/benchmarks/results/benchmark_1iep.json",
        stability_csv_path="data/foldx_results.csv",
        output_path="plots/stability_correlation.png",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StabilityCorrelationResult:
    """Result of correlating resistance classification with stability data."""

    variant: str
    mechanism_class: str
    confidence: float
    ddg: float
    is_misfolding_candidate: bool
    site_delta: float
    hub_propagation_sum: float


# Destabilization threshold (kcal/mol) above which a mutation is likely
# causing misfolding rather than specific resistance
DDG_DESTABILIZATION_THRESHOLD = 2.0


def load_resistance_results(resistance_json_path: str | Path) -> list[dict]:
    """Load resistance profiler benchmark results."""
    path = Path(resistance_json_path)
    with open(path) as f:
        data = json.load(f)

    # Handle both benchmark runner output and raw BatchReport format
    if "gold_standard" in data:
        return data["gold_standard"]
    elif "reports" in data:
        return [
            {
                "variant": r.get("variant", ""),
                "predicted": r.get("mechanism_class", "Unknown"),
                "confidence": r.get("confidence_score", 0.0),
                "site_delta": r.get("metrics", {}).get("site_uncertainty_delta", 0.0),
                "hub_propagation_sum": sum(
                    abs(h.get("delta_epistemic", 0.0))
                    for h in r.get("hub_details", [])
                ),
            }
            for r in data["reports"]
        ]
    else:
        raise ValueError(f"Unrecognized format in {path}")


def load_stability_data(stability_csv_path: str | Path) -> dict[str, float]:
    """Load stability predictions from CSV.

    Expected CSV format:
        variant,ddG
        T315I,0.8
        E255K,2.3
        ...

    Returns:
        Dict mapping variant name to ΔΔG value.
    """
    path = Path(stability_csv_path)
    results: dict[str, float] = {}

    with open(path) as f:
        header = f.readline().strip().split(",")
        variant_col = header.index("variant")
        ddg_col = header.index("ddG")

        for line in f:
            parts = line.strip().split(",")
            if len(parts) > max(variant_col, ddg_col):
                variant = parts[variant_col].strip()
                try:
                    ddg = float(parts[ddg_col].strip())
                    results[variant] = ddg
                except ValueError:
                    continue

    return results


def flag_misfolding_candidates(
    resistance_results: list[dict],
    stability_data: dict[str, float],
    ddg_threshold: float = DDG_DESTABILIZATION_THRESHOLD,
) -> list[StabilityCorrelationResult]:
    """Identify mutations that may be misfolding artifacts.

    A mutation is flagged if:
    - It is classified as Type_II_Allosteric (or Hybrid with allosteric component)
    - Its ΔΔG exceeds the destabilization threshold

    Args:
        resistance_results: List of dicts from benchmark runner output.
        stability_data: Dict mapping variant → ΔΔG.
        ddg_threshold: ΔΔG threshold for flagging (default 2.0 kcal/mol).

    Returns:
        List of StabilityCorrelationResult for all matched mutations.
    """
    results: list[StabilityCorrelationResult] = []

    for r in resistance_results:
        variant = r.get("variant", "")
        if variant not in stability_data:
            continue

        ddg = stability_data[variant]
        mechanism = r.get("predicted", r.get("mechanism_class", "Unknown"))
        confidence = r.get("confidence", r.get("confidence_score", 0.0))
        site_delta = r.get("site_delta", 0.0)
        hub_sum = r.get("hub_propagation_sum", 0.0)

        # Flag if allosteric classification + high destabilization
        is_candidate = (
            mechanism in ("Type_II_Allosteric", "Hybrid")
            and ddg > ddg_threshold
        )

        results.append(
            StabilityCorrelationResult(
                variant=variant,
                mechanism_class=mechanism,
                confidence=confidence,
                ddg=ddg,
                is_misfolding_candidate=is_candidate,
                site_delta=site_delta,
                hub_propagation_sum=hub_sum,
            )
        )

    return results


def analyze_stability_correlation(
    resistance_json_path: str | Path,
    stability_csv_path: str | Path,
    output_path: str | Path = "plots/stability_correlation.png",
    ddg_threshold: float = DDG_DESTABILIZATION_THRESHOLD,
) -> list[StabilityCorrelationResult]:
    """Full correlation analysis with optional plot generation.

    Correlates Resistance Profiler 'Allosteric' class calls with stability
    scores to flag potential misfolding artifacts.

    Args:
        resistance_json_path: Path to benchmark runner JSON output.
        stability_csv_path: Path to CSV with columns [variant, ddG].
        output_path: Path for the scatter plot output.
        ddg_threshold: ΔΔG threshold for misfolding flag.

    Returns:
        List of StabilityCorrelationResult (misfolding candidates highlighted).
    """
    resistance_results = load_resistance_results(resistance_json_path)
    stability_data = load_stability_data(stability_csv_path)

    correlations = flag_misfolding_candidates(
        resistance_results, stability_data, ddg_threshold
    )

    # Attempt to generate plot (optional dependency)
    try:
        _generate_plot(correlations, output_path, ddg_threshold)
    except ImportError:
        print(
            "matplotlib/seaborn not available — skipping plot generation. "
            "Install with: pip install matplotlib seaborn"
        )

    # Report
    candidates = [c for c in correlations if c.is_misfolding_candidate]
    print(f"\nStability Correlation Analysis:")
    print(f"  Total mutations with stability data: {len(correlations)}")
    print(f"  Misfolding candidates flagged: {len(candidates)}")

    if candidates:
        print(f"\n  ⚠ Potential misfolding artifacts (ΔΔG > {ddg_threshold} kcal/mol):")
        for c in candidates:
            print(
                f"    {c.variant}: {c.mechanism_class} (conf={c.confidence:.3f}) "
                f"| ΔΔG={c.ddg:.2f} kcal/mol"
            )
        print(
            "\n  These mutations may be causing protein destabilization rather "
            "than specific resistance mechanisms. Validate with experimental "
            "expression/folding data."
        )

    return correlations


def _generate_plot(
    correlations: list[StabilityCorrelationResult],
    output_path: str | Path,
    ddg_threshold: float,
) -> None:
    """Generate the stability correlation scatter plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame([
        {
            "variant": c.variant,
            "mechanism_class": c.mechanism_class,
            "confidence": c.confidence,
            "ddG": c.ddg,
            "misfolding_candidate": c.is_misfolding_candidate,
        }
        for c in correlations
    ])

    if df.empty:
        print("No data to plot.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="ddG",
        y="confidence",
        hue="mechanism_class",
        style="mechanism_class",
        s=120,
    )

    # Destabilization threshold line
    plt.axvline(
        x=ddg_threshold, color="r", linestyle="--",
        label=f"Destabilization Threshold ({ddg_threshold} kcal/mol)",
    )

    # Annotate misfolding candidates
    for _, row in df[df["misfolding_candidate"]].iterrows():
        plt.annotate(
            row["variant"],
            (row["ddG"], row["confidence"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    plt.title("Resistance Mechanism vs. Thermodynamic Stability")
    plt.xlabel("FoldX ΔΔG (kcal/mol)")
    plt.ylabel("Classification Confidence Score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Plot saved to: {output_path}")
