export interface UncertaintySummary {
  mean: number;
  median: number;
  p90: number;
  max: number;
  spikeCount: number;
  spikeResidueIds: string[];
  hasLocalizedSpikes: boolean;
  coefficientOfVariation: number;
}

export function summarizeUncertainty(
  residues: Array<{ residue_id: string; epistemic_uncertainty: number }>,
): UncertaintySummary {
  if (residues.length === 0) {
    return {
      mean: 0,
      median: 0,
      p90: 0,
      max: 0,
      spikeCount: 0,
      spikeResidueIds: [],
      hasLocalizedSpikes: false,
      coefficientOfVariation: 0,
    };
  }

  const values = residues.map((r) => r.epistemic_uncertainty);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const p90 = sorted[Math.floor(sorted.length * 0.9)] ?? median;
  const max = sorted[sorted.length - 1] ?? 0;
  const std =
    values.length > 1
      ? Math.sqrt(
          values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length,
        )
      : 0;
  const coefficientOfVariation = mean > 0 ? std / mean : 0;

  // Localized spikes: exceed both p90 and 15% above median (scale-agnostic).
  const spikeThreshold = Math.max(p90, median * 1.15);
  const spikes = residues.filter((r) => r.epistemic_uncertainty >= spikeThreshold);
  const spikeResidueIds = spikes
    .sort((a, b) => b.epistemic_uncertainty - a.epistemic_uncertainty)
    .slice(0, 5)
    .map((r) => r.residue_id);

  return {
    mean,
    median,
    p90,
    max,
    spikeCount: spikes.length,
    spikeResidueIds,
    hasLocalizedSpikes: spikes.length > 0 && spikes.length < residues.length * 0.25,
    coefficientOfVariation,
  };
}

export function proteinStateLabel(summary: UncertaintySummary): {
  title: string;
  description: string;
} {
  if (summary.spikeCount === 0) {
    return {
      title: "Uniform uncertainty field",
      description:
        "No residue exceeds the 90th-percentile uncertainty threshold for this structure.",
    };
  }
  if (summary.hasLocalizedSpikes) {
    return {
      title: "Localized uncertainty peaks",
      description: `${summary.spikeCount} residues exceed the structure's 90th-percentile epistemic uncertainty.`,
    };
  }
  return {
    title: "Broad uncertainty elevation",
    description:
      "Uncertainty is elevated across many residues rather than concentrated in a small cluster.",
  };
}

export function formatLambda2(lambda2: number | null | undefined): {
  display: string;
  note: string | null;
} {
  if (lambda2 == null || Number.isNaN(lambda2)) {
    return { display: "n/a", note: null };
  }
  if (lambda2 > 10) {
    return {
      display: lambda2.toFixed(2),
      note: "Legacy raw Laplacian λ₂ from an older run. Re-run resistance mapping for normalized values in [0, 2].",
    };
  }
  return { display: lambda2.toFixed(3), note: null };
}
