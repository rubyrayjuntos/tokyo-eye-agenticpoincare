import { useMemo } from "react";

import { useHydration } from "../context/HydrationProvider";
import {
  buildResidueMetricSeries,
  type ResidueMetricSeries,
  type ViewportColorMode,
} from "../lib/viewportColorMetrics";
import type { ResidueEmbedding } from "../lib/types";
import { useEmbeddingResidues } from "./useEmbeddingResidues";

export function useResidueMetricSeries(
  structureId: string | null,
  colorMode: ViewportColorMode,
  residuesOverride?: ResidueEmbedding[],
): {
  residues: ResidueEmbedding[];
  loading: boolean;
  error: string | null;
  metricSeries: ResidueMetricSeries;
} {
  const { residues, loading, error } = useEmbeddingResidues(structureId);
  const {
    graphMetrics,
    resistanceData,
    pharmacophorePockets,
    drugCandidates,
  } = useHydration();

  const effectiveResidues = residuesOverride ?? residues;

  const metricSeries = useMemo(
    () =>
      buildResidueMetricSeries(effectiveResidues, colorMode, {
        graphMetrics,
        resistanceData,
        pharmacophorePockets,
        drugCandidates,
        plasticityDepthRange: null,
        plasticityUncertRange: null,
      }),
    [
      effectiveResidues,
      colorMode,
      graphMetrics,
      resistanceData,
      pharmacophorePockets,
      drugCandidates,
    ],
  );

  return { residues: effectiveResidues, loading, error, metricSeries };
}
