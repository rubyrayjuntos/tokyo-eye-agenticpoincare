import type {
  DrugCandidateData,
  GraphMetricsData,
  PharmacophoreData,
  ResistanceData,
  ResidueEmbedding,
  StructureColorModeType,
} from "./types";

/** All nine structure / viewport color codings (matches MolecularViewer). */
export type ViewportColorMode = StructureColorModeType;

export const VIEWPORT_COLOR_MODE_OPTIONS: { value: ViewportColorMode; label: string }[] = [
  { value: "spectrum", label: "Spectrum" },
  { value: "cone_depth", label: "Cone Depth" },
  { value: "epistemic", label: "Epistemic" },
  { value: "aleatoric", label: "Aleatoric" },
  { value: "plasticity", label: "Plasticity" },
  { value: "allosteric", label: "Allosteric" },
  { value: "resistance", label: "Resistance" },
  { value: "pockets", label: "Pockets" },
  { value: "drug_candidates", label: "Drug Candidates" },
];

export interface ViewportColorContext {
  graphMetrics: GraphMetricsData | null;
  resistanceData: ResistanceData | null;
  pharmacophorePockets: PharmacophoreData | null;
  drugCandidates: DrugCandidateData | null;
  plasticityDepthRange: { min: number; max: number } | null;
  plasticityUncertRange: { min: number; max: number } | null;
}

export interface ResidueMetricSeries {
  values: Map<string, number>;
  min: number;
  max: number;
  /** False when hydration-backed metrics are missing for this mode. */
  dataAvailable: boolean;
}

export function computePlasticityRisk(
  coneDepth: number,
  uncertainty: number,
  dMin: number,
  dMax: number,
  uMin: number,
  uMax: number,
): number {
  const normDepth = (coneDepth - dMin) / (dMax - dMin + 1e-8);
  const normUncert = (uncertainty - uMin) / (uMax - uMin + 1e-8);
  return Math.pow((1.0 - normDepth) * 0.55 + normUncert * 0.45, 1.3);
}

function buildPlasticityRanges(residues: ResidueEmbedding[]) {
  if (!residues.length) return { depth: null, uncert: null };
  let dMin = Infinity;
  let dMax = -Infinity;
  let uMin = Infinity;
  let uMax = -Infinity;
  for (const r of residues) {
    dMin = Math.min(dMin, r.cone_depth);
    dMax = Math.max(dMax, r.cone_depth);
    uMin = Math.min(uMin, r.epistemic_uncertainty);
    uMax = Math.max(uMax, r.epistemic_uncertainty);
  }
  return {
    depth: { min: dMin, max: dMax },
    uncert: { min: uMin, max: uMax },
  };
}

function buildAllostericScores(
  residues: ResidueEmbedding[],
  graphMetrics: GraphMetricsData | null,
): number[] {
  const metricsMap = new Map((graphMetrics?.metrics ?? []).map((m) => [m.residue_id, m]));
  return residues.map((residue) => {
    const gm = metricsMap.get(residue.residue_id);
    const betweenness = gm?.betweenness ?? 0;
    return Math.sqrt(betweenness) * Math.pow(residue.epistemic_uncertainty + 0.01, 0.8);
  });
}

function buildPocketLookup(pharmacophorePockets: PharmacophoreData | null) {
  const map = new Map<string, number>();
  for (const pocket of pharmacophorePockets?.pockets ?? []) {
    for (const rid of pocket.residue_ids) {
      map.set(rid, pocket.druggability_score);
    }
  }
  return map;
}

function buildDrugCandidateLookup(
  pharmacophorePockets: PharmacophoreData | null,
  drugCandidates: DrugCandidateData | null,
) {
  const candidateByPocket = new Map<number, number>();
  for (const c of drugCandidates?.candidates ?? []) {
    candidateByPocket.set(c.pocket_index, c.combined_druggability);
  }
  const map = new Map<string, number>();
  for (const pocket of pharmacophorePockets?.pockets ?? []) {
    const score = candidateByPocket.get(pocket.pocket_index);
    if (score == null) continue;
    for (const rid of pocket.residue_ids) {
      map.set(rid, score);
    }
  }
  return map;
}

export function buildResidueMetricSeries(
  residues: ResidueEmbedding[],
  mode: ViewportColorMode,
  ctx: ViewportColorContext,
): ResidueMetricSeries {
  const values = new Map<string, number>();
  if (!residues.length) {
    return { values, min: 0, max: 1, dataAvailable: true };
  }

  const plasticityRanges = buildPlasticityRanges(residues);
  const allostericScores =
    mode === "allosteric" ? buildAllostericScores(residues, ctx.graphMetrics) : [];
  const resistanceMap = new Map(
    (ctx.resistanceData?.residues ?? []).map((r) => [r.residue_id, r.sensitivity_score]),
  );
  const pocketMap = buildPocketLookup(ctx.pharmacophorePockets);
  const drugMap = buildDrugCandidateLookup(ctx.pharmacophorePockets, ctx.drugCandidates);

  let dataAvailable = true;
  const indexMax = Math.max(...residues.map((r) => r.residue_index), 1);

  residues.forEach((residue, i) => {
    let value = 0;
    switch (mode) {
      case "spectrum":
        value = residue.residue_index / indexMax;
        break;
      case "cone_depth":
        value = residue.cone_depth;
        break;
      case "epistemic":
        value = residue.epistemic_uncertainty;
        break;
      case "aleatoric":
        value = residue.aleatoric_uncertainty;
        break;
      case "plasticity":
        if (!plasticityRanges.depth || !plasticityRanges.uncert) {
          dataAvailable = false;
          value = 0;
        } else {
          value = computePlasticityRisk(
            residue.cone_depth,
            residue.epistemic_uncertainty,
            plasticityRanges.depth.min,
            plasticityRanges.depth.max,
            plasticityRanges.uncert.min,
            plasticityRanges.uncert.max,
          );
        }
        break;
      case "allosteric":
        if (!ctx.graphMetrics?.metrics?.length) {
          dataAvailable = false;
          value = 0;
        } else {
          value = allostericScores[i] ?? 0;
        }
        break;
      case "resistance":
        if (!ctx.resistanceData?.residues?.length) {
          dataAvailable = false;
          value = 0;
        } else {
          value = resistanceMap.get(residue.residue_id) ?? 0;
        }
        break;
      case "pockets":
        if (!ctx.pharmacophorePockets?.pockets?.length) {
          dataAvailable = false;
          value = 0;
        } else {
          value = pocketMap.get(residue.residue_id) ?? 0;
        }
        break;
      case "drug_candidates":
        if (!drugMap.size) {
          dataAvailable = false;
          value = 0;
        } else {
          value = drugMap.get(residue.residue_id) ?? 0;
        }
        break;
      default:
        value = residue.cone_depth;
    }
    values.set(residue.residue_id, value);
  });

  const nums = [...values.values()];
  const min = nums.length ? Math.min(...nums) : 0;
  const max = nums.length ? Math.max(...nums) : 1;

  return { values, min, max, dataAvailable };
}

function mixRgb(a: [number, number, number], b: [number, number, number], t: number): string {
  const k = Math.max(0, Math.min(1, t));
  const r = Math.round(a[0] + (b[0] - a[0]) * k);
  const g = Math.round(a[1] + (b[1] - a[1]) * k);
  const bl = Math.round(a[2] + (b[2] - a[2]) * k);
  return `rgb(${r},${g},${bl})`;
}

function normalizeT(value: number, min: number, max: number): number {
  const span = max - min || 1;
  return Math.max(0, Math.min(1, (value - min) / span));
}

function plasticityRgb(t: number): string {
  if (t < 0.4) {
    const s = t / 0.4;
    return mixRgb([40, 80, 220], [180, 180, 200], s);
  }
  if (t < 0.7) {
    const s = (t - 0.4) / 0.3;
    return mixRgb([180, 180, 200], [255, 140, 30], s);
  }
  const s = (t - 0.7) / 0.3;
  return mixRgb([255, 140, 30], [255, 50, 20], s);
}

function allostericRgb(t: number): string {
  if (t < 0.5) {
    const s = t * 2;
    return mixRgb([20, 80, 40], [0, 220, 220], s);
  }
  const s = (t - 0.5) * 2;
  return mixRgb([0, 220, 220], [220, 40, 255], s);
}

function resistanceRgb(t: number): string {
  if (t < 0.3) {
    const s = t / 0.3;
    return mixRgb([240, 240, 240], [255, 220, 100], s);
  }
  if (t < 0.6) {
    const s = (t - 0.3) / 0.3;
    return mixRgb([255, 220, 100], [255, 140, 20], s);
  }
  const s = (t - 0.6) / 0.4;
  return mixRgb([255, 140, 20], [180, 20, 10], s);
}

function pocketRgb(t: number): string {
  if (t < 0.5) {
    const s = t * 2;
    return mixRgb([40, 200, 80], [240, 220, 30], s);
  }
  const s = (t - 0.5) * 2;
  return mixRgb([240, 220, 30], [255, 60, 30], s);
}

function drugCandidateRgb(t: number): string {
  if (t < 0.4) return mixRgb([40, 160, 180], [40, 160, 180], 0);
  if (t < 0.7) {
    const s = (t - 0.4) / 0.3;
    return mixRgb([40, 160, 180], [220, 200, 40], s);
  }
  const s = (t - 0.7) / 0.3;
  return mixRgb([220, 200, 40], [255, 200, 30], s);
}

function spectrumRgb(t: number): string {
  const hue = Math.round(t * 300);
  return `hsl(${hue}, 78%, 58%)`;
}

function viridisRgb(t: number): string {
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + (t < 0.5 ? t * 2 * (168 - 84) : (1 - t) * 2 * 168));
  return `rgb(${r},${g},${b})`;
}

function coneDepthRgb(t: number): string {
  return mixRgb([10, 98, 91], [153, 240, 220], t);
}

function epistemicRgb(t: number): string {
  return mixRgb([44, 49, 66], [236, 165, 58], t);
}

function aleatoricRgb(t: number): string {
  return mixRgb([44, 49, 66], [220, 92, 233], t);
}

export function metricValueToColor(
  value: number,
  mode: ViewportColorMode,
  min: number,
  max: number,
): string {
  const t = normalizeT(value, min, max);
  switch (mode) {
    case "spectrum":
      return spectrumRgb(t);
    case "cone_depth":
      return coneDepthRgb(t);
    case "epistemic":
      return epistemicRgb(t);
    case "aleatoric":
      return aleatoricRgb(t);
    case "plasticity":
      return plasticityRgb(t);
    case "allosteric":
      return allostericRgb(t);
    case "resistance":
      return resistanceRgb(t);
    case "pockets":
      return pocketRgb(t);
    case "drug_candidates":
      return drugCandidateRgb(t);
    default:
      return viridisRgb(t);
  }
}

/** @deprecated Use metricValueToColor — kept for poincareViewportMath callers. */
export function metricColor(
  value: number,
  mode: ViewportColorMode,
  min: number,
  max: number,
): string {
  return metricValueToColor(value, mode, min, max);
}

/** @deprecated Use buildResidueMetricSeries — kept for legacy imports. */
export function metricValue(
  residue: ResidueEmbedding,
  mode: ViewportColorMode,
): number {
  switch (mode) {
    case "spectrum":
      return residue.residue_index;
    case "cone_depth":
      return residue.cone_depth;
    case "epistemic":
      return residue.epistemic_uncertainty;
    case "aleatoric":
      return residue.aleatoric_uncertainty;
    default:
      return residue.cone_depth;
  }
}
