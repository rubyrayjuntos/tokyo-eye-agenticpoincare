/**
 * Adapts backend Poincaré residue data into the NodeData/EdgeData format
 * used by the 3D/2D visualizers.
 */

import * as THREE from "three";
import type { NodeData, EdgeData } from "./math";
import { hyperbolicDistance } from "./math";
import type { PoincareResidue, PoincareDataResponse } from "./agentClient";

export type MetricKey = "cone_depth" | "epistemic" | "aleatoric" | "uncertainty";

/**
 * Convert agent PoincareDataResponse into visualizer-ready nodes and edges.
 */
export function adaptPoincareData(
  response: PoincareDataResponse,
  colorMetric: MetricKey = "cone_depth"
): { nodes: NodeData[]; edges: EdgeData[] } {
  const nodes = response.residues.map((r) => toNodeData(r, colorMetric));

  // Build edges from k-nearest using hyperbolic distance (consistent with math.ts)
  const edges: EdgeData[] = [];
  const k = 4;
  for (let i = 0; i < nodes.length; i++) {
    const dists = nodes.map((n, idx) => ({
      idx,
      d: hyperbolicDistance(nodes[i].position, n.position),
    })).sort((a, b) => a.d - b.d);

    for (let j = 1; j <= k; j++) {
      if (dists[j] && dists[j].idx > i) {
        edges.push({
          source: nodes[i].id,
          target: nodes[dists[j].idx].id,
          distance: dists[j].d,
        });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Re-color existing nodes based on a different metric from the residue data.
 */
export function recolorNodes(
  nodes: NodeData[],
  residues: PoincareResidue[],
  colorMetric: MetricKey = "cone_depth"
): NodeData[] {
  const residueMap = new Map(residues.map((r) => [r.residue_id, r]));

  return nodes.map((n) => {
    const r = residueMap.get(n.id);
    if (!r) return n;
    const metricValue = getMetricValue(r, colorMetric);
    const hue = metricToHue(metricValue);
    const isOutlier = r.cone_depth > 0.8 && r.epistemic_uncertainty > 0.6;
    return {
      ...n,
      color: `hsl(${hue}, 70%, ${isOutlier ? 60 : 50}%)`,
      isOutlier,
    };
  });
}

function toNodeData(r: PoincareResidue, colorMetric: MetricKey): NodeData {
  const metricValue = getMetricValue(r, colorMetric);
  const hue = metricToHue(metricValue);
  const isOutlier = r.cone_depth > 0.8 && r.epistemic_uncertainty > 0.6;

  // Backend provides 2D disc coordinates (x, y).
  // Lift to 3D ball: use (x, y, 0) as the position in 3D Poincaré ball.
  const pos3D = new THREE.Vector3(r.x, r.y, 0);
  if (pos3D.length() > 0.95) pos3D.setLength(0.95);

  const pos2D: [number, number] = [r.x, r.y];

  return {
    id: r.residue_id,
    position: pos3D,
    position2D: pos2D,
    color: `hsl(${hue}, 70%, ${isOutlier ? 60 : 50}%)`,
    domain: r.chain_label,
    isOutlier,
    value: r.epistemic_uncertainty,
    depth: r.cone_depth,
    expertId: 0,
  };
}

function getMetricValue(r: PoincareResidue, metric: MetricKey): number {
  switch (metric) {
    case "cone_depth": return r.cone_depth;
    case "epistemic": return r.epistemic_uncertainty;
    case "aleatoric": return r.aleatoric_uncertainty;
    case "uncertainty": return (r.epistemic_uncertainty + r.aleatoric_uncertainty) / 2;
    default: return r.cone_depth;
  }
}

function metricToHue(value: number): number {
  // Map 0..1 → blue (200) → magenta (360)
  return Math.round(200 + Math.min(value, 1) * 160);
}
