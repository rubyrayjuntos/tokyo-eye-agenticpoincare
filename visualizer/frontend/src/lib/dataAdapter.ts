import * as THREE from "three";
import type { NodeData, EdgeData } from "./math";
import { depthToColor, uncertaintyToColor } from "./math";
import type { PoincareDataResponse, PoincareResidue } from "./agentClient";

export type ColorMetric = "cone_depth" | "epistemic" | "aleatoric" | "centrality";

export interface AdaptedData {
  nodes: NodeData[];
  edges: EdgeData[];
}

/**
 * Convert a raw PoincareDataResponse (from /api/structures/:id/embeddings)
 * into NodeData/EdgeData ready for the visualizers.
 *
 * High SASA residues → high cone_depth → disc periphery (|pos| near 1).
 * This matches the corrected geometry (r=0.738 checkpoint).
 */
export function adaptPoincareData(
  response: PoincareDataResponse,
  metric: ColorMetric = "cone_depth"
): AdaptedData {
  const nodes: NodeData[] = response.residues.map((r) => {
    const depth = clamp(r.cone_depth, 0, 1);
    const epistemic = clamp(r.epistemic_uncertainty, 0, 1);
    const aleatoric = clamp(r.aleatoric_uncertainty, 0, 1);

    // Disc position (x, y) already normalised to [-1, 1] from backend
    const x = clamp(r.x, -1, 1);
    const y = clamp(r.y, -1, 1);

    // Lift into ball: z from residual radius so |pos|² = depth²
    const rxy2 = x * x + y * y;
    const zSq = Math.max(0, depth * depth - rxy2);
    const z = Math.sqrt(zSq);

    return {
      id: r.residue_id,
      position: new THREE.Vector3(x, y, z),
      position2D: [x, y],
      color: colorForMetric(r, metric),
      depth,
      value: metricValue(r, metric),
      isOutlier: depth > 0.75 && epistemic > 0.5,
      domain: r.chain_label ?? "",
      label: r.residue_name ?? "",
      epistemic,
      aleatoric,
    };
  });

  // Build proximity contact edges on the disc
  const edges = buildEdges(nodes, response.residues);

  return { nodes, edges };
}

/**
 * Re-color existing nodes when the user switches metric tabs.
 * Returns a new array (does not mutate).
 */
export function recolorNodes(
  nodes: NodeData[],
  residues: PoincareResidue[],
  metric: ColorMetric
): NodeData[] {
  const residueMap = new Map<string, PoincareResidue>(
    residues.map((r) => [r.residue_id, r])
  );

  return nodes.map((node) => {
    const r = residueMap.get(node.id);
    if (!r) return node;
    return {
      ...node,
      color: colorForMetric(r, metric),
      value: metricValue(r, metric),
    };
  });
}

// ---------- Helpers ----------

function colorForMetric(r: PoincareResidue, metric: ColorMetric): string {
  switch (metric) {
    case "cone_depth":
      return depthToColor(r.cone_depth);
    case "epistemic":
      return uncertaintyToColor(r.epistemic_uncertainty);
    case "aleatoric":
      return uncertaintyToColor(r.aleatoric_uncertainty);
    default:
      return depthToColor(r.cone_depth);
  }
}

function metricValue(r: PoincareResidue, metric: ColorMetric): number {
  switch (metric) {
    case "cone_depth":      return r.cone_depth;
    case "epistemic":       return r.epistemic_uncertainty;
    case "aleatoric":       return r.aleatoric_uncertainty;
    default:                return r.cone_depth;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v ?? 0));
}

function buildEdges(nodes: NodeData[], residues: PoincareResidue[]): EdgeData[] {
  // Sequential residue connectivity (backbone) + short-range disc proximity
  const edges: EdgeData[] = [];
  const threshold = 0.2;

  for (let i = 0; i + 1 < residues.length; i++) {
    edges.push({ source: nodes[i].id, target: nodes[i + 1].id, weight: 1.0 });
  }

  // Spatial edges (contact-map approximation on disc)
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 2; j < nodes.length; j++) {
      const dx = nodes[i].position2D[0] - nodes[j].position2D[0];
      const dy = nodes[i].position2D[1] - nodes[j].position2D[1];
      if (dx * dx + dy * dy < threshold * threshold) {
        edges.push({ source: nodes[i].id, target: nodes[j].id, weight: 0.5 });
      }
    }
  }

  return edges;
}
