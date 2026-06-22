import * as THREE from "three";
import type { NodeData, EdgeData } from "./math";
import { depthToColor, generateMockGNNData } from "./math";

const API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000";

export interface StructureMetadata {
  pdb_id: string;
  gene?: string;
  description?: string;
  curvature?: number;
  model_version?: string;
  residue_count?: number;
}

export interface LoadedData {
  nodes: NodeData[];
  edges: EdgeData[];
  metadata?: StructureMetadata;
}

// Attempt to load from the agent REST endpoint, fall back to mock
export async function loadGNNData(pdbId: string): Promise<LoadedData> {
  try {
    const res = await fetch(
      `${API_BASE}/api/structures/${pdbId}/embeddings`,
      { signal: AbortSignal.timeout(5000) }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return parseEmbeddingResponse(data, pdbId);
  } catch {
    return { ...generateMockGNNData(), metadata: undefined };
  }
}

export async function loadGNNDataFromFile(file: File): Promise<LoadedData> {
  const text = await file.text();
  const json = JSON.parse(text);
  if (json.residues) return parseEmbeddingResponse(json, json.pdb_id ?? file.name);
  if (json.nodes && json.edges) return json as LoadedData;
  throw new Error("Unrecognised file format");
}

function parseEmbeddingResponse(data: Record<string, unknown>, pdbId: string): LoadedData {
  const rawResidues = (data.residues as Record<string, unknown>[]) ?? [];
  const curvature = (data.curvature as number) ?? 1.0;

  const nodes: NodeData[] = rawResidues.map((r) => {
    const x = Number(r.x ?? 0);
    const y = Number(r.y ?? 0);
    const depth = Number(r.cone_depth ?? 0);
    const epistemic = Number(r.epistemic_uncertainty ?? 0);
    const aleatoric = Number(r.aleatoric_uncertainty ?? 0);
    // Reconstruct z from depth and (x, y) — approximate equatorial lift
    const r2 = Math.sqrt(Math.max(0, depth * depth - x * x - y * y));

    return {
      id: String(r.residue_id ?? ""),
      position: new THREE.Vector3(x, y, r2),
      position2D: [x, y] as [number, number],
      color: depthToColor(depth),
      depth,
      value: depth,
      isOutlier: depth > 0.75 && epistemic > 0.5,
      domain: String(r.domain ?? r.chain_label ?? ""),
      label: String(r.residue_name ?? ""),
      epistemic,
      aleatoric,
    };
  });

  // Build sparse contact edges from spatial proximity (no edge data from API yet)
  const edges: EdgeData[] = buildProximityEdges(nodes, 0.25);

  return {
    nodes,
    edges,
    metadata: {
      pdb_id: pdbId,
      curvature,
      model_version: String(data.model_version ?? "v6"),
      residue_count: nodes.length,
    },
  };
}

function buildProximityEdges(nodes: NodeData[], threshold: number): EdgeData[] {
  const edges: EdgeData[] = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].position2D[0] - nodes[j].position2D[0];
      const dy = nodes[i].position2D[1] - nodes[j].position2D[1];
      if (dx * dx + dy * dy < threshold * threshold) {
        edges.push({ source: nodes[i].id, target: nodes[j].id });
      }
    }
  }
  return edges;
}
