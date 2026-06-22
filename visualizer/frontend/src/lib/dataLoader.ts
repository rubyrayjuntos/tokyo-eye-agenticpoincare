import * as THREE from "three";
import type { NodeData, EdgeData } from "./math";
import { hyperbolicDistance, generateMockGNNData } from "./math";

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

  const nodes: NodeData[] = rawResidues.map((r, i) => {
    const x = Number(r.x ?? 0);
    const y = Number(r.y ?? 0);
    const depth = Number(r.cone_depth ?? 0);
    const pos3D = new THREE.Vector3(x, y, 0);
    if (pos3D.length() > 0.95) pos3D.setLength(0.95);

    return {
      id: String(r.residue_id ?? `res_${i}`),
      position: pos3D,
      position2D: [x, y] as [number, number],
      color: `hsl(${Math.round(200 + depth * 160)}, 70%, 50%)`,
      domain: String(r.chain_label ?? ""),
      isOutlier: depth > 0.8 && Number(r.epistemic_uncertainty ?? 0) > 0.6,
      value: Number(r.epistemic_uncertainty ?? 0),
      depth,
      expertId: 0,
    };
  });

  const edges = buildKnnEdges(nodes, 4);

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

function buildKnnEdges(nodes: NodeData[], k: number): EdgeData[] {
  const edges: EdgeData[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const dists = nodes
      .map((n, idx) => ({ idx, d: hyperbolicDistance(nodes[i].position, n.position) }))
      .sort((a, b) => a.d - b.d);
    for (let j = 1; j <= k; j++) {
      if (dists[j] && dists[j].idx > i) {
        edges.push({ source: nodes[i].id, target: nodes[dists[j].idx].id, distance: dists[j].d });
      }
    }
  }
  return edges;
}
