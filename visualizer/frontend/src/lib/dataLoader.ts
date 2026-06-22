/**
 * Data loader for Tokyo Eyes v5 viewer.
 * Priority: live server → static JSON → mock data.
 */
import * as THREE from 'three';
import { NodeData, EdgeData, generateMockGNNData } from './math';

interface ExportedNode {
  id: string;
  position: [number, number, number];
  position2D: [number, number];
  color: string;
  domain: string;
  isOutlier: boolean;
  value: number;
  depth: number;
  expertId: number;
}

interface ExportedEdge {
  source: string;
  target: string;
  distance: number;
}

interface ExportedData {
  metadata: {
    pdb_id: string;
    chain: string;
    curvature: number;
    version: string;
    architecture: string;
    n_residues: number;
    radial_scale: number;
    depth_range: [number, number];
    depth_std: number;
  };
  nodes: ExportedNode[];
  edges: ExportedEdge[];
}

export interface LoadedData {
  nodes: NodeData[];
  edges: EdgeData[];
  metadata?: ExportedData['metadata'];
}

const LIVE_SERVER_URL = 'http://localhost:8765';

/**
 * Try to load from live inference server first, then static JSON, then mock.
 */
export async function loadGNNData(pdbId: string = '4OBE', chain: string = 'A'): Promise<LoadedData> {
  // Try live server
  try {
    const response = await fetch(
      `${LIVE_SERVER_URL}/api/embeddings?pdb_id=${pdbId}&chain=${chain}`,
      { signal: AbortSignal.timeout(2000) }
    );
    if (response.ok) {
      const data: ExportedData = await response.json();
      if (data.nodes && data.nodes.length > 0) {
        console.log(`[Tokyo Eyes] Connected to live server — ${data.metadata.pdb_id} (${data.nodes.length} residues)`);
        return parseExportedData(data);
      }
    }
  } catch {
    // Live server not available, fall through
  }

  // Try static JSON
  try {
    const response = await fetch('/viewer_data.json');
    if (response.ok) {
      const data: ExportedData = await response.json();
      if (data.nodes && data.nodes.length > 0) {
        console.log(`[Tokyo Eyes] Loaded static export — ${data.metadata.pdb_id} (${data.nodes.length} residues)`);
        return parseExportedData(data);
      }
    }
  } catch {
    // No static file, fall through
  }

  // Fall back to mock
  console.log('[Tokyo Eyes] Using mock data (no server or export found)');
  return { ...generateMockGNNData(), metadata: undefined };
}

/**
 * Check if live server is available.
 */
export async function checkLiveServer(): Promise<{ available: boolean; status?: any }> {
  try {
    const response = await fetch(`${LIVE_SERVER_URL}/api/status`, {
      signal: AbortSignal.timeout(1500)
    });
    if (response.ok) {
      const status = await response.json();
      return { available: true, status };
    }
  } catch {
    // not available
  }
  return { available: false };
}

/**
 * Fetch available proteins from live server.
 */
export async function fetchProteinList(): Promise<Array<{ pdb_id: string; gene: string; desc: string }>> {
  try {
    const response = await fetch(`${LIVE_SERVER_URL}/api/proteins`, {
      signal: AbortSignal.timeout(2000)
    });
    if (response.ok) {
      const data = await response.json();
      return data.proteins || [];
    }
  } catch {
    // not available
  }
  return [];
}

/**
 * Parse exported JSON into viewer-compatible format.
 */
function parseExportedData(data: ExportedData): LoadedData {
  const nodes: NodeData[] = data.nodes.map(n => ({
    id: n.id,
    position: new THREE.Vector3(n.position[0], n.position[1], n.position[2]),
    position2D: n.position2D,
    color: n.color,
    domain: n.domain,
    isOutlier: n.isOutlier,
    value: n.value,
    depth: n.depth,
    expertId: n.expertId,
  }));

  const edges: EdgeData[] = data.edges.map(e => ({
    source: e.source,
    target: e.target,
    distance: e.distance,
  }));

  return { nodes, edges, metadata: data.metadata };
}

/**
 * Load from a File object (drag-and-drop or file input).
 */
export async function loadGNNDataFromFile(file: File): Promise<LoadedData> {
  const text = await file.text();
  const data: ExportedData = JSON.parse(text);
  return parseExportedData(data);
}
