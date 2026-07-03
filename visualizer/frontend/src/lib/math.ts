/**
 * Hyperbolic Geometry Math Utilities for the Poincaré Ball model
 * Tokyo Eyes v5 — supports both 2D disc and 3D ball representations
 */
import * as THREE from 'three';

// Möbius addition in the Poincaré ball
export function mobiusAdd(a: THREE.Vector3, x: THREE.Vector3, c: number = 1): THREE.Vector3 {
  const normA_sq = a.lengthSq();
  const normX_sq = x.lengthSq();
  const dotAX = a.dot(x);

  const num = a.clone().multiplyScalar(1 + 2 * c * dotAX + c * normX_sq)
      .add(x.clone().multiplyScalar(1 - c * normA_sq));
  const den = 1 + 2 * c * dotAX + c * c * normA_sq * normX_sq;

  return num.divideScalar(den);
}

// 2D Möbius addition for the Poincaré disc
export function mobiusAdd2D(a: [number, number], x: [number, number], c: number = 1): [number, number] {
  const normA_sq = a[0] * a[0] + a[1] * a[1];
  const normX_sq = x[0] * x[0] + x[1] * x[1];
  const dotAX = a[0] * x[0] + a[1] * x[1];

  const den = 1 + 2 * c * dotAX + c * c * normA_sq * normX_sq;
  const numX = (1 + 2 * c * dotAX + c * normX_sq) * a[0] + (1 - c * normA_sq) * x[0];
  const numY = (1 + 2 * c * dotAX + c * normX_sq) * a[1] + (1 - c * normA_sq) * x[1];

  return [numX / den, numY / den];
}

/** Transport disc point so (ax, ay) maps to the origin — curvature-correct ⊕_c. */
export function mobiusRecenter2D(
  x: number,
  y: number,
  ax: number,
  ay: number,
  c: number = 1,
): [number, number] {
  return mobiusAdd2D([-ax, -ay], [x, y], c);
}

// Hyperbolic distance between two points in the Poincaré ball
export function hyperbolicDistance(x: THREE.Vector3, y: THREE.Vector3, c: number = 1): number {
  const minusX = x.clone().multiplyScalar(-1);
  const mobiusDiff = mobiusAdd(minusX, y, c);
  const normDiff = mobiusDiff.length();
  
  if (normDiff >= 1 / Math.sqrt(c)) return Infinity;
  return (2 / Math.sqrt(c)) * Math.atanh(Math.sqrt(c) * normDiff);
}

// 2D hyperbolic distance
export function hyperbolicDistance2D(x: [number, number], y: [number, number], c: number = 1): number {
  const minusX: [number, number] = [-x[0], -x[1]];
  const diff = mobiusAdd2D(minusX, y, c);
  const normDiff = Math.sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
  
  if (normDiff >= 1 / Math.sqrt(c)) return Infinity;
  return (2 / Math.sqrt(c)) * Math.atanh(Math.sqrt(c) * normDiff);
}

export interface NodeData {
  id: string;
  position: THREE.Vector3;     // 3D ball position
  position2D: [number, number]; // 2D disc position
  color: string;
  domain: string;
  isOutlier: boolean;
  value: number;
  depth: number;               // radial depth (from model)
  expertId: number;            // dominant expert routing
}

export interface EdgeData {
  source: string;
  target: string;
  distance: number;
}

// Generates mock data simulating v5 GNN outputs with decoupled radial/angular
export function generateMockGNNData(numNodes: number = 200, numHubs: number = 5): { nodes: NodeData[], edges: EdgeData[] } {
  const nodes: NodeData[] = [];
  const edges: EdgeData[] = [];
  
  const domains = ['P-loop', 'Switch-I', 'Switch-II', 'α3-helix', 'C-terminal'];
  const domainColors = ['#ff0055', '#00ffcc', '#aa00ff', '#ffaa00', '#00aaff'];

  // Hub directions (angular) — evenly spaced on sphere
  const hubDirections3D: THREE.Vector3[] = [];
  const hubAngles2D: number[] = [];
  for (let i = 0; i < numHubs; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / numHubs);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    hubDirections3D.push(new THREE.Vector3(
      Math.sin(phi) * Math.cos(theta),
      Math.sin(phi) * Math.sin(theta),
      Math.cos(phi)
    ).normalize());
    hubAngles2D.push((2 * Math.PI * i) / numHubs);
  }

  for (let i = 0; i < numNodes; i++) {
    const domainIdx = i % domains.length;
    const hubDir = hubDirections3D[domainIdx];
    const hubAngle = hubAngles2D[domainIdx];

    // Radial depth — simulates burial (independent of angular)
    const isOutlier = Math.random() > 0.95;
    const baseDepth = isOutlier ? 0.85 + Math.random() * 0.1 : 0.2 + Math.random() * 0.6;
    
    // Angular noise around hub direction
    const angNoise = new THREE.Vector3(
      (Math.random() - 0.5) * 0.4,
      (Math.random() - 0.5) * 0.4,
      (Math.random() - 0.5) * 0.4
    );
    const direction = hubDir.clone().add(angNoise).normalize();
    
    // 3D position = depth * direction (mimics v5 recombination)
    const pos3D = direction.multiplyScalar(baseDepth * 0.95);
    if (pos3D.length() > 0.95) pos3D.setLength(0.95);

    // 2D disc position — project angular structure to disc
    const angle2D = hubAngle + (Math.random() - 0.5) * 0.8;
    const radius2D = baseDepth * 0.95;
    const pos2D: [number, number] = [
      Math.min(0.95, radius2D) * Math.cos(angle2D),
      Math.min(0.95, radius2D) * Math.sin(angle2D),
    ];

    nodes.push({
      id: `res_${i}`,
      position: pos3D,
      position2D: pos2D,
      color: domainColors[domainIdx],
      domain: domains[domainIdx],
      isOutlier,
      value: Math.random(),
      depth: baseDepth,
      expertId: domainIdx % 4,
    });
  }

  // Generate k-nearest edges (using 3D hyperbolic distance)
  const k = 4;
  for (let i = 0; i < numNodes; i++) {
    const dists = nodes.map((n, idx) => ({
      idx,
      d: hyperbolicDistance(nodes[i].position, n.position)
    })).sort((a, b) => a.d - b.d);

    for (let j = 1; j <= k; j++) {
      if (dists[j] && dists[j].idx > i) {
        edges.push({
          source: nodes[i].id,
          target: nodes[dists[j].idx].id,
          distance: dists[j].d
        });
      }
    }
  }

  return { nodes, edges };
}
