import * as THREE from "three";

// ---------- Core types ----------

export interface NodeData {
  id: string;
  position: THREE.Vector3;   // 3D position in Poincaré ball (unit ball)
  position2D: [number, number]; // 2D equatorial projection
  color: string;
  depth: number;             // cone_depth [0, 1]
  value: number;             // current metric value (for stats)
  isOutlier: boolean;
  domain: string;            // domain label, e.g. "Switch-II"
  label?: string;
  epistemic?: number;
  aleatoric?: number;
}

export interface EdgeData {
  source: string;
  target: string;
  weight?: number;
}

// ---------- Möbius addition on Poincaré ball (3D) ----------
// mobius_add(u, v, c) = ((1 + 2c<u,v> + c|v|²)u + (1 - c|u|²)v)
//                        / (1 + 2c<u,v> + c²|u|²|v|²)

export function mobiusAdd(
  u: THREE.Vector3,
  v: THREE.Vector3,
  c: number
): THREE.Vector3 {
  const uu = u.dot(u);
  const vv = v.dot(v);
  const uv = u.dot(v);

  const denom = 1 + 2 * c * uv + c * c * uu * vv;
  if (Math.abs(denom) < 1e-10) return v.clone();

  const s1 = (1 + 2 * c * uv + c * vv) / denom;
  const s2 = (1 - c * uu) / denom;

  return new THREE.Vector3(
    s1 * u.x + s2 * v.x,
    s1 * u.y + s2 * v.y,
    s1 * u.z + s2 * v.z
  );
}

// ---------- Möbius addition on Poincaré disc (2D) ----------

export function mobiusAdd2D(
  u: [number, number],
  v: [number, number],
  c: number
): [number, number] {
  const uu = u[0] * u[0] + u[1] * u[1];
  const vv = v[0] * v[0] + v[1] * v[1];
  const uv = u[0] * v[0] + u[1] * v[1];

  const denom = 1 + 2 * c * uv + c * c * uu * vv;
  if (Math.abs(denom) < 1e-10) return [...v] as [number, number];

  const s1 = (1 + 2 * c * uv + c * vv) / denom;
  const s2 = (1 - c * uu) / denom;

  return [s1 * u[0] + s2 * v[0], s1 * u[1] + s2 * v[1]];
}

// ---------- Hyperbolic distances ----------

export function hyperbolicDistance(
  u: THREE.Vector3,
  v: THREE.Vector3,
  c: number
): number {
  const diff = new THREE.Vector3().subVectors(u, v);
  const num = diff.lengthSq();
  const uu = u.lengthSq();
  const vv = v.lengthSq();
  const denom = (1 - c * uu) * (1 - c * vv);
  if (denom <= 0) return Infinity;
  const arg = 1 + 2 * c * (num / denom);
  return (1 / Math.sqrt(c)) * Math.acosh(Math.max(1, arg));
}

export function hyperbolicDistance2D(
  u: [number, number],
  v: [number, number],
  c: number
): number {
  const dx = u[0] - v[0];
  const dy = u[1] - v[1];
  const num = dx * dx + dy * dy;
  const uu = u[0] * u[0] + u[1] * u[1];
  const vv = v[0] * v[0] + v[1] * v[1];
  const denom = (1 - c * uu) * (1 - c * vv);
  if (denom <= 0) return Infinity;
  const arg = 1 + 2 * c * (num / denom);
  return (1 / Math.sqrt(c)) * Math.acosh(Math.max(1, arg));
}

// ---------- Color mapping ----------

const DEPTH_COLORS: Array<[number, string]> = [
  [0.0, "#1e3a5f"],   // buried — deep blue
  [0.4, "#0ea5e9"],   // mid — sky blue
  [0.7, "#22d3ee"],   // near-surface — cyan
  [1.0, "#f0f9ff"],   // surface — white
];

export function depthToColor(depth: number): string {
  const t = Math.max(0, Math.min(1, depth));
  for (let i = 1; i < DEPTH_COLORS.length; i++) {
    const [t0, c0] = DEPTH_COLORS[i - 1];
    const [t1, c1] = DEPTH_COLORS[i];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return lerpHex(c0, c1, f);
    }
  }
  return DEPTH_COLORS[DEPTH_COLORS.length - 1][1];
}

export function uncertaintyToColor(u: number): string {
  const t = Math.max(0, Math.min(1, u));
  return lerpHex("#064e3b", "#f59e0b", t);
}

function lerpHex(a: string, b: string, t: number): string {
  const ar = parseInt(a.slice(1, 3), 16);
  const ag = parseInt(a.slice(3, 5), 16);
  const ab = parseInt(a.slice(5, 7), 16);
  const br = parseInt(b.slice(1, 3), 16);
  const bg = parseInt(b.slice(3, 5), 16);
  const bb = parseInt(b.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${bl.toString(16).padStart(2, "0")}`;
}

// ---------- Mock data generator ----------

const DOMAINS = ["P-loop", "Switch-I", "Switch-II", "α3", "α4", "C-term", "Other"];

export interface MockGNNData {
  nodes: NodeData[];
  edges: EdgeData[];
}

export function generateMockGNNData(count = 80): MockGNNData {
  const rng = seededRng(42);
  const nodes: NodeData[] = [];

  for (let i = 0; i < count; i++) {
    // Distribute nodes across the ball with realistic depth distribution
    const depth = rng() ** 2; // bias toward centre
    const theta = rng() * Math.PI;
    const phi = rng() * 2 * Math.PI;
    const r = depth * 0.92;

    const x = r * Math.sin(theta) * Math.cos(phi);
    const y = r * Math.sin(theta) * Math.sin(phi);
    const z = r * Math.cos(theta);

    const epistemic = rng() * 0.6 + (depth > 0.7 ? 0.3 : 0);
    const aleatoric = rng() * 0.4;
    const domain = DOMAINS[Math.floor(rng() * DOMAINS.length)];

    nodes.push({
      id: `res_${i.toString().padStart(3, "0")}`,
      position: new THREE.Vector3(x, y, z),
      position2D: [x, y],
      color: depthToColor(depth),
      depth,
      value: depth,
      isOutlier: depth > 0.82 && epistemic > 0.6,
      domain,
      epistemic,
      aleatoric,
    });
  }

  // Sparse random edges (contact graph)
  const edges: EdgeData[] = [];
  for (let i = 0; i < count; i++) {
    const numEdges = Math.floor(rng() * 3) + 1;
    for (let j = 0; j < numEdges; j++) {
      const target = Math.floor(rng() * count);
      if (target !== i) {
        edges.push({ source: nodes[i].id, target: nodes[target].id });
      }
    }
  }

  return { nodes, edges };
}

function seededRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0x100000000;
  };
}
