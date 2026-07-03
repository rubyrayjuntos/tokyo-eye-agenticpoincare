import type { ResidueEmbedding } from "./types";
import {
  metricColor,
  metricValue,
  metricValueToColor,
  type ViewportColorMode,
} from "./viewportColorMetrics";
import { mobiusRecenter2D } from "./math";

export type { ViewportColorMode };
export { metricColor, metricValue, metricValueToColor };
export { VIEWPORT_COLOR_MODE_OPTIONS } from "./viewportColorMetrics";
export { mobiusRecenter2D };

export type ViewportLayoutMode = "tri" | "structure" | "ball" | "disc";

export interface BallPoint {
  bx: number;
  by: number;
  bz: number;
}

/** @deprecated Use mobiusRecenter2D with explicit curvature for geoopt-consistent transport. */
export function mobiusTransform(
  x: number,
  y: number,
  ax: number,
  ay: number,
): { x: number; y: number } {
  const [tx, ty] = mobiusRecenter2D(x, y, ax, ay, 1);
  return { x: tx, y: ty };
}

export function computeBallPoint(
  residue: ResidueEmbedding,
  depthMin: number,
  depthMax: number,
  maxRadius: number,
  curvature = 1,
): BallPoint {
  const depthSpan = depthMax - depthMin || 1;
  const normDepth = (residue.cone_depth - depthMin) / depthSpan;
  const scale = (0.9 / Math.max(maxRadius, 0.05)) * Math.pow(curvature, 0.6);
  return {
    bx: residue.x * scale,
    by: normDepth * 0.8 - 0.4,
    bz: residue.y * scale,
  };
}

export function projectBallPoint(
  point: BallPoint,
  cx: number,
  cy: number,
  radius: number,
  angle: number,
  tilt = 0.5,
): { sx: number; sy: number; z: number } {
  const cosa = Math.cos(angle);
  const sina = Math.sin(angle);
  const cost = Math.cos(tilt);
  const sint = Math.sin(tilt);
  const x1 = point.bx * cosa + point.bz * sina;
  const z1 = -point.bx * sina + point.bz * cosa;
  const y2 = point.by * cost - z1 * sint;
  const z2 = point.by * sint + z1 * cost;
  return { sx: cx + x1 * radius, sy: cy - y2 * radius, z: z2 };
}
