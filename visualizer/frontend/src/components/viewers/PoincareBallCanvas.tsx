import { useEffect, useMemo, useRef, type MouseEvent } from "react";

import type { ResidueEmbedding } from "../../lib/types";
import {
  computeBallPoint,
  metricValueToColor,
  projectBallPoint,
  type ViewportColorMode,
} from "../../lib/poincareViewportMath";
import type { ResidueMetricSeries } from "../../lib/viewportColorMetrics";

export interface PoincareBallCanvasProps {
  residues: ResidueEmbedding[];
  colorMode: ViewportColorMode;
  metricSeries: ResidueMetricSeries;
  highlightedResidues: string[];
  selectedResidueId: string | null;
  hoveredResidueId: string | null;
  curvature?: number;
  rotationAngle: number;
  autoRotate?: boolean;
  onResidueClick?: (residueId: string) => void;
  onResidueHover?: (residueId: string | null) => void;
}

export function PoincareBallCanvas({
  residues,
  colorMode,
  metricSeries,
  highlightedResidues,
  selectedResidueId,
  hoveredResidueId,
  curvature = 1,
  rotationAngle,
  autoRotate = true,
  onResidueClick,
  onResidueHover,
}: PoincareBallCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const angleRef = useRef(rotationAngle);
  const hitRegionsRef = useRef<
    Array<{ residueId: string; sx: number; sy: number; radius: number }>
  >([]);

  const stats = useMemo(() => {
    if (!residues.length) {
      return { depthMin: 0, depthMax: 1, maxRadius: 1 };
    }
    let depthMin = Infinity;
    let depthMax = -Infinity;
    let maxRadius = 0;
    for (const residue of residues) {
      depthMin = Math.min(depthMin, residue.cone_depth);
      depthMax = Math.max(depthMax, residue.cone_depth);
      maxRadius = Math.max(maxRadius, Math.hypot(residue.x, residue.y));
    }
    return { depthMin, depthMax, maxRadius: maxRadius || 1 };
  }, [residues]);

  useEffect(() => {
    angleRef.current = rotationAngle;
  }, [rotationAngle]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    let angle = angleRef.current;

    const draw = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const dpr = window.devicePixelRatio || 1;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      if (!w || !h) return;

      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const R = Math.min(w, h) * 0.4;

      ctx.strokeStyle = "rgba(184,189,204,0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(184,189,204,0.05)";
      for (const f of [0.5, 0.82]) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, R, R * f, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.ellipse(cx, cy, R * 0.5, R, 0, 0, Math.PI * 2);
      ctx.stroke();

      const highlightSet = new Set(highlightedResidues);
      const anyHi = highlightSet.size > 0;
      const points = residues
        .map((residue) => {
          const ball = computeBallPoint(
            residue,
            stats.depthMin,
            stats.depthMax,
            stats.maxRadius,
            curvature,
          );
          const projected = projectBallPoint(ball, cx, cy, R, angle);
          return { residue, projected };
        })
        .sort((a, b) => a.projected.z - b.projected.z);

      const hitRegions: Array<{ residueId: string; sx: number; sy: number; radius: number }> =
        [];

      for (const { residue, projected } of points) {
        const inHi = highlightSet.has(residue.residue_id);
        const dim = anyHi && !inHi;
        const front = (projected.z + 1) / 2;
        const rad = (1.8 + 3 * 0.4) * (0.65 + 0.5 * front);
        const metric = metricSeries.values.get(residue.residue_id) ?? 0;
        const color = metricValueToColor(
          metric,
          colorMode,
          metricSeries.min,
          metricSeries.max,
        );

        ctx.globalAlpha = dim ? 0.16 : 0.42 + 0.55 * front;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(projected.sx, projected.sy, rad, 0, Math.PI * 2);
        ctx.fill();

        if (inHi) {
          ctx.globalAlpha = 0.9;
          ctx.strokeStyle = "#DC5CE9";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(projected.sx, projected.sy, rad + 2.6, 0, Math.PI * 2);
          ctx.stroke();
        }
        if (
          residue.residue_id === selectedResidueId ||
          residue.residue_id === hoveredResidueId
        ) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = "#EAECF2";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(projected.sx, projected.sy, rad + 4.2, 0, Math.PI * 2);
          ctx.stroke();
        }

        hitRegions.push({
          residueId: residue.residue_id,
          sx: projected.sx,
          sy: projected.sy,
          radius: rad + 4,
        });
      }

      ctx.globalAlpha = 1;
      hitRegionsRef.current = hitRegions;
      if (autoRotate) angle += 0.0042;
      angleRef.current = angle;
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [
    residues,
    colorMode,
    metricSeries,
    highlightedResidues,
    selectedResidueId,
    hoveredResidueId,
    curvature,
    autoRotate,
    stats,
  ]);

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    for (const hit of hitRegionsRef.current) {
      if (Math.hypot(x - hit.sx, y - hit.sy) <= hit.radius) {
        onResidueClick?.(hit.residueId);
        return;
      }
    }
  };

  const handleMove = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let hovered: string | null = null;
    for (const hit of hitRegionsRef.current) {
      if (Math.hypot(x - hit.sx, y - hit.sy) <= hit.radius) {
        hovered = hit.residueId;
        break;
      }
    }
    onResidueHover?.(hovered);
  };

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full cursor-pointer"
      onClick={handleClick}
      onMouseMove={handleMove}
      onMouseLeave={() => onResidueHover?.(null)}
    />
  );
}
