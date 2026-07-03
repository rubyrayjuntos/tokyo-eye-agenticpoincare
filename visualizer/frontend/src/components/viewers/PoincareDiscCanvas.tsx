import { useCallback, useMemo, useRef } from "react";

import type { ResidueEmbedding } from "../../lib/types";
import {
  metricValueToColor,
  mobiusRecenter2D,
  type ViewportColorMode,
} from "../../lib/poincareViewportMath";
import type { ResidueMetricSeries } from "../../lib/viewportColorMetrics";

export interface PoincareDiscCanvasProps {
  residues: ResidueEmbedding[];
  colorMode: ViewportColorMode;
  metricSeries: ResidueMetricSeries;
  highlightedResidues: string[];
  selectedResidueId: string | null;
  hoveredResidueId: string | null;
  mobiusX?: number;
  mobiusY?: number;
  curvature?: number;
  onMobiusDrag?: (x: number, y: number) => void;
  onResidueClick?: (residueId: string) => void;
  onResidueHover?: (residueId: string | null) => void;
}

export function PoincareDiscCanvas({
  residues,
  colorMode,
  metricSeries,
  highlightedResidues,
  selectedResidueId,
  hoveredResidueId,
  mobiusX = 0,
  mobiusY = 0,
  curvature = 1,
  onMobiusDrag,
  onResidueClick,
  onResidueHover,
}: PoincareDiscCanvasProps) {
  const cx0 = 180;
  const cy0 = 180;
  const Rpix = 165;
  const rscale = Math.pow(curvature, 0.6);
  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef(false);

  const clientToNorm = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const rect = svg.getBoundingClientRect();
    const ox = rect.left + rect.width / 2;
    const oy = rect.top + rect.height / 2;
    const scale = (Math.min(rect.width, rect.height) / 2) * (Rpix / 180);
    return { x: (clientX - ox) / scale, y: -(clientY - oy) / scale };
  }, []);

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      if (!onMobiusDrag) return;
      draggingRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      const { x, y } = clientToNorm(event.clientX, event.clientY);
      onMobiusDrag(x, y);
    },
    [clientToNorm, onMobiusDrag],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      if (!draggingRef.current || !onMobiusDrag) return;
      const { x, y } = clientToNorm(event.clientX, event.clientY);
      onMobiusDrag(x, y);
    },
    [clientToNorm, onMobiusDrag],
  );

  const handlePointerUp = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const points = useMemo(() => {
    const highlightSet = new Set(highlightedResidues);
    const anyHi = highlightSet.size > 0;
    return residues.map((residue) => {
      const transformed = mobiusRecenter2D(
        residue.x,
        residue.y,
        mobiusX,
        mobiusY,
        curvature,
      );
      let wx = transformed[0] * rscale;
      let wy = transformed[1] * rscale;
      const rr = Math.hypot(wx, wy);
      if (rr > 0.965) {
        wx *= 0.965 / rr;
        wy *= 0.965 / rr;
      }
      const cx = cx0 + wx * Rpix;
      const cy = cy0 - wy * Rpix;
      const inHi = highlightSet.has(residue.residue_id);
      const isSel = residue.residue_id === selectedResidueId;
      const isHov = residue.residue_id === hoveredResidueId;
      const dim = anyHi && !inHi;
      let rad = 2 + 4 * 0.4;
      if (isSel || isHov) rad += 1.6;
      return {
        residue,
        cx,
        cy,
        rad,
        inHi,
        isSel,
        isHov,
        dim,
        fill: metricValueToColor(
          metricSeries.values.get(residue.residue_id) ?? 0,
          colorMode,
          metricSeries.min,
          metricSeries.max,
        ),
      };
    });
  }, [
    residues,
    colorMode,
    metricSeries,
    highlightedResidues,
    selectedResidueId,
    hoveredResidueId,
    mobiusX,
    mobiusY,
    curvature,
    rscale,
  ]);

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 360 360"
      className="h-full w-full touch-none"
      onMouseLeave={() => onResidueHover?.(null)}
    >
      <rect
        x={0}
        y={0}
        width={360}
        height={360}
        fill="transparent"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      />
      <circle
        cx={cx0}
        cy={cy0}
        r={Rpix}
        fill="rgba(13,148,136,0.015)"
        stroke="rgba(184,189,204,0.14)"
        strokeWidth={1}
      />
      {[0.33, 0.66].map((f) => (
        <circle
          key={f}
          cx={cx0}
          cy={cy0}
          r={Rpix * f}
          fill="none"
          stroke="rgba(184,189,204,0.05)"
          strokeWidth={1}
        />
      ))}
      <line
        x1={cx0 - Rpix}
        y1={cy0}
        x2={cx0 + Rpix}
        y2={cy0}
        stroke="rgba(184,189,204,0.05)"
        strokeWidth={1}
      />
      <line
        x1={cx0}
        y1={cy0 - Rpix}
        x2={cx0}
        y2={cy0 + Rpix}
        stroke="rgba(184,189,204,0.05)"
        strokeWidth={1}
      />
      {points.map((point) => (
        <g key={point.residue.residue_id}>
          <circle
            cx={point.cx}
            cy={point.cy}
            r={point.rad}
            fill={point.fill}
            fillOpacity={point.dim ? 0.22 : 0.95}
            stroke={point.inHi ? "#DC5CE9" : point.isSel || point.isHov ? "#EAECF2" : "none"}
            strokeWidth={point.inHi || point.isSel || point.isHov ? 1.4 : 0}
            className="cursor-pointer"
            onMouseEnter={() => onResidueHover?.(point.residue.residue_id)}
            onClick={() => onResidueClick?.(point.residue.residue_id)}
          />
          {(point.inHi || point.isSel || point.isHov) && (
            <text
              x={point.cx + point.rad + 3}
              y={point.cy + 3}
              fill="#EAECF2"
              fontSize={8}
              fontFamily="JetBrains Mono, monospace"
              style={{ pointerEvents: "none" }}
            >
              {point.residue.residue_index}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
