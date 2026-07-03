import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { useDashboard } from "../lib/context";
import { useHydration } from "../context/HydrationProvider";
import { api } from "../lib/api";
import { buildHydrationView } from "../lib/hydrationView";
import { getArtifactSurfaceState } from "../lib/artifactAvailability";
import type { EmbeddingData, ResidueEmbedding, SelectedResidueInfo, PoincareColorMode } from "../lib/types";
import { THERAPEUTIC_GOALS } from "../lib/therapeuticCompiler";

// --- Hyperbolic math utilities ---

function atanhSafe(x: number): number {
  const clamped = Math.max(-0.999999, Math.min(0.999999, x));
  return 0.5 * Math.log((1 + clamped) / (1 - clamped));
}

function mobiusTransform(
  x: number, y: number, ax: number, ay: number
): { x: number; y: number } {
  const numRe = x - ax;
  const numIm = y - ay;
  const denRe = 1 - (ax * x + ay * y);
  const denIm = -(ax * y - ay * x);
  const denMagSq = denRe * denRe + denIm * denIm + 1e-8;
  return {
    x: (numRe * denRe + numIm * denIm) / denMagSq,
    y: (numIm * denRe - numRe * denIm) / denMagSq,
  };
}

function toScreen(
  x: number, y: number, cx: number, cy: number, radius: number
): { x: number; y: number } {
  return { x: cx + x * radius, y: cy - y * radius };
}

function zoneFromRadius(r: number): "core" | "mid" | "periphery" {
  if (r < 0.35) return "core";
  if (r < 0.7) return "mid";
  return "periphery";
}

function valueToColor(v: number, min = 0, max = 1): string {
  const t = Math.max(0, Math.min(1, (v - min) / (max - min + 1e-8)));
  // Viridis-inspired: dark purple → teal → yellow
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(
    84 + (t < 0.5 ? t * 2 * (168 - 84) : (1 - t) * 2 * 168)
  );
  return `rgb(${r},${g},${b})`;
}

function percentileRank(values: number[], value: number): number {
  if (values.length === 0) return 0;
  let count = 0;
  for (const v of values) {
    if (v <= value) count += 1;
  }
  return (count / values.length) * 100;
}

type ColorMode = "cone_depth" | "uncertainty";

interface TransformedPoint extends ResidueEmbedding {
  tx: number;
  ty: number;
  euclidR: number;
  hyperbolicR: number;
}

interface PoincareScatterProps {
  onColorModeChange?: (mode: PoincareColorMode) => void;
  onSelectedResidueChange?: (residue: SelectedResidueInfo | null) => void;
  onMobiusFocusChange?: (enabled: boolean) => void;
  onBrushSelectionChange?: (ids: string[]) => void;
  // Opt-in resistance topology layer from Therapeutic Compiler (hyperbolic r/theta data).
  // When provided or auto-fetched via toggle, renders as non-default overlay.
  // Defaults to completely disabled; no change to core GNN Poincaré rendering, color modes, hover, mobius, brush, or overlays.
  resistanceTopology?: { nodes?: any[]; edges?: any[] } | null;
}

export default function PoincareScatter({
  onColorModeChange,
  onSelectedResidueChange,
  onMobiusFocusChange,
  onBrushSelectionChange,
}: PoincareScatterProps = {}) {
  const {
    activeStructure,
    currentDirective,
    compareState,
    therapeuticCompilerState: ctxCompilerState,
    setCollapseSimulationState,
    collapseSimulationState,
  } = useDashboard();
  const {
    embeddings,
    isHydrating,
    refresh,
    artifactAvailability,
  } = useHydration();
  const data = embeddings;
  const loading = isHydrating;
  const [colorMode, setColorMode] = useState<ColorMode>("cone_depth");
  const [hovered, setHovered] = useState<string | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mobiusFocus, setMobiusFocus] = useState(false);
  const [brushStart, setBrushStart] = useState<{ x: number; y: number } | null>(null);
  const [brushEnd, setBrushEnd] = useState<{ x: number; y: number } | null>(null);
  const [regionSelection, setRegionSelection] = useState<string[]>([]);
  const [directiveHighlightColors, setDirectiveHighlightColors] = useState<Record<string, string>>({});

  // Overlay mode state for compare
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const [secondaryData, setSecondaryData] = useState<EmbeddingData | null>(null);
  const [secondaryLoading, setSecondaryLoading] = useState(false);

  // Opt-in resistance topology layer (from Therapeutic Compiler hyperbolic data).
  // Completely independent of core Poincaré (GNN points, colors, mobius, brush, existing overlays).
  // Toggle only appears / has effect when user explicitly enables it.
  const [showResistanceLayer, setShowResistanceLayer] = useState(false);
  const [resistanceData, setResistanceData] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  const [resistanceLoading, setResistanceLoading] = useState(false);
  // Goal selector exposed to Poincaré for the resistance layer (UI parity + future styling/filtering based on therapeutic goal)
  const [resistanceGoal, setResistanceGoal] = useState<string>(THERAPEUTIC_GOALS[0].id);

  // Zoom and pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [panStartOffset, setPanStartOffset] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const size = 400;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 20;

  // Compute the viewBox based on zoom and pan
  const viewBoxSize = size / zoom;
  const viewBoxX = (size - viewBoxSize) / 2 - pan.x;
  const viewBoxY = (size - viewBoxSize) / 2 - pan.y;
  const viewBox = `${viewBoxX} ${viewBoxY} ${viewBoxSize} ${viewBoxSize}`;

  // Propagate state changes to parent via callbacks
  useEffect(() => {
    onColorModeChange?.(colorMode);
  }, [colorMode, onColorModeChange]);

  useEffect(() => {
    onMobiusFocusChange?.(mobiusFocus);
  }, [mobiusFocus, onMobiusFocusChange]);

  useEffect(() => {
    onBrushSelectionChange?.(regionSelection);
  }, [regionSelection, onBrushSelectionChange]);

  // Keyboard controls for zoom/pan
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if the container or its children are focused
      if (!containerRef.current?.contains(document.activeElement) &&
          document.activeElement !== containerRef.current) return;

      const panStep = 20 / zoom;
      switch (e.key) {
        case "+":
        case "=":
          e.preventDefault();
          setZoom((z) => Math.min(10, z * 1.2));
          break;
        case "-":
          e.preventDefault();
          setZoom((z) => Math.max(0.5, z / 1.2));
          break;
        case "0":
          e.preventDefault();
          setZoom(1);
          setPan({ x: 0, y: 0 });
          break;
        case "ArrowLeft":
          e.preventDefault();
          setPan((p) => ({ ...p, x: p.x + panStep }));
          break;
        case "ArrowRight":
          e.preventDefault();
          setPan((p) => ({ ...p, x: p.x - panStep }));
          break;
        case "ArrowUp":
          e.preventDefault();
          setPan((p) => ({ ...p, y: p.y + panStep }));
          break;
        case "ArrowDown":
          e.preventDefault();
          setPan((p) => ({ ...p, y: p.y - panStep }));
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [zoom]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setZoom((z) => Math.max(0.5, Math.min(10, z * factor)));
  }, []);

  // Middle-click or Ctrl+click pan
  const handlePanStart = useCallback((e: React.MouseEvent) => {
    // Middle button or Ctrl+left click (but not shift which is brush)
    if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      setPanStartOffset({ ...pan });
    }
  }, [pan]);

  const handlePanMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return;
    const dx = (e.clientX - panStart.x) / zoom;
    const dy = (e.clientY - panStart.y) / zoom;
    setPan({ x: panStartOffset.x + dx, y: panStartOffset.y + dy });
  }, [isPanning, panStart, panStartOffset, zoom]);

  const handlePanEnd = useCallback(() => {
    setIsPanning(false);
  }, []);

  // Respond to agent viewport directives
  useEffect(() => {
    if (!currentDirective) return;

    if (currentDirective.action === "set_metric" && currentDirective.metric) {
      const metric = currentDirective.metric;
      if (
        metric === "cone_depth" ||
        metric === "uncertainty" ||
        metric === "epistemic" ||
        metric === "aleatoric" ||
        metric === "total_uncertainty"
      ) {
        setColorMode(metric === "cone_depth" ? "cone_depth" : "uncertainty");
      }
      return;
    }

    if (currentDirective.action === "clear") {
      setRegionSelection([]);
      setSelectedId(null);
      setMobiusFocus(false);
      setDirectiveHighlightColors({});
      return;
    }

    if (
      currentDirective.action === "highlight" ||
      currentDirective.action === "focus"
    ) {
      const residueIds =
        currentDirective.highlight_groups?.flatMap((group) => group.residue_ids) ??
        currentDirective.focus_residues ??
        [];

      if (residueIds.length === 0) return;

      setRegionSelection(residueIds);
      setSelectedId(residueIds[0]);
      setDirectiveHighlightColors(
        Object.fromEntries(
          (currentDirective.highlight_groups ?? []).flatMap((group) =>
            group.residue_ids.map((residueId) => [residueId, group.color || "#facc15"])
          )
        )
      );

      if (currentDirective.action === "focus") {
        setMobiusFocus(true);
        setZoom((z) => Math.max(z, 1.75));
        setPan({ x: 0, y: 0 });
      }
      return;
    }

    setDirectiveHighlightColors({});
  }, [currentDirective]);

  // Fetch secondary embeddings when compare mode is active and overlay enabled
  useEffect(() => {
    if (!compareState.active || !compareState.secondaryStructure || !overlayEnabled) {
      setSecondaryData(null);
      return;
    }
    setSecondaryLoading(true);
    api
      .hydrate(compareState.secondaryStructure.structure_id)
      .then((hydration) => {
        const view = buildHydrationView(
          hydration,
          compareState.secondaryStructure?.structure_id,
        );
        setSecondaryData(view.embeddings);
      })
      .catch(() => setSecondaryData(null))
      .finally(() => setSecondaryLoading(false));
  }, [compareState.active, compareState.secondaryStructure, overlayEnabled]);

  // Reset overlay when exiting compare mode
  useEffect(() => {
    if (!compareState.active) {
      setOverlayEnabled(false);
      setSecondaryData(null);
    }
  }, [compareState.active]);

  // Fetch resistance topology data (real aggregator) when user enables the layer.
  // Prefers shared dashboard context (set by DataToolsPanel) for tighter integration.
  // Falls back to direct fetch. Non-blocking, defaults off, no impact on core Poincaré.
  useEffect(() => {
    if (!showResistanceLayer) {
      setResistanceData(null);
      return;
    }
    // Prefer context data if present (from compiler panel)
    if (ctxCompilerState && ctxCompilerState.hyperbolic) {
      setResistanceData({ nodes: ctxCompilerState.hyperbolic.nodes || [], edges: ctxCompilerState.hyperbolic.edges || [] });
      setResistanceLoading(false);
      return;
    }
    if (!activeStructure) {
      setResistanceData(null);
      return;
    }
    setResistanceLoading(true);
    const pw = "KRAS"; // default mapping; extend with more if multiple structures loaded
    fetch(`/api/therapeutic-compiler/state?pathways=${pw}`)
      .then(r => r.ok ? r.json() : null)
      .then(state => {
        if (state && state.hyperbolic) {
          setResistanceData({ nodes: state.hyperbolic.nodes || [], edges: state.hyperbolic.edges || [] });
        } else {
          setResistanceData(null);
        }
      })
      .catch(() => setResistanceData(null))
      .finally(() => setResistanceLoading(false));
  }, [showResistanceLayer, activeStructure, ctxCompilerState]);

  // Compute transformed points with Möbius recentering
  const transformedPoints = useMemo<TransformedPoint[]>(() => {
    if (!data) return [];
    const anchor = data.residues.find((p) => p.residue_id === selectedId);
    const ax = anchor?.x ?? 0;
    const ay = anchor?.y ?? 0;
    return data.residues.map((p) => {
      const coords =
        mobiusFocus && anchor
          ? mobiusTransform(p.x, p.y, ax, ay)
          : { x: p.x, y: p.y };
      const euclidR = Math.sqrt(p.x * p.x + p.y * p.y);
      const hyperbolicR = 2 * atanhSafe(euclidR);
      return { ...p, tx: coords.x, ty: coords.y, euclidR, hyperbolicR };
    });
  }, [data, selectedId, mobiusFocus]);

  // Compute transformed points for secondary data (overlay)
  const secondaryTransformedPoints = useMemo<TransformedPoint[]>(() => {
    if (!secondaryData || !overlayEnabled) return [];
    return secondaryData.residues.map((p) => {
      const euclidR = Math.sqrt(p.x * p.x + p.y * p.y);
      const hyperbolicR = 2 * atanhSafe(euclidR);
      return { ...p, tx: p.x, ty: p.y, euclidR, hyperbolicR };
    });
  }, [secondaryData, overlayEnabled]);

  // Metric values for color mapping
  const metricValues = useMemo(() => {
    return transformedPoints.map((p) =>
      colorMode === "cone_depth" ? p.cone_depth : p.epistemic_uncertainty
    );
  }, [transformedPoints, colorMode]);

  // Secondary metric values for shared color normalization
  const secondaryMetricValues = useMemo(() => {
    return secondaryTransformedPoints.map((p) =>
      colorMode === "cone_depth" ? p.cone_depth : p.epistemic_uncertainty
    );
  }, [secondaryTransformedPoints, colorMode]);

  // Shared color range across both structures when overlay is active
  const colorRange = useMemo(() => {
    const allValues = overlayEnabled
      ? [...metricValues, ...secondaryMetricValues]
      : metricValues;
    if (allValues.length === 0) return { min: 0, max: 1 };
    return { min: Math.min(...allValues), max: Math.max(...allValues) };
  }, [metricValues, secondaryMetricValues, overlayEnabled]);

  // Radial zone counts
  const radialZoneCounts = useMemo(() => {
    const counts = { core: 0, mid: 0, periphery: 0 };
    for (const p of transformedPoints) {
      counts[zoneFromRadius(p.euclidR)] += 1;
    }
    return counts;
  }, [transformedPoints]);

  // Boundary rug marks (residues near the disc edge)
  const boundaryRugAngles = useMemo(() => {
    return transformedPoints
      .filter((p) => p.euclidR >= 0.9)
      .slice(0, 120)
      .map((p) => Math.atan2(p.ty, p.tx));
  }, [transformedPoints]);

  // Selected point and geodesic targets
  const selectedPoint = useMemo(
    () => transformedPoints.find((p) => p.residue_id === selectedId) || null,
    [transformedPoints, selectedId]
  );

  // Propagate selected residue to parent
  useEffect(() => {
    if (!selectedPoint) {
      onSelectedResidueChange?.(null);
    } else {
      onSelectedResidueChange?.({
        residue_id: selectedPoint.residue_id,
        residue_name: null,
        chain_label: selectedPoint.chain_label,
        epistemic_uncertainty: selectedPoint.epistemic_uncertainty,
        cone_depth: selectedPoint.cone_depth,
      });
    }
  }, [selectedPoint, onSelectedResidueChange]);

  const geodesicTargets = useMemo(() => {
    if (!selectedPoint) return [] as TransformedPoint[];
    return transformedPoints
      .filter((p) => p.residue_id !== selectedPoint.residue_id)
      .sort((a, b) => {
        const da = (a.hyperbolicR - selectedPoint.hyperbolicR) ** 2;
        const db = (b.hyperbolicR - selectedPoint.hyperbolicR) ** 2;
        return da - db;
      })
      .slice(0, 5);
  }, [transformedPoints, selectedPoint]);

  // Find the matching secondary point for displacement vector drawing
  const secondaryMatchForSelected = useMemo(() => {
    if (!selectedPoint || !overlayEnabled || secondaryTransformedPoints.length === 0) return null;
    // Match by chain_label + residue_index (canonical position)
    return secondaryTransformedPoints.find(
      (p) =>
        p.chain_label === selectedPoint.chain_label &&
        p.residue_index === selectedPoint.residue_index
    ) || null;
  }, [selectedPoint, overlayEnabled, secondaryTransformedPoints]);

  // Brush rectangle
  const brushRect = useMemo(() => {
    if (!brushStart || !brushEnd) return null;
    const x = Math.min(brushStart.x, brushEnd.x);
    const y = Math.min(brushStart.y, brushEnd.y);
    const w = Math.abs(brushStart.x - brushEnd.x);
    const h = Math.abs(brushStart.y - brushEnd.y);
    return { x, y, w, h };
  }, [brushStart, brushEnd]);

  const handleBrushEnd = () => {
    if (!brushRect) {
      setBrushStart(null);
      setBrushEnd(null);
      return;
    }
    const selected = transformedPoints
      .filter((p) => {
        const s = toScreen(p.tx, p.ty, cx, cy, radius);
        return (
          s.x >= brushRect.x &&
          s.x <= brushRect.x + brushRect.w &&
          s.y >= brushRect.y &&
          s.y <= brushRect.y + brushRect.h
        );
      })
      .map((p) => p.residue_id);
    setRegionSelection(selected);
    setBrushStart(null);
    setBrushEnd(null);
  };

  // Selected point metrics
  const selectedMetricValue = selectedPoint
    ? colorMode === "cone_depth"
      ? selectedPoint.cone_depth
      : selectedPoint.epistemic_uncertainty
    : null;
  const metricMean =
    metricValues.length > 0
      ? metricValues.reduce((a, b) => a + b, 0) / metricValues.length
      : null;
  const selectedPercentile =
    selectedMetricValue !== null
      ? percentileRank(metricValues, selectedMetricValue)
      : null;

  if (!activeStructure) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
        Select a structure to view embeddings
      </div>
    );
  }

  const gnnState = getArtifactSurfaceState(artifactAvailability, "gnn_hyp");
  if (!loading && gnnState === "tier1_empty") {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 px-4 text-center text-zinc-500 text-xs">
        <span className="text-warning">GNN embeddings unavailable</span>
        <span>Tier-1 hyperbolic embeddings are not in the hydrate bundle yet.</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Controls */}
      <div className="flex items-center justify-between px-2 py-1 shrink-0 flex-wrap gap-1">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Poincaré Disc
        </span>
        <div className="flex gap-1 items-center flex-wrap">
          <select
            value={colorMode}
            onChange={(e) => setColorMode(e.target.value as ColorMode)}
            className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300"
          >
            <option value="cone_depth">Cone Depth</option>
            <option value="uncertainty">Uncertainty</option>
          </select>
          <label className="inline-flex items-center gap-1 text-[10px] text-zinc-400">
            <input
              type="checkbox"
              checked={mobiusFocus}
              onChange={(e) => setMobiusFocus(e.target.checked)}
              className="w-3 h-3"
            />
            Möbius
          </label>
          {compareState.active && (
            <label className="inline-flex items-center gap-1 text-[10px] text-amber-400">
              <input
                type="checkbox"
                checked={overlayEnabled}
                onChange={(e) => setOverlayEnabled(e.target.checked)}
                className="w-3 h-3"
              />
              Overlay
            </label>
          )}
          {/* Opt-in resistance topology layer - safe additive only, core Poincaré unchanged when off */}
          <label className="inline-flex items-center gap-1 text-[10px] text-red-400">
            <input
              type="checkbox"
              checked={showResistanceLayer}
              onChange={(e) => setShowResistanceLayer(e.target.checked)}
              className="w-3 h-3"
            />
            Resistance Topology {resistanceLoading ? "…" : ""}
          </label>
          {showResistanceLayer && (
            <select
              value={resistanceGoal}
              onChange={(e) => {
                const newGoal = e.target.value;
                setResistanceGoal(newGoal);
                // Unified trigger: update shared collapse state so the effect in DataToolsPanel immediately re-calls /collapse
                // with the *current* fraction + this new goal. No need to jiggle the slider.
                setCollapseSimulationState({
                  fraction: collapseSimulationState.fraction,
                  goal: newGoal,
                  active: true,
                });
              }}
              className="text-[9px] bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300"
              title="Therapeutic goal for resistance layer — changing this immediately re-optimizes the β-vector for the current slider fraction via unified context state"
            >
              {THERAPEUTIC_GOALS.map(g => (
                <option key={g.id} value={g.id}>{g.label}</option>
              ))}
            </select>
          )}
          <button
            onClick={refresh}
            disabled={loading}
            className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
          >
            {loading ? "…" : "↻"}
          </button>
          {regionSelection.length > 0 && (
            <span className="text-[10px] text-cyan-300">
              sel: {regionSelection.length}
            </span>
          )}
        </div>
      </div>

      {/* SVG Disc */}
      <div
        ref={containerRef}
        className="flex-1 min-h-0 p-1 relative"
        tabIndex={0}
        onWheel={handleWheel}
        onMouseDown={handlePanStart}
        onMouseMove={(e) => {
          handlePanMove(e);
        }}
        onMouseUp={handlePanEnd}
        onMouseLeave={handlePanEnd}
      >
        <svg
          viewBox={viewBox}
          className="w-full h-full"
          style={{ cursor: isPanning ? "grabbing" : zoom > 1 ? "grab" : "default" }}
          onMouseMove={(e) => {
            if (!brushStart) return;
            const rect = (
              e.currentTarget as SVGSVGElement
            ).getBoundingClientRect();
            const localX = viewBoxX + ((e.clientX - rect.left) / rect.width) * viewBoxSize;
            const localY = viewBoxY + ((e.clientY - rect.top) / rect.height) * viewBoxSize;
            setBrushEnd({ x: localX, y: localY });
          }}
          onMouseUp={handleBrushEnd}
          onMouseLeave={() => {
            setHovered(null);
            setHoverPos(null);
            handleBrushEnd();
          }}
        >
          {/* Disc boundary */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="#334155"
            strokeWidth="1.5"
          />
          {/* Grid circles */}
          {[0.25, 0.5, 0.75].map((f) => (
            <circle
              key={f}
              cx={cx}
              cy={cy}
              r={radius * f}
              fill="none"
              stroke="#1e293b"
              strokeWidth="0.5"
            />
          ))}
          {/* Crosshairs */}
          <line
            x1={cx - radius}
            y1={cy}
            x2={cx + radius}
            y2={cy}
            stroke="#1e293b"
            strokeWidth="0.5"
          />
          <line
            x1={cx}
            y1={cy - radius}
            x2={cx}
            y2={cy + radius}
            stroke="#1e293b"
            strokeWidth="0.5"
          />

          {/* Geodesic lines from selected to nearest neighbors */}
          {geodesicTargets.map((p, i) => {
            if (!selectedPoint) return null;
            const a = toScreen(
              selectedPoint.tx,
              selectedPoint.ty,
              cx,
              cy,
              radius
            );
            const b = toScreen(p.tx, p.ty, cx, cy, radius);
            return (
              <line
                key={`geo-${i}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke="#0ea5e9"
                strokeWidth="0.8"
                strokeDasharray="2 2"
                opacity={0.6}
              />
            );
          })}

          {/* Boundary rug marks */}
          {boundaryRugAngles.map((theta, i) => {
            const x1 = cx + Math.cos(theta) * (radius + 2);
            const y1 = cy + Math.sin(theta) * (radius + 2);
            const x2 = cx + Math.cos(theta) * (radius + 7);
            const y2 = cy + Math.sin(theta) * (radius + 7);
            return (
              <line
                key={`rug-${i}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#475569"
                strokeWidth="0.8"
                opacity={0.8}
              />
            );
          })}

          {/* Data points */}
          {transformedPoints.map((p, i) => {
            const px = cx + p.tx * radius;
            const py = cy - p.ty * radius;
            const isHovered = hovered === p.residue_id;
            const isSelected = selectedId === p.residue_id;
            const isRegionSelected = regionSelection.includes(p.residue_id);
            const directiveColor = directiveHighlightColors[p.residue_id];
            const isDirectiveHighlighted = Boolean(directiveColor);
            const baseRadius = isDirectiveHighlighted
              ? 5.5
              : isHovered || isSelected
              ? 4.5
              : 2.5;
            const pointRadius = baseRadius / Math.sqrt(zoom); // Scale down with zoom so density stays readable
            const metricValue = metricValues[i];
            return (
              <circle
                key={i}
                cx={px}
                cy={py}
                r={pointRadius}
                fill={
                  isDirectiveHighlighted
                    ? directiveColor
                    : valueToColor(metricValue, colorRange.min, colorRange.max)
                }
                stroke={
                  isDirectiveHighlighted
                    ? "#f8fafc"
                    : isSelected
                    ? "#f8fafc"
                    : isHovered
                    ? "#fff"
                    : isRegionSelected
                    ? "#14b8a6"
                    : "none"
                }
                strokeWidth={
                  isDirectiveHighlighted ? 2.25 : isSelected || isHovered || isRegionSelected ? 1.5 : 0
                }
                opacity={isDirectiveHighlighted || isSelected ? 1 : 0.82}
                className="cursor-pointer transition-all duration-100"
                onMouseEnter={(e) => {
                  setHovered(p.residue_id);
                  const rect = (
                    e.currentTarget.ownerSVGElement as SVGSVGElement
                  ).getBoundingClientRect();
                  setHoverPos({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                  });
                }}
                onMouseMove={(e) => {
                  const rect = (
                    e.currentTarget.ownerSVGElement as SVGSVGElement
                  ).getBoundingClientRect();
                  setHoverPos({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                  });
                }}
                onMouseLeave={() => {
                  setHovered(null);
                  setHoverPos(null);
                }}
                onClick={() => setSelectedId(p.residue_id)}
                onMouseDown={(e) => {
                  if (!e.shiftKey) return;
                  const rect = (
                    e.currentTarget.ownerSVGElement as SVGSVGElement
                  ).getBoundingClientRect();
                  const localX =
                    ((e.clientX - rect.left) / rect.width) * size;
                  const localY =
                    ((e.clientY - rect.top) / rect.height) * size;
                  setBrushStart({ x: localX, y: localY });
                  setBrushEnd({ x: localX, y: localY });
                }}
              />
            );
          })}

          {/* Brush selection rectangle */}
          {brushRect && (
            <rect
              x={brushRect.x}
              y={brushRect.y}
              width={brushRect.w}
              height={brushRect.h}
              fill="#0ea5e933"
              stroke="#38bdf8"
              strokeWidth="1"
              strokeDasharray="3 2"
            />
          )}

          {/* Secondary data points (overlay) */}
          {overlayEnabled && secondaryTransformedPoints.map((p, i) => {
            const px = cx + p.tx * radius;
            const py = cy - p.ty * radius;
            const pointRadius = 2.5 / Math.sqrt(zoom);
            const metricValue = secondaryMetricValues[i];
            return (
              <circle
                key={`sec-${i}`}
                cx={px}
                cy={py}
                r={pointRadius}
                fill="none"
                stroke={valueToColor(metricValue, colorRange.min, colorRange.max)}
                strokeWidth={1.2 / Math.sqrt(zoom)}
                opacity={0.6}
                className="pointer-events-none"
              />
            );
          })}

          {/* Opt-in Resistance Topology layer (from Therapeutic Compiler hyperbolic r/theta).
              Rendered only when explicitly toggled. Uses separate data and styling.
              Core GNN points, colors, interactions, and existing overlays are completely unaffected.
              Slider-driven before/after: uses interpolated_hyperbolic_embedding when fraction <1.
              Nodes without live DB-backed Phase 7 (X,Y) or graph metrics (no 'mean_betweenness' or 'live' source from multiHydration):
              fall back to static topological proxies (from GRAPHS) but are visually grayed out (dashed stroke, low opacity, tooltip)
              to enforce strict reliance on completed DTIE pipeline runs for the 3 branches. */}
          {showResistanceLayer && resistanceData && (
            <g>
              {/* Slider collapse visualization with live/proxy distinction */}
              {ctxCompilerState?.collapseTest && (ctxCompilerState.collapseTest.interpolated_hyperbolic_embedding || ctxCompilerState.collapseTest.new_hyperbolic_embedding) && (
                <>
                  {(() => {
                    const interp = ctxCompilerState.collapseTest.interpolated_hyperbolic_embedding || ctxCompilerState.collapseTest.new_hyperbolic_embedding || [];
                    const frac = ctxCompilerState.collapseTest.slider_fraction || 1.0;
                    return (
                      <>
                        {/* Fragmented / broken edges for the transition */}
                        {interp.filter((h: any) => h.fragmented || h.interp_fraction > 0).map((h: any, ei: number) => {
                          const baseR = h.r - (h.delta_r || 0) * (h.interp_fraction || 1);
                          const baseTheta = h.theta - 0.3;
                          const sx = cx + baseR * Math.cos(baseTheta) * radius;
                          const sy = cy - baseR * Math.sin(baseTheta) * radius;
                          const tx = cx + h.r * Math.cos(h.theta) * radius;
                          const ty = cy - h.r * Math.sin(h.theta) * radius;
                          const hasLive = !!h.mean_betweenness || h.source === 'live' || h.has_live_data;
                          return (
                            <line
                              key={`frag-edge-${ei}`}
                              x1={sx} y1={sy} x2={tx} y2={ty}
                              stroke={hasLive ? "#ef4444" : "#64748b"}
                              strokeWidth={(hasLive ? 1.5 : 1) / Math.sqrt(zoom)}
                              strokeDasharray={hasLive ? "2 3" : "4 2"}
                              opacity={hasLive ? 0.75 : 0.4}
                            />
                          );
                        })}
                        {/* Nodes: live vs proxy gray-out */}
                        {interp.map((tn: any, i: number) => {
                          const px = cx + (tn.r || 0.5) * Math.cos(tn.theta || 0) * radius;
                          const py = cy - (tn.r || 0.5) * Math.sin(tn.theta || 0) * radius;
                          const isFrag = tn.fragmented || (tn.interp_fraction || 0) > 0;
                          const hasLive = !!tn.mean_betweenness || tn.source === 'live' || tn.has_live_data;
                          const strokeCol = hasLive ? (isFrag ? "#b91c1c" : "#ef4444") : "#64748b";
                          const fillCol = hasLive ? (isFrag ? "#fee2e2" : "none") : "none";
                          const op = hasLive ? (isFrag ? 0.95 : 0.85) : 0.35;
                          return (
                            <g key={`frag-node-${i}`} onMouseEnter={() => setHovered(tn.id)} onMouseLeave={() => setHovered(null)}>
                              <circle
                                cx={px}
                                cy={py}
                                r={(isFrag ? 6 : 3.5) / Math.sqrt(zoom)}
                                fill={fillCol}
                                stroke={strokeCol}
                                strokeWidth={(isFrag ? 2.5 : (hasLive ? 1.5 : 0.8)) / Math.sqrt(zoom)}
                                opacity={op}
                                strokeDasharray={hasLive ? "none" : "2 2"}
                              />
                              {(isFrag || !hasLive) && (
                                <text x={px + 7} y={py + 2} fontSize={6.5 / Math.sqrt(zoom)} fill={hasLive ? "#991b1b" : "#475569"} opacity={hasLive ? 0.95 : 0.6}>
                                  {tn.id}{isFrag ? ` +${(tn.delta_r || 0).toFixed(2)}` : " (proxy)"}
                                </text>
                              )}
                            </g>
                          );
                        })}
                      </>
                    );
                  })()}
                </>
              )}
              {/* Base resistance edges and nodes (from resistanceData, for non-collapse or base layer) */}
              {(resistanceData.edges || []).map((e: any, ei: number) => {
                const src = (resistanceData.nodes || []).find((n: any) => n.id === e.source);
                const tgt = (resistanceData.nodes || []).find((n: any) => n.id === e.target);
                if (!src || !tgt) return null;
                const sTx = (src.r || 0.5) * Math.cos(src.theta || 0);
                const sTy = (src.r || 0.5) * Math.sin(src.theta || 0);
                const tTx = (tgt.r || 0.5) * Math.cos(tgt.theta || 0);
                const tTy = (tgt.r || 0.5) * Math.sin(tgt.theta || 0);
                const sx = cx + sTx * radius;
                const sy = cy - sTy * radius;
                const tx = cx + tTx * radius;
                const ty = cy - tTy * radius;
                return (
                  <line
                    key={`res-edge-${ei}`}
                    x1={sx} y1={sy} x2={tx} y2={ty}
                    stroke="#b91c1c"
                    strokeWidth={1.2 / Math.sqrt(zoom)}
                    strokeDasharray="3 2"
                    opacity={0.65}
                  />
                );
              })}
              {/* Base nodes */}
              {(resistanceData.nodes || []).map((tn: any, i: number) => {
                const tx = (tn.r || 0.5) * Math.cos(tn.theta || 0);
                const ty = (tn.r || 0.5) * Math.sin(tn.theta || 0);
                const px = cx + tx * radius;
                const py = cy - ty * radius;
                const isHub = (tn.centrality || 0) > 0.05;
                const hasLiveBase = !!tn.mean_betweenness || tn.source === 'live';
                return (
                  <g key={`res-topo-${i}`}>
                    <circle
                      cx={px}
                      cy={py}
                      r={isHub ? 5 / Math.sqrt(zoom) : 3.5 / Math.sqrt(zoom)}
                      fill="none"
                      stroke={hasLiveBase ? "#ef4444" : "#64748b"}
                      strokeWidth={(isHub ? 2 : 1.5) / Math.sqrt(zoom)}
                      opacity={hasLiveBase ? 0.85 : 0.4}
                      strokeDasharray={hasLiveBase ? "none" : "2 2"}
                    />
                    {isHub && (
                      <text x={px + 6} y={py + 3} fontSize={7 / Math.sqrt(zoom)} fill={hasLiveBase ? "#b91c1c" : "#475569"} opacity={hasLiveBase ? 0.9 : 0.5}>
                        {String(tn.id || "").split("_")[0]}{hasLiveBase ? "" : " (proxy)"}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* Displacement vector from primary to secondary position */}
          {overlayEnabled && selectedPoint && secondaryMatchForSelected && (() => {
            const primaryScreen = toScreen(selectedPoint.tx, selectedPoint.ty, cx, cy, radius);
            const secondaryScreen = toScreen(secondaryMatchForSelected.tx, secondaryMatchForSelected.ty, cx, cy, radius);
            return (
              <line
                x1={primaryScreen.x}
                y1={primaryScreen.y}
                x2={secondaryScreen.x}
                y2={secondaryScreen.y}
                stroke="#f59e0b"
                strokeWidth={1.5 / Math.sqrt(zoom)}
                strokeDasharray="4 2"
                opacity={0.9}
                markerEnd="url(#arrowhead-displacement)"
              />
            );
          })()}

          {/* Arrow marker for displacement vector */}
          <defs>
            <marker
              id="arrowhead-displacement"
              markerWidth="6"
              markerHeight="4"
              refX="5"
              refY="2"
              orient="auto"
            >
              <polygon points="0 0, 6 2, 0 4" fill="#f59e0b" opacity="0.9" />
            </marker>
          </defs>

          {/* Origin marker */}
          <circle cx={cx} cy={cy} r="2" fill="#475569" />

          {/* Radial zone labels */}
          <text x={cx + 4} y={cy - radius * 0.18} fill="#94a3b8" fontSize="8">
            core
          </text>
          <text x={cx + 4} y={cy - radius * 0.5} fill="#94a3b8" fontSize="8">
            mid
          </text>
          <text x={cx + 4} y={cy - radius * 0.82} fill="#94a3b8" fontSize="8">
            periphery
          </text>
        </svg>

        {/* Hover tooltip */}
        {hovered && hoverPos && (() => {
          const hp = transformedPoints.find((p) => p.residue_id === hovered);
          if (!hp) return null;
          const zone = zoneFromRadius(hp.euclidR);
          const hoverMetric =
            colorMode === "cone_depth"
              ? hp.cone_depth
              : hp.epistemic_uncertainty;
          return (
            <div
              className="absolute z-20 pointer-events-none bg-zinc-900/95 border border-zinc-700 rounded-md p-2 shadow-xl text-xs"
              style={{
                left: Math.min(hoverPos.x + 10, 300),
                top: Math.max(hoverPos.y - 72, 10),
              }}
            >
              <div className="font-semibold text-zinc-100">
                {hp.residue_id}
              </div>
              <div className="text-zinc-300">
                chain {hp.chain_label} | idx {hp.residue_index}
              </div>
              <div className="text-cyan-300 mt-1">
                {colorMode}: {hoverMetric.toFixed(4)}
              </div>
              <div className="text-zinc-400">
                hyperbolic radius: {hp.hyperbolicR.toFixed(3)}
              </div>
              <div className="text-zinc-400">zone: {zone}</div>
            </div>
          );
        })()}
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-2 px-2 py-1 text-[10px] text-zinc-500">
        <span>{colorMode}</span>
        <div className="flex h-2 w-20 rounded overflow-hidden">
          {Array.from({ length: 20 }, (_, i) => (
            <div
              key={i}
              className="flex-1"
              style={{ backgroundColor: valueToColor(i / 19, 0, 1) }}
            />
          ))}
        </div>
        <span>
          {colorRange.min.toFixed(2)} – {colorRange.max.toFixed(2)}
        </span>
        {overlayEnabled && (
          <span className="text-amber-400 ml-1">
            {secondaryLoading ? "loading…" : "● primary ○ secondary"}
          </span>
        )}
      </div>

      {/* Radial zone counts */}
      <div className="grid grid-cols-3 gap-1 px-2 pb-1 text-[10px] text-zinc-400">
        <div className="rounded border border-zinc-800 bg-zinc-900/30 px-1.5 py-0.5">
          <span className="text-zinc-500">Core</span>{" "}
          <span className="text-zinc-200">{radialZoneCounts.core}</span>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-900/30 px-1.5 py-0.5">
          <span className="text-zinc-500">Mid</span>{" "}
          <span className="text-zinc-200">{radialZoneCounts.mid}</span>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-900/30 px-1.5 py-0.5">
          <span className="text-zinc-500">Periph</span>{" "}
          <span className="text-zinc-200">{radialZoneCounts.periphery}</span>
        </div>
      </div>

      {/* Pinned residue panel */}
      {selectedPoint && (
        <div className="mx-2 mb-1 rounded border border-cyan-800/60 bg-zinc-950/80 p-2 text-[10px] space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-cyan-300 font-semibold">Pinned</span>
            <span className="text-zinc-200">{selectedPoint.residue_id}</span>
            <button
              className="ml-auto px-1.5 py-0.5 rounded border border-zinc-700 text-zinc-300 hover:text-white"
              onClick={() => setSelectedId(null)}
            >
              ×
            </button>
          </div>
          <div className="text-zinc-300">
            zone: {zoneFromRadius(selectedPoint.euclidR)} | hyp radius:{" "}
            {selectedPoint.hyperbolicR.toFixed(3)}
          </div>
          {selectedMetricValue !== null && metricMean !== null && (
            <div className="text-zinc-300">
              {colorMode}:{" "}
              <span className="text-cyan-300">
                {selectedMetricValue.toFixed(4)}
              </span>{" "}
              | vs mean:{" "}
              <span
                className={
                  selectedMetricValue >= metricMean
                    ? "text-emerald-300"
                    : "text-amber-300"
                }
              >
                {(selectedMetricValue - metricMean).toFixed(4)}
              </span>{" "}
              | percentile:{" "}
              <span className="text-cyan-300">
                {(selectedPercentile || 0).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      )}

      <div className="text-[9px] text-zinc-600 px-2 pb-1">
        Shift+drag to select region | Scroll to zoom | Ctrl+drag to pan | +/-/0 keys
        {zoom !== 1 && (
          <button
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
            className="ml-2 text-cyan-400 hover:text-cyan-300"
          >
            Reset ({zoom.toFixed(1)}×)
          </button>
        )}
      </div>
    </div>
  );
}
