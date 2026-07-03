import { useCallback, useMemo, useState } from "react";
import { ChevronDown, Dna } from "lucide-react";

import { useResidueMetricSeries } from "../../hooks/useResidueMetricSeries";
import type { SelectedResidueInfo } from "../../lib/types";
import type { ViewportLayoutMode } from "../../lib/poincareViewportMath";
import { VIEWPORT_COLOR_MODE_OPTIONS } from "../../lib/viewportColorMetrics";
import { ColorLegend, MobiusStrip, ResidueReadout } from "../cockpit";
import { MolstarViewer } from "./MolstarViewer";
import { PoincareBallCanvas } from "./PoincareBallCanvas";
import { PoincareDiscCanvas } from "./PoincareDiscCanvas";
import type { PanelPoincareColorMode } from "./PoincarePanel";

export interface TripleViewportPanelProps {
  structureId: string | null;
  pdbId: string | null;
  colorMode: PanelPoincareColorMode;
  selectedResidue: SelectedResidueInfo | null;
  highlightedResidues: string[];
  mobiusFocusEnabled: boolean;
  onResidueClick: (residueId: string) => void;
  onColorModeChange: (mode: PanelPoincareColorMode) => void;
  onBrushSelect: (residueIds: string[]) => void;
  onMobiusFocusToggle: (enabled: boolean) => void;
  onOpenStructurePicker?: () => void;
}

const COLOR_MODES = VIEWPORT_COLOR_MODE_OPTIONS;

const VIEW_MODES: { value: ViewportLayoutMode; label: string }[] = [
  { value: "tri", label: "Tri" },
  { value: "structure", label: "Struct" },
  { value: "ball", label: "Ball" },
  { value: "disc", label: "Disc" },
];

export function TripleViewportPanel({
  structureId,
  pdbId,
  colorMode,
  selectedResidue,
  highlightedResidues,
  mobiusFocusEnabled,
  onResidueClick,
  onColorModeChange,
  onBrushSelect,
  onMobiusFocusToggle,
  onOpenStructurePicker,
}: TripleViewportPanelProps) {
  const { residues, loading, error, metricSeries } = useResidueMetricSeries(
    structureId,
    colorMode,
  );
  const [layoutMode, setLayoutMode] = useState<ViewportLayoutMode>("tri");
  const [curvature, setCurvature] = useState(1);
  const [hoveredResidueId, setHoveredResidueId] = useState<string | null>(null);
  const [rotationAngle] = useState(0);
  const [mobiusX, setMobiusX] = useState(0);
  const [mobiusY, setMobiusY] = useState(0);

  const selectedResidueId = selectedResidue?.residue_id ?? null;
  const effectivePdbId = pdbId ?? structureId;

  const colorModeLabel = useMemo(
    () => COLOR_MODES.find((m) => m.value === colorMode)?.label ?? colorMode,
    [colorMode],
  );

  const mobiusAnchor = useMemo(() => {
    if (mobiusFocusEnabled && selectedResidueId) {
      const anchor = residues.find((r) => r.residue_id === selectedResidueId);
      if (anchor) return { x: anchor.x, y: anchor.y };
    }
    return { x: mobiusX, y: mobiusY };
  }, [mobiusFocusEnabled, selectedResidueId, residues, mobiusX, mobiusY]);

  const readoutResidue = useMemo(() => {
    const id = hoveredResidueId ?? selectedResidueId;
    if (!id) return null;
    return residues.find((r) => r.residue_id === id) ?? null;
  }, [hoveredResidueId, selectedResidueId, residues]);

  const handleMobiusDrag = useCallback((x: number, y: number) => {
    const mag = Math.hypot(x, y);
    const clamped = mag > 0.92 ? { x: (x / mag) * 0.92, y: (y / mag) * 0.92 } : { x, y };
    setMobiusX(clamped.x);
    setMobiusY(clamped.y);
    if (mobiusFocusEnabled) onMobiusFocusToggle(false);
  }, [mobiusFocusEnabled, onMobiusFocusToggle]);

  const handleResidueClick = useCallback(
    (residueId: string) => {
      onResidueClick(residueId);
      onBrushSelect([residueId]);
    },
    [onResidueClick, onBrushSelect],
  );

  const handleClearSelection = useCallback(() => {
    onBrushSelect([]);
  }, [onBrushSelect]);

  const showStructure = layoutMode === "tri" || layoutMode === "structure";
  const showBall = layoutMode === "tri" || layoutMode === "ball";
  const showDisc = layoutMode === "tri" || layoutMode === "disc";

  return (
    <div className="flex h-full min-h-0 flex-col bg-bg">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-slate px-3 py-2">
        <span className="text-xs font-display uppercase tracking-wider text-teal panel-header-glow">
          Discovery Viewports
        </span>
        <div className="flex flex-wrap items-center gap-2">
          {onOpenStructurePicker ? (
            <button
              type="button"
              onClick={onOpenStructurePicker}
              className="flex items-center gap-1 rounded-[var(--radius-badge)] border border-teal-dim/40 bg-teal-dim/15 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-teal hover:bg-teal-dim/25"
            >
              <Dna size={11} />
              Switch protein
            </button>
          ) : null}
          <div className="flex items-center gap-1">
            {VIEW_MODES.map((mode) => (
              <button
                key={mode.value}
                type="button"
                onClick={() => setLayoutMode(mode.value)}
                className={`rounded-[var(--radius-badge)] border px-2 py-0.5 text-[9px] font-semibold tracking-wide ${
                  layoutMode === mode.value
                    ? "border-teal-dim bg-teal-dim/20 text-teal"
                    : "border-slate-light text-text-muted hover:border-teal-dim"
                }`}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <div className="relative">
            <select
              value={colorMode}
              onChange={(e) => onColorModeChange(e.target.value as PanelPoincareColorMode)}
              className="appearance-none rounded-[var(--radius-badge)] border border-slate-light bg-bg-elevated py-0.5 pl-2 pr-5 text-[10px] text-text-secondary focus:border-teal focus:outline-none"
            >
              {COLOR_MODES.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.label}
                </option>
              ))}
            </select>
            <ChevronDown
              size={10}
              className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted"
            />
          </div>
          <ColorLegend mode={colorMode} />
          <button
            type="button"
            onClick={() => onMobiusFocusToggle(!mobiusFocusEnabled)}
            className={`rounded-[var(--radius-badge)] border px-1.5 py-0.5 text-[9px] ${
              mobiusFocusEnabled
                ? "border-magenta-dim bg-magenta-dim/30 text-magenta"
                : "border-slate-light text-text-muted hover:border-magenta-dim"
            }`}
          >
            Möbius
          </button>
          <label className="flex items-center gap-1 text-[9px] text-text-muted">
            <span>κ</span>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={curvature}
              onChange={(e) => setCurvature(parseFloat(e.target.value))}
              className="w-16"
            />
          </label>
          {!metricSeries.dataAvailable && (
            <span
              className="text-[9px] text-warning"
              title={`${colorModeLabel} needs pipeline hydration data`}
            >
              No {colorModeLabel} data
            </span>
          )}
        </div>
      </div>

      {!structureId ? (
        <div className="flex flex-1 items-center justify-center text-xs text-text-muted">
          No structure loaded
        </div>
      ) : (
        <>
        <div
          className={`grid min-h-0 flex-1 grid-cols-1 gap-px bg-slate ${
            layoutMode === "tri" ? "md:grid-cols-3" : ""
          }`}
        >
          {showStructure && (
            <section className="relative flex min-h-[220px] min-w-0 flex-col bg-[#070810]">
              <header className="border-b border-slate/60 px-2 py-1 text-[9px] uppercase tracking-[0.18em] text-text-muted">
                Molecular Structure · Mol*
              </header>
              <div className="relative min-h-0 flex-1">
                <MolstarViewer
                  pdbId={effectivePdbId}
                  structureId={structureId}
                  colorMode={colorMode}
                  residues={residues}
                  metricSeries={metricSeries}
                  highlightedResidues={highlightedResidues}
                  selectedResidueId={selectedResidueId}
                  onResidueClick={handleResidueClick}
                  onClearSelection={handleClearSelection}
                />
              </div>
            </section>
          )}
          {showBall && (
            <section className="relative flex min-h-[220px] min-w-0 flex-col bg-bg">
              <header className="border-b border-slate/60 px-2 py-1 text-[9px] uppercase tracking-[0.18em] text-text-muted">
                Poincaré Ball · D³
              </header>
              <div className="relative min-h-0 flex-1">
                {loading ? (
                  <div className="flex h-full items-center justify-center text-xs text-text-muted">
                    Loading embeddings…
                  </div>
                ) : error ? (
                  <div className="flex h-full items-center justify-center px-4 text-center text-xs text-error">
                    {error}
                  </div>
                ) : (
                <PoincareBallCanvas
                  residues={residues}
                  colorMode={colorMode}
                  metricSeries={metricSeries}
                  highlightedResidues={highlightedResidues}
                  selectedResidueId={selectedResidueId}
                  hoveredResidueId={hoveredResidueId}
                  curvature={curvature}
                  rotationAngle={rotationAngle}
                  onResidueClick={handleResidueClick}
                  onResidueHover={setHoveredResidueId}
                />
                )}
              </div>
            </section>
          )}
          {showDisc && (
            <section className="relative flex min-h-[220px] min-w-0 flex-col bg-bg">
              <header className="border-b border-slate/60 px-2 py-1 text-[9px] uppercase tracking-[0.18em] text-text-muted">
                Poincaré Disc · D²
              </header>
              <div className="relative min-h-0 flex-1 p-2">
                {loading ? (
                  <div className="flex h-full items-center justify-center text-xs text-text-muted">
                    Loading embeddings…
                  </div>
                ) : error ? (
                  <div className="flex h-full items-center justify-center px-4 text-center text-xs text-error">
                    {error}
                  </div>
                ) : (
                <>
                <PoincareDiscCanvas
                  residues={residues}
                  colorMode={colorMode}
                  metricSeries={metricSeries}
                  highlightedResidues={highlightedResidues}
                  selectedResidueId={selectedResidueId}
                  hoveredResidueId={hoveredResidueId}
                  mobiusX={mobiusAnchor.x}
                  mobiusY={mobiusAnchor.y}
                  curvature={curvature}
                  onMobiusDrag={handleMobiusDrag}
                  onResidueClick={handleResidueClick}
                  onResidueHover={setHoveredResidueId}
                />
                <div className="pointer-events-none absolute bottom-3 left-3 right-3">
                  <ResidueReadout residue={readoutResidue} />
                </div>
                </>
                )}
              </div>
            </section>
          )}
        </div>
        <MobiusStrip
          mobiusX={mobiusAnchor.x}
          mobiusY={mobiusAnchor.y}
          onReset={() => {
            setMobiusX(0);
            setMobiusY(0);
          }}
        />
        </>
      )}
    </div>
  );
}
