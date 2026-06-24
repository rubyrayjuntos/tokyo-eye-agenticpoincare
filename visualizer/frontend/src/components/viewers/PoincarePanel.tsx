import { useCallback } from "react";
import { ChevronDown } from "lucide-react";
import PoincareScatter from "../PoincareScatter";
import { ResidueContextCard } from "./ResidueContextCard";
import type { PoincareColorMode, SelectedResidueInfo } from "../../lib/types";

/**
 * PoincarePanel — Discovery Cockpit panel chrome wrapping PoincareScatter
 *
 * Panel chrome: color mode selector dropdown, Möbius focus toggle
 * Wires click handler for selection, color mode for directives, highlight state
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
 */

export type PanelPoincareColorMode =
  | "cone_depth"
  | "epistemic_uncertainty"
  | "aleatoric_uncertainty"
  | "plasticity"
  | "allosteric"
  | "resistance";

export interface PoincarePanelProps {
  structureId: string | null;
  colorMode: PanelPoincareColorMode;
  selectedResidue: SelectedResidueInfo | null;
  highlightedResidues: string[];
  mobiusFocusEnabled: boolean;
  onResidueClick: (residueId: string) => void;
  onColorModeChange: (mode: PanelPoincareColorMode) => void;
  onBrushSelect: (residueIds: string[]) => void;
  onMobiusFocusToggle: (enabled: boolean) => void;
}

const COLOR_MODES: { value: PanelPoincareColorMode; label: string }[] = [
  { value: "cone_depth", label: "Cone Depth" },
  { value: "epistemic_uncertainty", label: "Epistemic" },
  { value: "aleatoric_uncertainty", label: "Aleatoric" },
  { value: "plasticity", label: "Plasticity" },
  { value: "allosteric", label: "Allosteric" },
  { value: "resistance", label: "Resistance" },
];

export function PoincarePanel({
  structureId,
  colorMode,
  selectedResidue,
  highlightedResidues,
  mobiusFocusEnabled,
  onResidueClick,
  onColorModeChange,
  onBrushSelect,
  onMobiusFocusToggle,
}: PoincarePanelProps) {
  // Map panel color mode to PoincareScatter's internal color mode
  const scatterColorMode: PoincareColorMode =
    colorMode === "cone_depth" ? "cone_depth" : "uncertainty";

  const handleColorModeChange = useCallback(
    (mode: PoincareColorMode) => {
      // Map PoincareScatter's limited modes back to panel-level
      onColorModeChange(mode === "cone_depth" ? "cone_depth" : "epistemic_uncertainty");
    },
    [onColorModeChange],
  );

  const handleSelectedResidueChange = useCallback(
    (residue: SelectedResidueInfo | null) => {
      if (residue) {
        onResidueClick(residue.residue_id);
      }
    },
    [onResidueClick],
  );

  return (
    <div className="flex flex-col h-full">
      {/* Panel chrome header */}
      <div className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-slate">
        <span className="text-xs font-display tracking-wider text-magenta uppercase panel-header-glow">
          Poincaré
        </span>

        <div className="flex items-center gap-2">
          {/* Color mode selector */}
          <div className="relative">
            <select
              value={colorMode}
              onChange={(e) => onColorModeChange(e.target.value as PanelPoincareColorMode)}
              className="appearance-none bg-bg-elevated border border-slate-light rounded-[var(--radius-badge)] text-[10px] text-text-secondary pl-2 pr-5 py-0.5 focus:outline-none focus:border-teal cursor-pointer"
            >
              {COLOR_MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
            <ChevronDown
              size={10}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
            />
          </div>

          {/* Möbius focus toggle */}
          <button
            onClick={() => onMobiusFocusToggle(!mobiusFocusEnabled)}
            className={`text-[9px] px-1.5 py-0.5 rounded-[var(--radius-badge)] border transition-colors ${
              mobiusFocusEnabled
                ? "bg-magenta-dim/30 text-magenta border-magenta-dim"
                : "text-text-muted border-slate-light hover:border-magenta-dim"
            }`}
          >
            Möbius
          </button>
        </div>
      </div>

      {/* Poincaré scatter canvas */}
      <div className="flex-1 min-h-0 relative">
        {structureId ? (
          <>
            {/* Depth ring — shows topological depth gradient at disc boundary */}
            <div className="absolute inset-4 pointer-events-none z-10">
              <div className="depth-ring w-full h-full animate-[spin_60s_linear_infinite] opacity-20" />
            </div>
            <PoincareScatter
              onColorModeChange={handleColorModeChange}
              onSelectedResidueChange={handleSelectedResidueChange}
              onMobiusFocusChange={onMobiusFocusToggle}
              onBrushSelectionChange={onBrushSelect}
            />
            {/* Residue context card — appears on selection */}
            {selectedResidue && (
              <ResidueContextCard residue={selectedResidue} />
            )}
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-xs text-text-muted">No structure loaded</span>
          </div>
        )}
      </div>
    </div>
  );
}
