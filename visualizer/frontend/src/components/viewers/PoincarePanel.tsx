import { useCallback } from "react";
import { ChevronDown } from "lucide-react";
import PoincareScatter from "../PoincareScatter";
import { ResidueContextCard } from "./ResidueContextCard";
import type { SelectedResidueInfo, StructureColorModeType } from "../../lib/types";
import { VIEWPORT_COLOR_MODE_OPTIONS } from "../../lib/viewportColorMetrics";

/**
 * PoincarePanel — Discovery Cockpit panel chrome wrapping PoincareScatter
 */

export type PanelPoincareColorMode = StructureColorModeType;

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
      <div className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-slate">
        <span className="text-xs font-display tracking-wider text-magenta uppercase panel-header-glow">
          Poincaré
        </span>

        <div className="flex items-center gap-2">
          <div className="relative">
            <select
              value={colorMode}
              onChange={(e) => onColorModeChange(e.target.value as PanelPoincareColorMode)}
              className="appearance-none bg-bg-elevated border border-slate-light rounded-[var(--radius-badge)] text-[10px] text-text-secondary pl-2 pr-5 py-0.5 focus:outline-none focus:border-teal cursor-pointer"
            >
              {VIEWPORT_COLOR_MODE_OPTIONS.map((m) => (
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

      <div className="flex-1 min-h-0 relative">
        {structureId ? (
          <>
            <div className="absolute inset-4 pointer-events-none z-10">
              <div className="depth-ring w-full h-full animate-[spin_60s_linear_infinite] opacity-20" />
            </div>
            <PoincareScatter
              onSelectedResidueChange={handleSelectedResidueChange}
              onMobiusFocusChange={onMobiusFocusToggle}
              onBrushSelectionChange={onBrushSelect}
            />
            {selectedResidue && <ResidueContextCard residue={selectedResidue} />}
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
