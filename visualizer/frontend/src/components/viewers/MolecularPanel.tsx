import { useCallback } from "react";
import { ChevronDown } from "lucide-react";
import MolecularViewer from "../MolecularViewer";
import type { StructureColorModeType, SelectedResidueInfo } from "../../lib/types";

/**
 * MolecularPanel — Discovery Cockpit panel chrome wrapping MolecularViewer
 *
 * Panel chrome: color mode selector, highlight info bar
 * Wires click handler for selection, camera animation, highlight state
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */

export type MolecularColorMode =
  | "spectrum"
  | "cone_depth"
  | "epistemic"
  | "aleatoric"
  | "plasticity"
  | "allosteric"
  | "resistance";

export interface MolecularPanelProps {
  structureId: string | null;
  colorMode: MolecularColorMode;
  selectedResidue: SelectedResidueInfo | null;
  highlightedResidues: string[];
  onResidueClick: (residueId: string) => void;
  onColorModeChange: (mode: MolecularColorMode) => void;
}

const COLOR_MODES: { value: MolecularColorMode; label: string }[] = [
  { value: "spectrum", label: "Spectrum" },
  { value: "cone_depth", label: "Cone Depth" },
  { value: "epistemic", label: "Epistemic" },
  { value: "aleatoric", label: "Aleatoric" },
  { value: "plasticity", label: "Plasticity" },
  { value: "allosteric", label: "Allosteric" },
  { value: "resistance", label: "Resistance" },
];

export function MolecularPanel({
  structureId,
  colorMode,
  selectedResidue,
  highlightedResidues,
  onResidueClick,
  onColorModeChange,
}: MolecularPanelProps) {
  const handleColorModeChange = useCallback(
    (mode: StructureColorModeType) => {
      onColorModeChange(mode as MolecularColorMode);
    },
    [onColorModeChange],
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Panel chrome header */}
      <div className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-slate">
        <span className="text-xs font-display tracking-wider text-teal uppercase panel-header-glow">
          3D Structure
        </span>

        <div className="flex items-center gap-2">
          {/* Color mode selector */}
          <div className="relative">
            <select
              value={colorMode}
              onChange={(e) => onColorModeChange(e.target.value as MolecularColorMode)}
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

          {/* Highlight info */}
          {highlightedResidues.length > 0 && (
            <span className="text-[9px] text-teal-dim px-1.5 py-0.5 rounded-[var(--radius-badge)] bg-teal-dim/10 border border-teal-dim/30 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-teal selection-pulse" />
              {highlightedResidues.length} highlighted
            </span>
          )}
        </div>
      </div>

      {/* Molecular viewer */}
      <div className="relative flex-1 min-h-0">
        {structureId ? (
          <MolecularViewer
            highlightResidues={highlightedResidues}
            onColorModeChange={handleColorModeChange}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-xs text-text-muted">No structure loaded</span>
          </div>
        )}
      </div>
    </div>
  );
}
