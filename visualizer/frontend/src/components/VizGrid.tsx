import type { ReactNode } from "react";
import PoincareScatter from "./PoincareScatter";
import ResidueBarChart from "./ResidueBarChart";
import LatentSpace3D from "./LatentSpace3D";
import MolecularViewer from "./MolecularViewer";
import { useDashboard } from "../lib/context";
import type { SelectedResidueInfo, PoincareColorMode, StructureColorModeType } from "../lib/types";

interface VizGridProps {
  onPoincareColorModeChange?: (mode: PoincareColorMode) => void;
  onPoincareSelectedResidueChange?: (residue: SelectedResidueInfo | null) => void;
  onMobiusFocusChange?: (enabled: boolean) => void;
  onBrushSelectionChange?: (ids: string[]) => void;
  onViewerColorModeChange?: (mode: StructureColorModeType) => void;
  onRiskThresholdChange?: (threshold: number) => void;
}

/**
 * VizGrid — 2x2 visualization grid with real components.
 */
export default function VizGrid({
  onPoincareColorModeChange,
  onPoincareSelectedResidueChange,
  onMobiusFocusChange,
  onBrushSelectionChange,
  onViewerColorModeChange,
  onRiskThresholdChange,
}: VizGridProps = {}) {
  const { highlightedResidues } = useDashboard();

  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-3 flex-1 min-h-0">
      <VizPanel>
        <PoincareScatter
          onColorModeChange={onPoincareColorModeChange}
          onSelectedResidueChange={onPoincareSelectedResidueChange}
          onMobiusFocusChange={onMobiusFocusChange}
          onBrushSelectionChange={onBrushSelectionChange}
        />
      </VizPanel>
      <VizPanel>
        <ResidueBarChart />
      </VizPanel>
      <VizPanel>
        <LatentSpace3D />
      </VizPanel>
      <VizPanel>
        <MolecularViewer
          highlightResidues={highlightedResidues}
          onColorModeChange={onViewerColorModeChange}
          onRiskThresholdChange={onRiskThresholdChange}
        />
      </VizPanel>
    </div>
  );
}

function VizPanel({ children }: { children: ReactNode }) {
  return (
    <div className="border border-zinc-800 rounded-lg bg-zinc-900/40 relative overflow-hidden">
      <div className="relative z-10 h-full">{children}</div>
    </div>
  );
}
