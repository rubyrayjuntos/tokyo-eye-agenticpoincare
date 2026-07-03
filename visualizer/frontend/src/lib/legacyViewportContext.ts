import type { ViewportContext } from "./viewportMachine";
import type {
  ActivePanelName,
  PoincareColorMode,
  SelectedResidueInfo,
  StructureColorModeType,
} from "./types";

export interface LegacyViewportProps {
  poincareColorMode?: PoincareColorMode;
  poincareSelectedResidue?: SelectedResidueInfo | null;
  mobiusFocus?: boolean;
  brushSelection?: string[];
  viewerColorMode?: StructureColorModeType;
  riskThreshold?: number;
  activePanel?: ActivePanelName;
  highlightedResidues?: string[];
  isRadarActive?: boolean;
  selectedPocketId?: number | null;
}

/** Maps legacy App.tsx / AppDocked parallel state into viewport machine shape. */
export function buildLegacyViewportContext(props: LegacyViewportProps): ViewportContext {
  const highlighted = props.highlightedResidues ?? [];
  const selected = props.poincareSelectedResidue ?? null;

  return {
    highlightedResidues: highlighted,
    brushSelectedIds: props.brushSelection ?? [],
    selectedResidue: selected,
    userSelectedResidue: selected?.residue_id ?? null,
    mobiusFocusEnabled: props.mobiusFocus ?? false,
    currentDirective: null,
    activeMetric: "cone_depth",
    isRadarActive: props.isRadarActive ?? false,
    poincareColorMode: props.poincareColorMode ?? "cone_depth",
    viewerColorMode: props.viewerColorMode ?? "spectrum",
    riskThreshold: props.riskThreshold ?? 0,
    selectedPocketId: props.selectedPocketId ?? null,
    activePanel: props.activePanel ?? "briefing",
    sidebarOpen: true,
    activeEditorTab: "structure",
    bottomPanelOpen: true,
    activeBottomPanel: "summary",
    layoutModelJSON: null,
  };
}