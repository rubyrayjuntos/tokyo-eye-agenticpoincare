import { useCallback, useMemo } from "react";
import type { SnapshotFrom } from "xstate";

import type {
  ActivePanelName,
  PoincareColorMode,
  SelectedResidueInfo,
  StructureColorModeType,
  ViewportDirective,
} from "./types";
import type { viewportMachine, ViewportEvent } from "./viewportMachine";

type ViewportSnapshot = SnapshotFrom<typeof viewportMachine>;

export interface LegacyViewportBridge {
  highlightedResidues: string[];
  setHighlightedResidues: (residues: string[]) => void;
  currentDirective: ViewportDirective | null;
  poincareColorMode: PoincareColorMode;
  setPoincareColorMode: (mode: PoincareColorMode) => void;
  poincareSelectedResidue: SelectedResidueInfo | null;
  setPoincareSelectedResidue: (residue: SelectedResidueInfo | null) => void;
  mobiusFocus: boolean;
  setMobiusFocus: (enabled: boolean) => void;
  brushSelection: string[];
  setBrushSelection: (ids: string[]) => void;
  viewerColorMode: StructureColorModeType;
  setViewerColorMode: (mode: StructureColorModeType) => void;
  riskThreshold: number;
  setRiskThreshold: (threshold: number) => void;
  activePanel: ActivePanelName;
  setActivePanel: (panel: ActivePanelName) => void;
  isRadarActive: boolean;
  setIsRadarActive: (active: boolean) => void;
  emitDirective: (directive: ViewportDirective) => void;
}

export function useLegacyViewportBridge(
  viewportState: ViewportSnapshot,
  sendViewport: (event: ViewportEvent) => void,
): LegacyViewportBridge {
  const vp = viewportState.context;

  const setHighlightedResidues = useCallback(
    (residues: string[]) => sendViewport({ type: "USER_SELECT", residues }),
    [sendViewport],
  );

  const setPoincareColorMode = useCallback(
    (mode: PoincareColorMode) =>
      sendViewport({ type: "SET_POINCARE_COLOR_MODE", mode }),
    [sendViewport],
  );

  const setPoincareSelectedResidue = useCallback(
    (residue: SelectedResidueInfo | null) =>
      sendViewport({ type: "SET_SELECTED_RESIDUE", residue }),
    [sendViewport],
  );

  const setMobiusFocus = useCallback(
    (enabled: boolean) => sendViewport({ type: "SET_MOBIUS_FOCUS", enabled }),
    [sendViewport],
  );

  const setBrushSelection = useCallback(
    (residues: string[]) =>
      sendViewport({ type: "SET_BRUSH_SELECTION", residues }),
    [sendViewport],
  );

  const setViewerColorMode = useCallback(
    (mode: StructureColorModeType) =>
      sendViewport({ type: "SET_VIEWER_COLOR_MODE", mode }),
    [sendViewport],
  );

  const setRiskThreshold = useCallback(
    (threshold: number) =>
      sendViewport({ type: "SET_RISK_THRESHOLD", threshold }),
    [sendViewport],
  );

  const setActivePanel = useCallback(
    (panel: ActivePanelName) => sendViewport({ type: "SET_ACTIVE_PANEL", panel }),
    [sendViewport],
  );

  const setIsRadarActive = useCallback(
    (active: boolean) => sendViewport({ type: "TOGGLE_RADAR", active }),
    [sendViewport],
  );

  const emitDirective = useCallback(
    (directive: ViewportDirective) => {
      sendViewport({ type: "DIRECTIVE_RECEIVED", directive });
      if (directive.action === "clear") {
        sendViewport({ type: "TOGGLE_RADAR", active: false });
      }
    },
    [sendViewport],
  );

  return useMemo(
    () => ({
      highlightedResidues: vp.highlightedResidues,
      setHighlightedResidues,
      currentDirective: vp.currentDirective,
      poincareColorMode: vp.poincareColorMode,
      setPoincareColorMode,
      poincareSelectedResidue: vp.selectedResidue,
      setPoincareSelectedResidue,
      mobiusFocus: vp.mobiusFocusEnabled,
      setMobiusFocus,
      brushSelection: vp.brushSelectedIds,
      setBrushSelection,
      viewerColorMode: vp.viewerColorMode,
      setViewerColorMode,
      riskThreshold: vp.riskThreshold,
      setRiskThreshold,
      activePanel: vp.activePanel,
      setActivePanel,
      isRadarActive: vp.isRadarActive,
      setIsRadarActive,
      emitDirective,
    }),
    [
      vp.highlightedResidues,
      vp.currentDirective,
      vp.poincareColorMode,
      vp.selectedResidue,
      vp.mobiusFocusEnabled,
      vp.brushSelectedIds,
      vp.viewerColorMode,
      vp.riskThreshold,
      vp.activePanel,
      vp.isRadarActive,
      setHighlightedResidues,
      setPoincareColorMode,
      setPoincareSelectedResidue,
      setMobiusFocus,
      setBrushSelection,
      setViewerColorMode,
      setRiskThreshold,
      setActivePanel,
      setIsRadarActive,
      emitDirective,
    ],
  );
}