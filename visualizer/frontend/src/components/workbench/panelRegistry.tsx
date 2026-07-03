import type { ReactNode } from "react";

import AgentTelemetryPanel from "../AgentTelemetryPanel";
import ResultsTable from "../ResultsTable";
import StructureBriefingPanel from "../panels/StructureBriefingPanel";
import PrototypeControlConsolePanel from "../panels/PrototypeControlConsolePanel";
import PrototypeFragmentsPanel from "../panels/PrototypeFragmentsPanel";
import PrototypeHypothesesDockPanel from "../panels/PrototypeHypothesesDockPanel";
import PrototypeMotifsPanel from "../panels/PrototypeMotifsPanel";
import PrototypePocketsPanel from "../panels/PrototypePocketsPanel";
import PrototypeSummaryPanel from "../panels/PrototypeSummaryPanel";
import RCSBSearchPanel from "../panels/RCSBSearchPanel";
import GraphTopologyPanel from "../panels/GraphTopologyPanel";
import HypothesisPanel from "../panels/HypothesisPanel";
import DataToolsPanel from "../panels/DataToolsPanel";
import ProvenanceTree from "../panels/ProvenanceTree";
import PlotGeneratorPanel from "../panels/PlotGeneratorPanel";
import ComparePanel from "../panels/ComparePanel";
import DataInspectorPanel from "../panels/DataInspectorPanel";
import type {
  ActivePanelName,
  BottomPanelId,
  EditorTabId,
  ToolPanelId,
} from "../../lib/types";
import type { PanelManifest, WorkbenchRegion } from "./panelContract";

export interface WorkbenchPanelDefinition<TId extends string> {
  id: TId;
  title: string;
  region: WorkbenchRegion;
  manifest: PanelManifest<TId>;
  render: () => ReactNode;
}

const BRIEFING_PANEL_MANIFEST: PanelManifest<"briefing"> = {
  id: "briefing",
  title: "Briefing",
  region: "activity-sidebar",
  ports: [
    { name: "selection", kind: "selection", direction: "output" },
    { name: "open-panel", kind: "open-panel", direction: "bidirectional" },
  ],
  subscribesTo: ["selection.changed", "panel.activated"],
  emits: ["selection.changed", "panel.activated"],
};

const CONTROL_CONSOLE_PANEL_MANIFEST: PanelManifest<"control_console"> = {
  id: "control_console",
  title: "Control Console",
  region: "activity-sidebar",
  ports: [
    { name: "metric", kind: "metric", direction: "output" },
    { name: "open-panel", kind: "open-panel", direction: "output" },
    {
      name: "open-bottom-panel",
      kind: "open-bottom-panel",
      direction: "output",
    },
  ],
  emits: ["directive.emitted", "panel.activated", "bottom-panel.activated"],
};

const SUMMARY_PANEL_MANIFEST: PanelManifest<"summary"> = {
  id: "summary",
  title: "Summary",
  region: "bottom-panel",
  ports: [{ name: "open-bottom-panel", kind: "open-bottom-panel", direction: "output" }],
  emits: ["bottom-panel.activated"],
};

const HYPOTHESES_PANEL_MANIFEST: PanelManifest<"hypotheses"> = {
  id: "hypotheses",
  title: "Hypotheses",
  region: "bottom-panel",
  ports: [{ name: "selection", kind: "selection", direction: "output" }],
  emits: ["selection.changed"],
};

const POCKETS_PANEL_MANIFEST: PanelManifest<"pockets"> = {
  id: "pockets",
  title: "Pockets",
  region: "bottom-panel",
  ports: [{ name: "directive", kind: "directive", direction: "output" }],
  emits: ["directive.emitted"],
};

const FRAGMENTS_PANEL_MANIFEST: PanelManifest<"fragments"> = {
  id: "fragments",
  title: "Fragments",
  region: "bottom-panel",
  ports: [{ name: "directive", kind: "directive", direction: "output" }],
  emits: ["directive.emitted"],
};

const MOTIFS_PANEL_MANIFEST: PanelManifest<"motifs"> = {
  id: "motifs",
  title: "Motifs",
  region: "bottom-panel",
  ports: [{ name: "selection", kind: "selection", direction: "output" }],
  emits: ["selection.changed"],
};

export const TOOL_PANEL_REGISTRY: Record<ToolPanelId, WorkbenchPanelDefinition<ToolPanelId>> = {
  briefing: {
    id: "briefing",
    title: "Briefing",
    region: "activity-sidebar",
    manifest: BRIEFING_PANEL_MANIFEST,
    render: () => <StructureBriefingPanel manifest={BRIEFING_PANEL_MANIFEST} />,
  },
  control_console: {
    id: "control_console",
    title: "Control Console",
    region: "activity-sidebar",
    manifest: CONTROL_CONSOLE_PANEL_MANIFEST,
    render: () => (
      <PrototypeControlConsolePanel manifest={CONTROL_CONSOLE_PANEL_MANIFEST} />
    ),
  },
  rcsb_search: {
    id: "rcsb_search",
    title: "RCSB Search",
    region: "activity-sidebar",
    manifest: {
      id: "rcsb_search",
      title: "RCSB Search",
      region: "activity-sidebar",
      ports: [{ name: "open-panel", kind: "open-panel", direction: "bidirectional" }],
      subscribesTo: ["panel.activated"],
      emits: ["panel.activated"],
    },
    render: () => <RCSBSearchPanel />,
  },
  graph_topology: {
    id: "graph_topology",
    title: "Graph Topology",
    region: "activity-sidebar",
    manifest: {
      id: "graph_topology",
      title: "Graph Topology",
      region: "activity-sidebar",
      ports: [
        { name: "selection", kind: "selection", direction: "bidirectional" },
        { name: "metric", kind: "metric", direction: "output" },
      ],
      subscribesTo: ["selection.changed"],
      emits: ["selection.changed", "directive.emitted"],
    },
    render: () => <GraphTopologyPanel />,
  },
  hypothesis: {
    id: "hypothesis",
    title: "Hypotheses",
    region: "activity-sidebar",
    manifest: {
      id: "hypothesis",
      title: "Hypotheses",
      region: "activity-sidebar",
      ports: [{ name: "directive", kind: "directive", direction: "output" }],
      emits: ["directive.emitted"],
    },
    render: () => <HypothesisPanel />,
  },
  data_tools: {
    id: "data_tools",
    title: "Data Tools",
    region: "activity-sidebar",
    manifest: {
      id: "data_tools",
      title: "Data Tools",
      region: "activity-sidebar",
      ports: [{ name: "layout", kind: "layout", direction: "output" }],
      emits: ["layout.changed"],
    },
    render: () => <DataToolsPanel />,
  },
  provenance: {
    id: "provenance",
    title: "Provenance",
    region: "activity-sidebar",
    manifest: {
      id: "provenance",
      title: "Provenance",
      region: "activity-sidebar",
      ports: [{ name: "open-bottom-panel", kind: "open-bottom-panel", direction: "output" }],
      emits: ["bottom-panel.activated"],
    },
    render: () => <ProvenanceTree />,
  },
  plot_generator: {
    id: "plot_generator",
    title: "Plot Generator",
    region: "activity-sidebar",
    manifest: {
      id: "plot_generator",
      title: "Plot Generator",
      region: "activity-sidebar",
      ports: [{ name: "open-editor", kind: "open-editor", direction: "output" }],
      emits: ["editor.activated"],
    },
    render: () => <PlotGeneratorPanel />,
  },
  compare: {
    id: "compare",
    title: "Compare",
    region: "activity-sidebar",
    manifest: {
      id: "compare",
      title: "Compare",
      region: "activity-sidebar",
      ports: [{ name: "selection", kind: "selection", direction: "output" }],
      subscribesTo: ["selection.changed"],
      emits: ["selection.changed"],
    },
    render: () => <ComparePanel />,
  },
  data_inspector: {
    id: "data_inspector",
    title: "Data Inspector",
    region: "activity-sidebar",
    manifest: {
      id: "data_inspector",
      title: "Data Inspector",
      region: "activity-sidebar",
      ports: [
        { name: "toggle", kind: "toggle", direction: "output" },
        { name: "directive", kind: "directive", direction: "output" },
      ],
      emits: ["panel.visibility.changed", "directive.emitted"],
    },
    render: () => <DataInspectorPanel />,
  },
};

export const BOTTOM_PANEL_REGISTRY: Record<BottomPanelId, WorkbenchPanelDefinition<BottomPanelId>> = {
  summary: {
    id: "summary",
    title: "Summary",
    region: "bottom-panel",
    manifest: SUMMARY_PANEL_MANIFEST,
    render: () => <PrototypeSummaryPanel manifest={SUMMARY_PANEL_MANIFEST} />,
  },
  hypotheses: {
    id: "hypotheses",
    title: "Hypotheses",
    region: "bottom-panel",
    manifest: HYPOTHESES_PANEL_MANIFEST,
    render: () => (
      <PrototypeHypothesesDockPanel manifest={HYPOTHESES_PANEL_MANIFEST} />
    ),
  },
  pockets: {
    id: "pockets",
    title: "Pockets",
    region: "bottom-panel",
    manifest: POCKETS_PANEL_MANIFEST,
    render: () => <PrototypePocketsPanel manifest={POCKETS_PANEL_MANIFEST} />,
  },
  fragments: {
    id: "fragments",
    title: "Fragments",
    region: "bottom-panel",
    manifest: FRAGMENTS_PANEL_MANIFEST,
    render: () => <PrototypeFragmentsPanel manifest={FRAGMENTS_PANEL_MANIFEST} />,
  },
  motifs: {
    id: "motifs",
    title: "Motifs",
    region: "bottom-panel",
    manifest: MOTIFS_PANEL_MANIFEST,
    render: () => <PrototypeMotifsPanel manifest={MOTIFS_PANEL_MANIFEST} />,
  },
  results: {
    id: "results",
    title: "Results",
    region: "bottom-panel",
    manifest: {
      id: "results",
      title: "Results",
      region: "bottom-panel",
      ports: [
        { name: "selection", kind: "selection", direction: "bidirectional" },
        { name: "open-editor", kind: "open-editor", direction: "output" },
      ],
      subscribesTo: ["selection.changed"],
      emits: ["selection.changed", "editor.activated"],
    },
    render: () => <ResultsTable />,
  },
  telemetry: {
    id: "telemetry",
    title: "Telemetry",
    region: "bottom-panel",
    manifest: {
      id: "telemetry",
      title: "Telemetry",
      region: "bottom-panel",
      ports: [{ name: "toggle", kind: "toggle", direction: "output" }],
      emits: ["panel.visibility.changed"],
    },
    render: () => <AgentTelemetryPanel />,
  },
};

export const EDITOR_TAB_REGISTRY: Record<EditorTabId, { id: EditorTabId; title: string }> = {
  manifold: { id: "manifold", title: "Manifold" },
  structure: { id: "structure", title: "Structure" },
  agent: { id: "agent", title: "Agent" },
};

export function getToolPanelDefinition(activePanel: ActivePanelName) {
  if (!activePanel) {
    return null;
  }

  return TOOL_PANEL_REGISTRY[activePanel];
}

export function getBottomPanelDefinition(activePanel: BottomPanelId) {
  return BOTTOM_PANEL_REGISTRY[activePanel];
}
