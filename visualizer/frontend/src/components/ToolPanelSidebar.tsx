import { useState, useEffect } from "react";
import { Search, GitBranch, Lightbulb, Database, BarChart3, ArrowLeftRight, TableProperties } from "lucide-react";
import CollapsiblePanel from "./controls/CollapsiblePanel";
import RCSBSearchPanel from "./panels/RCSBSearchPanel";
import GraphTopologyPanel from "./panels/GraphTopologyPanel";
import HypothesisPanel from "./panels/HypothesisPanel";
import DataToolsPanel from "./panels/DataToolsPanel";
import ProvenanceTree from "./panels/ProvenanceTree";
import PlotGeneratorPanel from "./panels/PlotGeneratorPanel";
import ComparePanel from "./panels/ComparePanel";
import DataInspectorPanel from "./panels/DataInspectorPanel";
import type { ActivePanelName } from "../lib/types";

interface ToolPanelSidebarProps {
  onActivePanelChange?: (panel: ActivePanelName) => void;
}

/**
 * Right-side tool panel sidebar with collapsible sections for each tool group.
 */
export default function ToolPanelSidebar({ onActivePanelChange }: ToolPanelSidebarProps = {}) {
  const [openPanel, setOpenPanel] = useState<ActivePanelName>(null);

  useEffect(() => {
    onActivePanelChange?.(openPanel);
  }, [openPanel, onActivePanelChange]);

  const handleToggle = (panelName: ActivePanelName) => {
    setOpenPanel((prev) => (prev === panelName ? null : panelName));
  };

  return (
    <aside className="w-80 shrink-0 border-l border-zinc-800 bg-zinc-900/60 flex flex-col overflow-y-auto">
      <div className="px-3 py-2 border-b border-zinc-800">
        <h2 className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          Tool Panels
        </h2>
      </div>

      <CollapsiblePanel
        title="RCSB Search"
        icon={<Search className="w-3 h-3" />}
        open={openPanel === "rcsb_search"}
        onToggle={() => handleToggle("rcsb_search")}
      >
        <div className="h-[400px]">
          <RCSBSearchPanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Graph Topology"
        icon={<GitBranch className="w-3 h-3" />}
        open={openPanel === "graph_topology"}
        onToggle={() => handleToggle("graph_topology")}
      >
        <div className="h-[500px]">
          <GraphTopologyPanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Hypotheses"
        icon={<Lightbulb className="w-3 h-3" />}
        open={openPanel === "hypothesis"}
        onToggle={() => handleToggle("hypothesis")}
      >
        <div className="h-[400px]">
          <HypothesisPanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Data Tools"
        icon={<Database className="w-3 h-3" />}
        open={openPanel === "data_tools"}
        onToggle={() => handleToggle("data_tools")}
      >
        <div className="h-[400px]">
          <DataToolsPanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Provenance"
        icon={<GitBranch className="w-3 h-3" />}
        open={openPanel === "provenance"}
        onToggle={() => handleToggle("provenance")}
      >
        <div className="h-[300px]">
          <ProvenanceTree />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Plot Generator"
        icon={<BarChart3 className="w-3 h-3" />}
        open={openPanel === "plot_generator"}
        onToggle={() => handleToggle("plot_generator")}
      >
        <div className="h-[350px]">
          <PlotGeneratorPanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Compare"
        icon={<ArrowLeftRight className="w-3 h-3" />}
        open={openPanel === "compare"}
        onToggle={() => handleToggle("compare")}
      >
        <div className="h-[500px]">
          <ComparePanel />
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="Data Inspector"
        icon={<TableProperties className="w-3 h-3" />}
        open={openPanel === "data_inspector"}
        onToggle={() => handleToggle("data_inspector")}
      >
        <div className="h-[500px]">
          <DataInspectorPanel />
        </div>
      </CollapsiblePanel>
    </aside>
  );
}
