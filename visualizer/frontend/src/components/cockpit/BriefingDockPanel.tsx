import StructureBriefingPanel from "../panels/StructureBriefingPanel";
import type { ToolPanelId } from "../../lib/types";
import type { PanelManifest } from "../workbench/panelContract";

const BRIEFING_MANIFEST: PanelManifest<ToolPanelId> = {
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

export function BriefingDockPanel() {
  return <StructureBriefingPanel manifest={BRIEFING_MANIFEST} />;
}
