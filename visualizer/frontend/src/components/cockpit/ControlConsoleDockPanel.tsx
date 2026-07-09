/** Dockview wrapper for Control Console (developer control center). */
import PrototypeControlConsolePanel from "../panels/PrototypeControlConsolePanel";
import type { ToolPanelId } from "../../lib/types";
import type { PanelManifest } from "../workbench/panelContract";

const MANIFEST: PanelManifest<ToolPanelId> = {
  id: "control_console",
  title: "Control Console",
  region: "activity-sidebar",
  ports: [{ name: "open-panel", kind: "open-panel", direction: "output" }],
  emits: ["panel.activated"],
};

export default function ControlConsoleDockPanel() {
  return (
    <div className="h-full overflow-auto">
      <PrototypeControlConsolePanel manifest={MANIFEST} />
    </div>
  );
}
