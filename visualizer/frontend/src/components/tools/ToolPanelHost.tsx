import type { ActivePanelName } from "../../lib/types";
import { getToolPanelDefinition } from "../workbench/panelRegistry";

export function ToolPanelHost({ activePanel }: { activePanel: ActivePanelName }) {
  const panel = getToolPanelDefinition(activePanel);

  if (!panel) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-text-muted">
        Select a tool from the activity bar to open a docked panel.
      </div>
    );
  }

  return <div className="h-full overflow-auto">{panel.render()}</div>;
}
