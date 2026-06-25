import type { BottomPanelId } from "../../lib/types";
import {
  BOTTOM_PANEL_REGISTRY,
  getBottomPanelDefinition,
} from "./panelRegistry";

interface BottomPanelHostProps {
  activePanel: BottomPanelId;
  onSelectPanel: (panel: BottomPanelId) => void;
}

export function BottomPanelHost({
  activePanel,
  onSelectPanel,
}: BottomPanelHostProps) {
  const panel = getBottomPanelDefinition(activePanel);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-1 border-b border-slate px-2 py-1.5">
        {Object.values(BOTTOM_PANEL_REGISTRY).map((definition) => {
          const active = definition.id === activePanel;

          return (
            <button
              key={definition.id}
              onClick={() => onSelectPanel(definition.id)}
              className={`rounded-t-[var(--radius-badge)] border border-b-0 px-3 py-1 text-[10px] uppercase tracking-[0.18em] transition-colors ${
                active
                  ? "border-teal-dim/50 bg-bg-elevated text-teal"
                  : "border-slate-light bg-bg text-text-secondary hover:text-text-primary"
              }`}
            >
              {definition.title}
            </button>
          );
        })}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-2">{panel.render()}</div>
    </div>
  );
}
