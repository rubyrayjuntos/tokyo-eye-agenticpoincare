import type { ReactNode } from "react";
import { Lock } from "lucide-react";

/**
 * ToolDock — Collapsible tool sidebar for Discovery Cockpit
 *
 * Shows allowed tools as clickable items, blocked tools as grayed-out with tooltip.
 * Opens tool panels as overlays without occluding main viewers.
 * Requirements: 1.5, 5.3
 */

export interface ToolDockProps {
  pinnedTools?: string[];
  allowedTools: string[];
  blockedTools: string[];
  activePanel: string | null;
  onPanelSelect: (panel: string | null) => void;
  mode?: "activity-bar" | "sidebar";
  /** Optional mapping of tool name → icon ReactNode */
  toolIcons?: Record<string, ReactNode>;
  /** Optional prerequisite tooltips for blocked tools */
  blockedReasons?: Record<string, string>;
}

export function ToolDock({
  pinnedTools = [],
  allowedTools,
  blockedTools,
  activePanel,
  onPanelSelect,
  mode = "sidebar",
  toolIcons,
  blockedReasons,
}: ToolDockProps) {
  const compact = mode === "activity-bar";
  const visibleAllowedTools = [
    ...pinnedTools,
    ...allowedTools.filter((tool) => !pinnedTools.includes(tool)),
  ];

  return (
    <div
      className={`flex h-full flex-col border-r border-slate bg-bg-surface ${
        compact ? "w-12 items-center py-2" : "w-48"
      }`}
    >
      <div
        className={`flex-1 overflow-y-auto ${
          compact ? "flex flex-col items-center gap-1 px-1" : "py-3 px-1"
        }`}
      >
        {/* Allowed tools */}
        {visibleAllowedTools.map((tool) => (
          <button
            key={tool}
            onClick={() =>
              onPanelSelect(activePanel === tool ? null : tool)
            }
            className={`${
              compact ? "h-10 w-10 justify-center px-0" : "mb-0.5 w-full px-2 py-2 text-left"
            } flex items-center gap-2 rounded-[var(--radius-button)] border transition-colors ${
              activePanel === tool
                ? "bg-teal-dim/30 text-teal border border-teal-dim"
                : "hover:bg-slate/50 text-text-secondary hover:text-text-primary border border-transparent"
            }`}
            title={tool}
          >
            {toolIcons?.[tool] ?? (
              <div className="w-5 h-5 rounded bg-slate-light flex items-center justify-center text-[10px] text-text-muted">
                {tool.charAt(0).toUpperCase()}
              </div>
            )}
            {!compact && (
              <span className="text-xs truncate capitalize">
                {tool.replace(/_/g, " ")}
              </span>
            )}
          </button>
        ))}

        {/* Divider */}
        {blockedTools.length > 0 && (
          <div className={`${compact ? "my-2 h-px w-6" : "mx-2 my-2 h-px"} bg-slate-light`} />
        )}

        {/* Blocked tools */}
        {blockedTools.map((tool) => (
          <div
            key={tool}
            className={`${
              compact ? "h-10 w-10 justify-center px-0" : "mb-0.5 w-full px-2 py-2"
            } flex items-center gap-2 rounded-[var(--radius-button)] opacity-40 cursor-not-allowed`}
            title={
              blockedReasons?.[tool] ??
              `Prerequisite not met for "${tool.replace(/_/g, " ")}"`
            }
          >
            {toolIcons?.[tool] ?? (
              <div className="w-5 h-5 rounded bg-slate-light flex items-center justify-center text-[10px] text-text-muted">
                {tool.charAt(0).toUpperCase()}
              </div>
            )}
            {!compact && (
              <>
                <span className="text-xs truncate capitalize text-text-muted">
                  {tool.replace(/_/g, " ")}
                </span>
                <Lock size={10} className="ml-auto text-text-muted shrink-0" />
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
