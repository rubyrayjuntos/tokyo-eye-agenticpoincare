import { useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Lock } from "lucide-react";

/**
 * ToolDock — Collapsible tool sidebar for Discovery Cockpit
 *
 * Shows allowed tools as clickable items, blocked tools as grayed-out with tooltip.
 * Opens tool panels as overlays without occluding main viewers.
 * Requirements: 1.5, 5.3
 */

export interface ToolDockProps {
  allowedTools: string[];
  blockedTools: string[];
  activePanel: string | null;
  onPanelSelect: (panel: string | null) => void;
  /** Optional mapping of tool name → icon ReactNode */
  toolIcons?: Record<string, ReactNode>;
  /** Optional prerequisite tooltips for blocked tools */
  blockedReasons?: Record<string, string>;
}

export function ToolDock({
  allowedTools,
  blockedTools,
  activePanel,
  onPanelSelect,
  toolIcons,
  blockedReasons,
}: ToolDockProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className={`relative flex flex-col h-full border-r border-slate bg-bg-surface transition-[width] duration-200 ${
        collapsed ? "w-10" : "w-48"
      }`}
    >
      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="absolute -right-3 top-4 z-10 w-6 h-6 flex items-center justify-center rounded-full bg-bg-elevated border border-slate-light hover:border-teal-dim text-text-muted hover:text-teal"
        aria-label={collapsed ? "Expand tool dock" : "Collapse tool dock"}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      {/* Tool list */}
      <div className="flex-1 overflow-y-auto py-3 px-1">
        {/* Allowed tools */}
        {allowedTools.map((tool) => (
          <button
            key={tool}
            onClick={() =>
              onPanelSelect(activePanel === tool ? null : tool)
            }
            className={`w-full flex items-center gap-2 px-2 py-2 rounded-[var(--radius-button)] text-left transition-colors mb-0.5 ${
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
            {!collapsed && (
              <span className="text-xs truncate capitalize">
                {tool.replace(/_/g, " ")}
              </span>
            )}
          </button>
        ))}

        {/* Divider */}
        {blockedTools.length > 0 && (
          <div className="my-2 mx-2 h-px bg-slate-light" />
        )}

        {/* Blocked tools */}
        {blockedTools.map((tool) => (
          <div
            key={tool}
            className="w-full flex items-center gap-2 px-2 py-2 rounded-[var(--radius-button)] opacity-40 cursor-not-allowed mb-0.5"
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
            {!collapsed && (
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
