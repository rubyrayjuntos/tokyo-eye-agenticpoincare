import type { ReactNode } from "react";

/**
 * CockpitLayout — Discovery Cockpit 3-panel CSS Grid layout
 *
 * Grid: [320px Poincaré] [1fr 3D Viewer] [380px Chat]
 * Responsive: collapses to stacked at <1024px
 * Slots: left (Poincaré), center (3D), right (chat), top (nav), collapsible sidebar (tools)
 *
 * Requirements: 1.1, 1.3, 1.6
 */

export interface CockpitLayoutProps {
  leftPanel: ReactNode;    // PoincarePanel
  centerPanel: ReactNode;  // MolecularPanel
  rightPanel: ReactNode;   // ChatRail
  toolDock: ReactNode;     // ToolDock (collapsible)
  navBar: ReactNode;       // NavBar
}

export function CockpitLayout({
  leftPanel,
  centerPanel,
  rightPanel,
  toolDock,
  navBar,
}: CockpitLayoutProps) {
  return (
    <div className="w-full h-screen flex flex-col bg-bg text-text-primary font-body overflow-hidden">
      {/* Top navigation bar */}
      <header className="shrink-0">{navBar}</header>

      {/* Main content area: tool dock + 3-panel grid */}
      <div className="flex flex-1 min-h-0">
        {/* Collapsible tool sidebar */}
        <aside className="shrink-0 h-full">{toolDock}</aside>

        {/* 3-panel responsive grid */}
        <main className="flex-1 min-h-0 min-w-0 grid grid-cols-1 lg:grid-cols-[320px_1fr_380px] gap-2 p-2">
          {/* Left: Poincaré disc */}
          <section className="min-h-0 min-w-0 overflow-hidden rounded-[var(--radius-panel)] bg-bg-surface border border-slate relative group hover:border-magenta-dim/50 transition-colors">
            {leftPanel}
          </section>

          {/* Center: 3D molecular viewer */}
          <section className="min-h-0 min-w-0 overflow-hidden rounded-[var(--radius-panel)] bg-bg-surface border border-slate relative group hover:border-teal-dim/50 transition-colors">
            {centerPanel}
          </section>

          {/* Right: Chat rail */}
          <section className="min-h-0 min-w-0 overflow-hidden rounded-[var(--radius-panel)] bg-bg-surface border border-slate">
            {rightPanel}
          </section>
        </main>
      </div>
    </div>
  );
}
