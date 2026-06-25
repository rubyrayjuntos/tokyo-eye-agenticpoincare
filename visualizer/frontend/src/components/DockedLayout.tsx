import React, { useState, ReactNode } from "react";
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
} from "react-resizable-panels";
import { GripVertical } from "lucide-react";
import clsx from "clsx";

/**
 * VSCode-style dockable panel layout
 * Horizontal split: Left sidebar | Center | Right panels
 * Center panel can be further split vertically for Poincaré and other visualizations
 */

interface DockPanelProps {
  title: string;
  children: ReactNode;
  className?: string;
  minimumSize?: number;
}

function DockPanel({ title, children, className = "" }: DockPanelProps) {
  return (
    <div className={clsx(
      "flex flex-col h-full bg-gradient-to-br from-[#0a0e27] to-[#0f1335] border border-cyan-900/30 rounded-lg overflow-hidden",
      className
    )}>
      <div className="flex items-center gap-2 px-4 py-3 bg-black/40 border-b border-cyan-900/20">
        <GripVertical size={14} className="text-cyan-500/40" />
        <h3 className="text-sm font-bold text-cyan-100 font-rajdhani uppercase tracking-wider">
          {title}
        </h3>
      </div>
      <div className="flex-1 overflow-auto">
        {children}
      </div>
    </div>
  );
}

interface DockedLayoutProps {
  poincare: ReactNode;
  molecularViewer: ReactNode;
  agentChat: ReactNode;
  sidebarLeft?: ReactNode;
}

export function DockedLayout({
  poincare,
  molecularViewer,
  agentChat,
  sidebarLeft,
}: DockedLayoutProps) {
  return (
    <div className="w-full h-screen bg-[#0a0e27] flex flex-col">
      {/* Main Docked Layout */}
      <PanelGroup direction="horizontal" className="flex-1">
        {/* Left Sidebar (Optional) */}
        {sidebarLeft && (
          <>
            <Panel defaultSize={15} minSize={10} maxSize={30} className="hidden lg:flex lg:flex-col">
              <div className="flex-1 overflow-auto p-4 bg-gradient-to-br from-[#050810] to-[#0a0e27]">
                {sidebarLeft}
              </div>
            </Panel>
            <PanelResizeHandle className="w-1 bg-gradient-to-b from-transparent via-cyan-500/20 to-transparent hover:via-cyan-500/60 transition-colors" />
          </>
        )}

        {/* Center: Poincaré + Molecular Viewer (Vertical Split) */}
        <Panel defaultSize={65} minSize={30}>
          <PanelGroup direction="vertical">
            {/* Top: Poincaré */}
            <Panel defaultSize={50} minSize={25}>
              <DockPanel title="Poincaré Manifold">
                {poincare}
              </DockPanel>
            </Panel>

            <PanelResizeHandle className="h-1 bg-gradient-to-r from-transparent via-purple-500/20 to-transparent hover:via-purple-500/60 transition-colors" />

            {/* Bottom: Molecular Viewer */}
            <Panel defaultSize={50} minSize={25}>
              <DockPanel title="Molecular Structure">
                {molecularViewer}
              </DockPanel>
            </Panel>
          </PanelGroup>
        </Panel>

        <PanelResizeHandle className="w-1 bg-gradient-to-b from-transparent via-blue-500/20 to-transparent hover:via-blue-500/60 transition-colors" />

        {/* Right: Agent Chat */}
        <Panel defaultSize={20} minSize={15} maxSize={40}>
          <DockPanel title="Agent Chat">
            {agentChat}
          </DockPanel>
        </Panel>
      </PanelGroup>
    </div>
  );
}
