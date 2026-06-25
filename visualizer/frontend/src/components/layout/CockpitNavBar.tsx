import { Wifi, WifiOff, RefreshCw } from "lucide-react";
import type { DiscoveryPhase } from "../../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../../lib/hypothesisLifecycleMachine";

/**
 * CockpitNavBar — Top navigation bar for Discovery Cockpit
 *
 * Displays: structure badge, 6-phase stepper, hypothesis lifecycle badge, connection indicator
 * Requirements: 1.4
 */

export interface CockpitNavBarProps {
  structureId: string | null;
  pdbId: string | null;
  discoveryPhase: DiscoveryPhase;
  hypothesisLifecycle: HypothesisLifecycleState;
  connectionStatus: "connected" | "disconnected" | "reconnecting";
  onStructureSearch?: () => void;
}

const PHASES: { key: DiscoveryPhase; label: string }[] = [
  { key: "residue", label: "Residue" },
  { key: "topology", label: "Topology" },
  { key: "structure", label: "Structure" },
  { key: "pocket", label: "Pocket" },
  { key: "screening", label: "Screening" },
  { key: "report", label: "Report" },
];

const LIFECYCLE_COLORS: Record<HypothesisLifecycleState, string> = {
  emergent: "bg-text-muted",
  framed: "bg-teal-dim",
  testing: "bg-warning",
  supported: "bg-success",
  contradicted: "bg-error",
  revised: "bg-magenta-dim",
  synthesized: "bg-teal",
};

export function CockpitNavBar({
  structureId,
  pdbId,
  discoveryPhase,
  hypothesisLifecycle,
  connectionStatus,
  onStructureSearch,
}: CockpitNavBarProps) {
  const currentPhaseIndex = PHASES.findIndex((p) => p.key === discoveryPhase);

  return (
    <nav className="h-12 flex items-center justify-between px-4 border-b border-slate bg-bg-surface/80 backdrop-blur-sm">
      {/* Left: Branding + Structure badge */}
      <div className="flex items-center gap-4">
        <span className="font-display text-sm tracking-wider text-teal">
          Tokyo Eye
        </span>

        {structureId ? (
          <button
            onClick={onStructureSearch}
            className="flex items-center gap-2 px-2.5 py-1 rounded-[var(--radius-badge)] bg-bg-elevated border border-slate-light hover:border-teal-dim"
          >
            <span className="text-xs font-mono text-teal-bright">
              {pdbId?.toUpperCase() ?? "—"}
            </span>
            <span className="text-[10px] text-text-muted truncate max-w-[120px]">
              {structureId}
            </span>
          </button>
        ) : (
          <button
            onClick={onStructureSearch}
            className="text-xs text-text-muted hover:text-text-secondary px-2 py-1 rounded-[var(--radius-badge)] border border-dashed border-slate-light hover:border-teal-dim"
          >
            Load structure…
          </button>
        )}
      </div>

      {/* Center: Phase stepper */}
      <div className="hidden md:flex items-center gap-1">
        {PHASES.map((phase, i) => {
          const isActive = i === currentPhaseIndex;
          const isCompleted = i < currentPhaseIndex;
          return (
            <div key={phase.key} className="flex items-center gap-1">
              {i > 0 && (
                <div
                  className={`w-4 h-px ${
                    isCompleted ? "bg-teal" : "bg-slate-lighter"
                  }`}
                />
              )}
              <div className="flex flex-col items-center gap-0.5">
                <div
                  className={`w-2.5 h-2.5 rounded-full border transition-colors ${
                    isActive
                      ? "bg-teal border-teal phase-active"
                      : isCompleted
                      ? "bg-teal-dim border-teal-dim phase-completed"
                      : "bg-transparent border-slate-lighter"
                  }`}
                />
                <span
                  className={`text-[9px] uppercase tracking-wide ${
                    isActive
                      ? "text-teal font-bold"
                      : isCompleted
                      ? "text-text-secondary"
                      : "text-text-muted"
                  }`}
                >
                  {phase.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Right: Lifecycle badge + connection status */}
      <div className="flex items-center gap-3">
        {/* Hypothesis lifecycle badge */}
        <div className="flex items-center gap-1.5 px-2 py-1 rounded-[var(--radius-badge)] bg-bg-elevated border border-slate-light">
          <div
            className={`w-2 h-2 rounded-full ${LIFECYCLE_COLORS[hypothesisLifecycle]}`}
          />
          <span className="text-[10px] uppercase tracking-wider text-text-secondary">
            {hypothesisLifecycle}
          </span>
        </div>

        {/* Connection indicator */}
        <div className="flex items-center gap-1.5">
          {connectionStatus === "connected" ? (
            <Wifi size={14} className="text-success" />
          ) : connectionStatus === "reconnecting" ? (
            <RefreshCw size={14} className="text-warning animate-spin" />
          ) : (
            <WifiOff size={14} className="text-error" />
          )}
          <span className="text-[10px] text-text-muted">
            {connectionStatus === "connected"
              ? "Live"
              : connectionStatus === "reconnecting"
              ? "Reconnecting"
              : "Offline"}
          </span>
        </div>
      </div>
    </nav>
  );
}
