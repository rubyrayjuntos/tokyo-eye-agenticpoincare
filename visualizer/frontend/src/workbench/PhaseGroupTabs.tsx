import clsx from "clsx";

import type { DiscoveryPhase } from "../lib/discoveryPhaseMachine";
import {
  PHASE_GROUP_CONFIG,
  PHASE_GROUP_ORDER,
  discoveryPhaseToGroup,
  type WorkbenchPhaseGroup,
} from "./phaseGroups";

interface PhaseGroupTabsProps {
  activeGroup: WorkbenchPhaseGroup;
  activeDiscoveryPhase: DiscoveryPhase;
  onSelectGroup: (group: WorkbenchPhaseGroup) => void;
  onSelectDiscoveryPhase: (phase: DiscoveryPhase) => void;
}

function formatPhaseLabel(phase: DiscoveryPhase): string {
  return phase.replace(/_/g, " ");
}

export function PhaseGroupTabs({
  activeGroup,
  activeDiscoveryPhase,
  onSelectGroup,
  onSelectDiscoveryPhase,
}: PhaseGroupTabsProps) {
  const activeConfig = PHASE_GROUP_CONFIG[activeGroup];

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <label className="block text-left text-[10px] font-bold uppercase tracking-[2px] text-[#8f82af]">
          Discovery Phase
        </label>
        {PHASE_GROUP_ORDER.map((group) => {
          const config = PHASE_GROUP_CONFIG[group];
          const isActive = activeGroup === group;
          return (
            <button
              key={group}
              type="button"
              onClick={() => onSelectGroup(group)}
              className={clsx(
                "flex w-full items-center justify-between rounded-full border p-3 text-left transition-all duration-200",
                isActive
                  ? "border-ide-accent bg-ide-accent/15 text-white shadow-[0_0_15px_rgba(157,78,221,0.25)]"
                  : "border-[#1e1930] bg-ide-bg/40 text-zinc-400 hover:bg-ide-accent/5 hover:text-zinc-200",
              )}
            >
              <span className="ml-1 font-mono text-xs font-bold uppercase tracking-wider">
                {config.label}
              </span>
              {isActive && (
                <span className="mr-1 rounded-full border border-ide-teal/20 bg-ide-teal/15 px-2 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider text-ide-teal">
                  Active
                </span>
              )}
            </button>
          );
        })}
        <p className="text-left text-[10.5px] leading-relaxed text-zinc-400">
          {activeConfig.description}
        </p>
      </div>

      <div className="space-y-1.5 border-t border-[#1e1930] pt-3">
        <label className="block text-left text-[10px] font-bold uppercase tracking-[2px] text-[#8f82af]">
          Step
        </label>
        <div className="flex flex-wrap gap-1.5">
          {activeConfig.discoveryPhases.map((phase) => {
            const isSelected = activeDiscoveryPhase === phase;
            return (
              <button
                key={phase}
                type="button"
                onClick={() => onSelectDiscoveryPhase(phase)}
                className={clsx(
                  "rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors",
                  isSelected
                    ? "border-ide-teal/40 bg-ide-teal/15 text-ide-teal"
                    : "border-[#1e1930] text-zinc-500 hover:border-ide-accent/30 hover:text-zinc-300",
                )}
              >
                {formatPhaseLabel(phase)}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function resolveActivePhaseGroup(discoveryPhase: string | undefined): WorkbenchPhaseGroup {
  if (!discoveryPhase) return "exploration";
  return discoveryPhaseToGroup(discoveryPhase as DiscoveryPhase);
}
