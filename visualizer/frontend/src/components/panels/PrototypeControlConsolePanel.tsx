import { useMemo } from "react";

import { PipelineAuditReadout } from "../cockpit/PipelineAuditReadout";
import { useDashboard } from "../../lib/context";
import { useDiscoveryPhaseActions } from "../../lib/useDiscoveryPhaseActions";
import type {
  DiscoveryPhase,
} from "../../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../../lib/hypothesisLifecycleMachine";
import type { ToolPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

const DISCOVERY_PHASES: DiscoveryPhase[] = [
  "residue",
  "topology",
  "structure",
  "pocket",
  "screening",
  "report",
];

const HYPOTHESIS_STATES: HypothesisLifecycleState[] = [
  "emergent",
  "framed",
  "testing",
  "supported",
  "contradicted",
  "revised",
  "synthesized",
];

export default function PrototypeControlConsolePanel({
  manifest,
}: {
  manifest: PanelManifest<ToolPanelId>;
}) {
  const {
    plannerPolicy,
    discoveryContext,
    hypothesisContext,
    sendHypothesis,
  } = useDashboard();
  const { requestDiscoveryPhase, isBackendAuthority } = useDiscoveryPhaseActions();
  const panelBroker = usePanelBroker();

  const toolGroups = useMemo(
    () => [
      {
        label: "Allowed",
        tools: plannerPolicy?.allowedTools ?? [],
      },
      {
        label: "Blocked",
        tools: plannerPolicy?.blockedTools ?? [],
      },
    ],
    [plannerPolicy?.allowedTools, plannerPolicy?.blockedTools],
  );

  const setLifecycle = (state: HypothesisLifecycleState) => {
    switch (state) {
      case "emergent":
        sendHypothesis({ type: "RESET_LIFECYCLE" });
        break;
      case "framed":
        sendHypothesis({
          type: "START_HYPOTHESIS",
          hypothesisText: hypothesisContext?.hypothesisText ?? "",
          source: "user",
        });
        break;
      case "testing":
        sendHypothesis({ type: "BEGIN_TESTING" });
        break;
      case "supported":
        sendHypothesis({ type: "MARK_SUPPORTED" });
        break;
      case "contradicted":
        sendHypothesis({
          type: "ADD_CONTRADICTION",
          reason: "manual control console override",
        });
        break;
      case "revised":
        sendHypothesis({
          type: "REVISE_HYPOTHESIS",
          hypothesisText:
            hypothesisContext?.hypothesisText ?? "Revised hypothesis",
        });
        break;
      case "synthesized":
        sendHypothesis({ type: "MARK_SYNTHESIZED" });
        break;
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0d14] text-text-primary">
      <div className="border-b border-slate px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.22em] text-text-secondary">
          Manual Control
        </div>
      </div>
      <div className="ck-scroll flex-1 overflow-y-auto px-4 py-4">
        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Viewport Metric
          </div>
          <div className="flex flex-wrap gap-2">
            {["cone_depth", "epistemic", "aleatoric"].map((metric) => (
              <button
                key={metric}
                onClick={() =>
                  panelBroker.emitPort(manifest, "metric", { metric })
                }
                className="rounded border border-slate px-2.5 py-1 font-mono text-[10px] text-text-primary transition-colors hover:border-teal-dim/40 hover:text-teal"
              >
                {metric}
              </button>
            ))}
          </div>
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Discovery Phase
          </div>
          <div className="flex flex-wrap gap-2">
            {DISCOVERY_PHASES.map((phase) => {
              const active = discoveryContext?.phase === phase;
              return (
                <button
                  key={phase}
                  onClick={() =>
                    void requestDiscoveryPhase(phase, "manual_control_console")
                  }
                  className={`rounded border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition-colors ${
                    active
                      ? "border-magenta-dim/40 text-magenta"
                      : "border-slate text-text-secondary hover:text-text-primary"
                  }`}
                >
                  {phase}
                </button>
              );
            })}
          </div>
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Hypothesis Lifecycle
            {isBackendAuthority ? (
              <span className="ml-2 text-[9px] font-normal normal-case tracking-normal text-text-muted">
                (backend authority)
              </span>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            {HYPOTHESIS_STATES.map((state) => {
              const active = hypothesisContext?.stateLabel === state;
              return (
                <button
                  key={state}
                  onClick={() => setLifecycle(state)}
                  disabled={isBackendAuthority}
                  className={`rounded border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition-colors ${
                    active
                      ? "border-teal-dim/40 text-teal"
                      : "border-slate text-text-secondary hover:text-text-primary"
                  }`}
                >
                  {state}
                </button>
              );
            })}
          </div>
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Panel Routing
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() =>
                panelBroker.emitPort(manifest, "open-panel", {
                  panelId: "rcsb_search",
                  open: true,
                })
              }
              className="rounded border border-slate px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-teal-dim/40 hover:text-text-primary"
            >
              Open RCSB
            </button>
            <button
              onClick={() =>
                panelBroker.emitPort(manifest, "open-panel", {
                  panelId: "data_inspector",
                  open: true,
                })
              }
              className="rounded border border-slate px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-teal-dim/40 hover:text-text-primary"
            >
              Open Inspector
            </button>
            <button
              onClick={() =>
                panelBroker.emitPort(manifest, "open-bottom-panel", {
                  panelId: "hypotheses",
                  open: true,
                })
              }
              className="rounded border border-slate px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-teal-dim/40 hover:text-text-primary"
            >
              Open Findings
            </button>
          </div>
        </section>

        <section className="mb-5">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Pipeline Audit
          </div>
          <PipelineAuditReadout />
        </section>

        <section>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
            Planner Policy
          </div>
          <div className="space-y-4">
            {toolGroups.map((group) => (
              <div key={group.label}>
                <div className="mb-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-text-secondary">
                  {group.label}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {group.tools.length > 0 ? (
                    group.tools.map((tool) => (
                      <span
                        key={`${group.label}-${tool}`}
                        className="rounded-full border border-slate px-2 py-0.5 font-mono text-[10px] text-text-primary"
                      >
                        {tool}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-text-muted">None</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
