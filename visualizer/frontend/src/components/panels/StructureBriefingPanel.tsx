import { useMemo } from "react";

import { useWorkbenchHydration } from "../../context/useWorkbenchHydration";
import { useDashboard } from "../../lib/context";
import { buildBriefingView } from "../../lib/briefingView";
import { formatArtifactReason } from "../../lib/artifactAvailability";
import type { ToolPanelId } from "../../lib/types";
import { usePanelBroker } from "../workbench/panelBroker";
import type { PanelManifest } from "../workbench/panelContract";

interface StructureBriefingPanelProps {
  manifest: PanelManifest<ToolPanelId>;
}

export default function StructureBriefingPanel({ manifest }: StructureBriefingPanelProps) {
  const {
    hydration,
    isHydrating,
    hydrateSignalsInferred,
    hydrationBusStatus,
  } = useWorkbenchHydration();
  const { activeStructure: structure } = useDashboard();
  const panelBroker = usePanelBroker();

  const briefing = useMemo(
    () => buildBriefingView(hydration, structure?.structure_id, isHydrating),
    [hydration, structure?.structure_id, isHydrating],
  );

  const handleResidueSelect = (residueId: string) => {
    panelBroker.emitPort(manifest, "selection", { residueIds: [residueId] });
  };

  if (!briefing) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-xs text-text-secondary">
        Load a structure to view the hydrate briefing.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b0d14] text-text-primary">
      <header className="border-b border-slate px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.22em] text-text-secondary">
          Structure Briefing
        </div>
        <div className="mt-1 font-mono text-sm text-teal">
          {briefing.structureId}
          {briefing.pdbId ? ` · ${briefing.pdbId.toUpperCase()}` : ""}
        </div>
        {briefing.title ? (
          <p className="mt-1 text-xs text-text-secondary">{briefing.title}</p>
        ) : null}
        <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
          <span className="rounded border border-teal-dim/40 px-2 py-0.5 text-teal">
            HydrateBundle · contract {briefing.contractVersion}
          </span>
          {briefing.degraded ? (
            <span className="rounded border border-warning/40 px-2 py-0.5 text-warning">
              Degraded · missing {briefing.missingTier1.join(", ")}
            </span>
          ) : (
            <span className="rounded border border-success/40 px-2 py-0.5 text-success">
              Tier-1 complete
            </span>
          )}
          {hydrateSignalsInferred || briefing.signalsInferred ? (
            <span className="rounded border border-slate px-2 py-0.5 text-text-secondary">
              Availability inferred client-side
            </span>
          ) : null}
        </div>
        <p className="mt-2 text-[10px] leading-5 text-text-muted">
          Read-only view of governed hydrate data. No fold-specific labels or narrative overlays.
        </p>
      </header>

      <div className="ck-scroll flex-1 overflow-y-auto px-4 py-4">
        {briefing.isHydrating ? (
          <div className="mb-4 rounded-md border border-slate bg-bg-elevated/40 px-3 py-2 text-xs text-text-secondary">
            Hydrating…{hydrationBusStatus ? ` (${hydrationBusStatus})` : ""}
          </div>
        ) : null}

        <section className="mb-5">
          <SectionTitle>Tier-1 artifacts</SectionTitle>
          <div className="space-y-1">
            {briefing.artifacts.map((artifact) => (
              <div
                key={artifact.key}
                className="flex items-center justify-between rounded-md border border-slate bg-bg-elevated/50 px-3 py-2 text-xs"
              >
                <span className="text-text-primary">{artifact.label}</span>
                <span
                  className={
                    artifact.present
                      ? "font-mono text-success"
                      : "font-mono text-warning"
                  }
                  title={artifact.reason ? formatArtifactReason(artifact.reason) : undefined}
                >
                  {artifact.present ? "present" : artifact.reason ?? "absent"}
                </span>
              </div>
            ))}
          </div>
        </section>

        {briefing.sections.map((section) => (
          <section key={section.artifactKey} className="mb-5">
            <SectionTitle>{section.title}</SectionTitle>
            {!section.present ? (
              <p className="text-xs text-text-secondary">
                Not in bundle
                {section.reason ? ` (${formatArtifactReason(section.reason)})` : "."}
              </p>
            ) : (
              <>
                {section.stats.length > 0 ? (
                  <div className="mb-3 grid grid-cols-2 gap-2">
                    {section.stats.map((stat) => (
                      <StatTile key={stat.label} label={stat.label} value={stat.value} />
                    ))}
                  </div>
                ) : null}
                {section.rows.length > 0 ? (
                  <div className="space-y-1">
                    {section.rows.map((row) => (
                      <button
                        key={row.residue_id}
                        type="button"
                        onClick={() => handleResidueSelect(row.residue_id)}
                        className="flex w-full items-center justify-between rounded-md border border-slate bg-bg-elevated/40 px-3 py-2 text-left text-xs transition-colors hover:border-teal-dim/40"
                      >
                        <span className="font-mono text-teal">{row.label}</span>
                        <span className="font-mono text-text-secondary">
                          {row.valueLabel} {row.value.toFixed(3)}
                        </span>
                      </button>
                    ))}
                  </div>
                ) : null}
                {section.footnote ? (
                  <p className="mt-2 text-[10px] leading-5 text-text-muted">{section.footnote}</p>
                ) : null}
              </>
            )}
          </section>
        ))}

        <section className="mb-5">
          <SectionTitle>Persistence</SectionTitle>
          <div className="grid grid-cols-2 gap-2">
            {briefing.persistence.map((row) => (
              <StatTile
                key={row.label}
                label={row.label}
                value={row.persisted ? "yes" : "no"}
                tone={row.persisted ? "text-success" : "text-text-secondary"}
              />
            ))}
          </div>
        </section>

        <section className="rounded-md border border-slate bg-bg-elevated/40 p-3 text-xs text-text-secondary">
          <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-text-muted">
            Provenance
          </div>
          <div className="mt-2">{briefing.hypothesisCount} hypotheses in workspace.</div>
          {Object.keys(briefing.runIds).length > 0 ? (
            <div className="mt-2 font-mono text-[10px] text-text-muted">
              {Object.entries(briefing.runIds)
                .map(([k, v]) => `${k}: ${v.slice(0, 8)}…`)
                .join(" · ")}
            </div>
          ) : (
            <div className="mt-2 text-[10px] text-text-muted">No run IDs in context_summary.</div>
          )}
        </section>
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
      {children}
    </div>
  );
}

function StatTile({
  label,
  value,
  tone = "text-text-primary",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="rounded-md border border-slate bg-bg-elevated/70 p-3">
      <div className="text-[9px] uppercase tracking-[0.1em] text-text-secondary">{label}</div>
      <div className={`mt-1 font-mono text-sm ${tone}`}>{value}</div>
    </div>
  );
}
