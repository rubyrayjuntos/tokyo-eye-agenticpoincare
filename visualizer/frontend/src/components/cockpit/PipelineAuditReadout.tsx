import { useEffect, useState } from "react";

import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import type { AuditEventRecord, StructureReadiness } from "../../lib/types";

const SEVERITY_STYLES: Record<string, string> = {
  error: "text-red-400 border-red-500/30 bg-red-500/10",
  warning: "text-amber-300 border-amber-500/30 bg-amber-500/10",
  info: "text-teal border-teal-dim/30 bg-teal-dim/10",
};

function formatTime(iso: string | undefined) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function PipelineAuditReadout() {
  const { activeStructure } = useDashboard();
  const structureId = activeStructure?.structure_id;
  const [readiness, setReadiness] = useState<StructureReadiness | null>(null);
  const [events, setEvents] = useState<AuditEventRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!structureId) {
      setReadiness(null);
      setEvents([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      api.getStructureReadiness(structureId),
      api.getStructureAudit(structureId, { limit: 12, severity: undefined }),
    ])
      .then(([readinessResult, auditResult]) => {
        if (cancelled) return;
        setReadiness(readinessResult);
        setEvents(auditResult.events ?? []);
      })
      .catch((err: { message?: string }) => {
        if (cancelled) return;
        setError(err.message ?? "Failed to load audit data");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [structureId]);

  if (!structureId) {
    return (
      <p className="text-xs text-text-muted">
        Select a structure to view geometric readiness and pipeline audit history.
      </p>
    );
  }

  const geo = readiness?.geometric_readiness;

  return (
    <div className="space-y-4">
      {loading ? (
        <p className="text-xs text-text-muted">Loading audit readout…</p>
      ) : null}
      {error ? <p className="text-xs text-red-400">{error}</p> : null}

      <div className="rounded border border-slate/80 bg-[#0f1219] p-3">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-text-secondary">
          Geometric Readiness
        </div>
        {geo ? (
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div>
              <span className="text-text-muted">Hyperbolic ready</span>
              <div className={geo.hyperbolic_ready ? "text-teal" : "text-amber-300"}>
                {geo.hyperbolic_ready ? "yes" : "no"}
              </div>
            </div>
            <div>
              <span className="text-text-muted">Learned κ</span>
              <div className="font-mono text-text-primary">
                {geo.learned_curvature != null ? geo.learned_curvature.toFixed(4) : "pending"}
              </div>
            </div>
            <div>
              <span className="text-text-muted">Curvature ready</span>
              <div className={geo.curvature_ready ? "text-teal" : "text-text-secondary"}>
                {geo.curvature_ready == null ? "n/a" : geo.curvature_ready ? "yes" : "no"}
              </div>
            </div>
            <div>
              <span className="text-text-muted">Status</span>
              <div className="text-text-primary">{readiness?.readiness_status ?? "—"}</div>
            </div>
          </div>
        ) : (
          <p className="text-xs text-text-muted">No geometric readiness data yet.</p>
        )}
      </div>

      <div>
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-text-secondary">
          Recent Audit Events
        </div>
        {events.length === 0 ? (
          <p className="text-xs text-text-muted">No persisted audit events for this structure.</p>
        ) : (
          <ul className="space-y-2">
            {events.map((event) => {
              const style = SEVERITY_STYLES[event.severity] ?? SEVERITY_STYLES.info;
              const detail =
                typeof event.details?.message === "string"
                  ? event.details.message
                  : typeof event.details?.note === "string"
                    ? event.details.note
                    : typeof event.details?.error === "string"
                      ? event.details.error
                      : event.job_name
                        ? `${event.job_name}`
                        : event.event_type;
              return (
                <li
                  key={event.event_id}
                  className={`rounded border px-2.5 py-2 text-[11px] ${style}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono uppercase tracking-wide">{event.event_type}</span>
                    <span className="text-[10px] opacity-80">{formatTime(event.timestamp)}</span>
                  </div>
                  <p className="mt-1 leading-5 text-text-primary">{detail}</p>
                  {event.enforcement_level ? (
                    <p className="mt-1 text-[10px] text-text-muted">
                      enforcement: {event.enforcement_level}
                      {event.contract_version ? ` · contract ${event.contract_version}` : ""}
                    </p>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
