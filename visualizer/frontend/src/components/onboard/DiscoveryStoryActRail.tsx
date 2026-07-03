import type { ActReadiness } from "../lib/types";

const ACT_ORDER = [
  "signal",
  "persistent_leak",
  "cryptic_pocket",
  "fragment",
  "verdict",
] as const;

const STATUS_STYLES: Record<string, string> = {
  complete: "border-success/40 bg-success/10 text-success",
  degraded: "border-warning/40 bg-warning/10 text-warning",
  running: "border-teal/40 bg-teal-dim/10 text-teal",
  failed: "border-error/40 bg-error/10 text-error",
  pending: "border-slate bg-bg-elevated/50 text-text-muted",
  not_implemented: "border-slate bg-bg-elevated/30 text-text-muted italic",
};

export interface DiscoveryStoryActRailProps {
  acts?: Record<string, ActReadiness>;
  currentAct?: number;
}

export function DiscoveryStoryActRail({ acts, currentAct = 0 }: DiscoveryStoryActRailProps) {
  if (!acts) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-text-secondary">
        Discovery Story
      </div>
      <div className="grid grid-cols-1 gap-1">
        {ACT_ORDER.map((actId) => {
          const act = acts[actId];
          if (!act) return null;
          const isCurrent = act.number === currentAct;
          const statusClass = STATUS_STYLES[act.status] ?? STATUS_STYLES.pending;
          return (
            <div
              key={actId}
              className={`rounded-md border px-2.5 py-1.5 text-[10px] ${statusClass} ${
                isCurrent ? "ring-1 ring-teal/50" : ""
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">
                  Act {act.number} · {act.title}
                </span>
                <span className="uppercase tracking-wider opacity-80">{act.status}</span>
              </div>
              <div className="mt-0.5 text-[9px] opacity-80">{act.question}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
