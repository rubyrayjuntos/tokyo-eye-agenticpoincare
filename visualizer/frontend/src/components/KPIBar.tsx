import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { KPIs } from "../lib/types";

function KPICard({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center px-4 py-2 rounded-lg bg-zinc-800/50 border border-zinc-700/50 min-w-[120px]">
      <span className="text-lg font-semibold text-zinc-100 tabular-nums">
        {value}
      </span>
      <span className="text-[10px] text-zinc-500 uppercase tracking-wider mt-0.5">
        {label}
      </span>
    </div>
  );
}

export default function KPIBar() {
  const { refreshKey } = useDashboard();
  const [kpis, setKpis] = useState<KPIs | null>(null);

  useEffect(() => {
    const poll = () => {
      api.getKPIs().then(setKpis).catch(() => {});
    };
    poll();
    const id = setInterval(poll, 60_000); // Poll every 60s — KPIs don't change often
    return () => clearInterval(id);
  }, [refreshKey]);

  if (!kpis) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 border-b border-zinc-800 bg-zinc-900/40">
        <span className="text-xs text-zinc-600">Loading KPIs...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-zinc-800 bg-zinc-900/40 overflow-x-auto shrink-0">
      <KPICard label="Structures" value={String(kpis.total_structures)} />
      <KPICard
        label="Mean Uncertainty"
        value={kpis.mean_uncertainty.toFixed(3)}
      />
      <KPICard
        label="Avg Inference"
        value={`${kpis.avg_inference_seconds.toFixed(1)}s`}
      />
      <KPICard label="Active Jobs" value={String(kpis.active_jobs)} />
      <KPICard
        label="Model"
        value={kpis.model_status.charAt(0).toUpperCase() + kpis.model_status.slice(1)}
      />
    </div>
  );
}
