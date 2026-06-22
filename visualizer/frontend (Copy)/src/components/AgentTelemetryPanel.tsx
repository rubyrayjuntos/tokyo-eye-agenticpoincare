import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { AgentTelemetrySnapshot } from "../lib/types";

function formatPct(value?: number) {
  return `${(value ?? 0).toFixed(1)}%`;
}

function formatTokens(value?: number) {
  return Math.round(value ?? 0);
}

export default function AgentTelemetryPanel() {
  const { latestAgentTelemetry, agentSessionId } = useDashboard();
  const [snapshot, setSnapshot] = useState<AgentTelemetrySnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!agentSessionId) return;
    let mounted = true;
    api
      .getAgentTelemetry(agentSessionId)
      .then((res) => {
        if (!mounted) return;
        setSnapshot(res);
        setError(null);
      })
      .catch((e: any) => {
        if (!mounted) return;
        setError(e?.message || "Telemetry unavailable");
      });
    return () => {
      mounted = false;
    };
  }, [agentSessionId, latestAgentTelemetry?.trace_id]);

  const localToolSummary = useMemo(() => {
    const calls = latestAgentTelemetry?.tool_calls ?? [];
    const errors = calls.filter((c) => c.is_error).length;
    return { total: calls.length, errors };
  }, [latestAgentTelemetry]);

  const totalAttribution = useMemo(() => {
    const source =
      latestAgentTelemetry?.token_attribution?.total ??
      snapshot?.latest_trace?.token_attribution?.total ??
      {};
    return Object.entries(source)
      .sort((a, b) => (b[1]?.estimated_tokens ?? 0) - (a[1]?.estimated_tokens ?? 0))
      .slice(0, 6);
  }, [latestAgentTelemetry, snapshot]);

  const latestCallBreakdown = useMemo(() => {
    const calls =
      latestAgentTelemetry?.llm_call_breakdown ??
      snapshot?.latest_trace?.llm_call_breakdown ??
      [];
    return calls;
  }, [latestAgentTelemetry, snapshot]);

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/60 flex flex-col overflow-y-auto">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Agent Telemetry
        </h2>
      </div>

      <div className="px-4 py-3 border-b border-zinc-800 space-y-1.5 text-xs">
        <div className="text-zinc-500 uppercase tracking-wide text-[10px]">Session</div>
        <div className="text-zinc-300 break-all">{agentSessionId || "No active chat session"}</div>
        {latestAgentTelemetry?.trace_id && (
          <>
            <div className="text-zinc-500 uppercase tracking-wide text-[10px] mt-2">Latest Trace</div>
            <div className="text-cyan-300 break-all">{latestAgentTelemetry.trace_id}</div>
          </>
        )}
      </div>

      <div className="px-4 py-3 border-b border-zinc-800 space-y-1.5 text-xs">
        <div className="text-zinc-500 uppercase tracking-wide text-[10px]">Current Request</div>
        <div className="text-zinc-300">Duration: {latestAgentTelemetry?.total_duration_ms ?? 0} ms</div>
        <div className="text-zinc-300">LLM calls: {latestAgentTelemetry?.llm_calls ?? 0}</div>
        <div className="text-zinc-300">Tokens: {latestAgentTelemetry?.tokens.total ?? 0}</div>
        <div className="text-zinc-300">
          Tool calls: {localToolSummary.total} ({localToolSummary.errors} errors)
        </div>
      </div>

      {totalAttribution.length > 0 && (
        <div className="px-4 py-3 border-b border-zinc-800 space-y-1.5 text-xs">
          <div className="text-zinc-500 uppercase tracking-wide text-[10px]">Token Breakdown</div>
          {totalAttribution.map(([key, value]) => (
            <div key={key} className="flex items-center justify-between gap-2 text-zinc-300">
              <span className="truncate">{key}</span>
              <span className="text-right text-zinc-400">
                {formatTokens(value?.estimated_tokens)} • {formatPct(value?.percent)}
              </span>
            </div>
          ))}
        </div>
      )}

      {latestCallBreakdown.length > 0 && (
        <div className="px-4 py-3 border-b border-zinc-800 space-y-2 text-xs">
          <div className="text-zinc-500 uppercase tracking-wide text-[10px]">LLM Call Breakdown</div>
          {latestCallBreakdown.map((call) => (
            <div key={call.call_index} className="rounded border border-zinc-800 p-2 bg-zinc-900">
              <div className="text-zinc-300">Call {call.call_index}</div>
              <div className="text-zinc-500">
                in {call.input_tokens} • out {call.output_tokens}
              </div>
              <div className="mt-1 space-y-1">
                {Object.entries(call.input || {})
                  .sort((a, b) => (b[1]?.estimated_tokens ?? 0) - (a[1]?.estimated_tokens ?? 0))
                  .slice(0, 4)
                  .map(([key, value]) => (
                    <div key={`${call.call_index}-${key}`} className="flex items-center justify-between gap-2 text-zinc-400">
                      <span className="truncate">{key}</span>
                      <span>{formatPct(value?.percent)}</span>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="px-4 py-3 border-b border-zinc-800 space-y-1.5 text-xs">
        <div className="text-zinc-500 uppercase tracking-wide text-[10px]">Delivery Health</div>
        <div className="text-zinc-300">WS connections: {snapshot?.viewport.active_connections ?? 0}</div>
        <div className="text-zinc-300">WS sessions: {snapshot?.viewport.active_sessions ?? 0}</div>
        <div className="text-zinc-300">Send failures: {snapshot?.viewport.delivery_failure_count ?? 0}</div>
        <div className="text-zinc-300">
          Failure rate: {((snapshot?.viewport.delivery_failure_rate ?? 0) * 100).toFixed(1)}%
        </div>
      </div>

      <div className="px-4 py-3 space-y-1.5 text-xs">
        <div className="text-zinc-500 uppercase tracking-wide text-[10px]">Recent Traces</div>
        {(snapshot?.recent_traces ?? []).slice().reverse().slice(0, 5).map((t) => (
          <div key={t.trace_id} className="rounded border border-zinc-800 p-2 bg-zinc-900">
            <div className="text-zinc-300 break-all">{t.trace_id}</div>
            <div className="text-zinc-500">{t.total_duration_ms} ms • {t.tokens.total} tokens</div>
          </div>
        ))}
        {error && <div className="text-red-400">{error}</div>}
      </div>
    </aside>
  );
}
