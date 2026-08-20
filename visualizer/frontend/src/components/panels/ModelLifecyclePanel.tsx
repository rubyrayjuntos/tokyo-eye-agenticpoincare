import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  FlaskConical,
  Loader2,
  RefreshCw,
  Rocket,
  ShieldCheck,
} from "lucide-react";

import { useDashboard } from "../../lib/context";
import { api } from "../../lib/api";
import type { LifecycleStatus } from "../../lib/types";
import type { PanelManifest } from "../workbench/panelContract";
import { usePanelBroker } from "../workbench/panelBroker";

interface ModelLifecyclePanelProps {
  manifest: PanelManifest<"model_lifecycle">;
  structureId?: string | null;
}

function formatErr(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  if (err && typeof err === "object" && "message" in err) {
    const message = (err as { message: unknown }).message;
    if (typeof message === "string" && message.trim()) return message;
  }
  try {
    return JSON.stringify(err);
  } catch {
    return "Unknown error";
  }
}

export default function ModelLifecyclePanel({
  manifest,
  structureId: structureIdProp,
}: ModelLifecyclePanelProps) {
  const { emitPort } = usePanelBroker();
  const { activeStructure } = useDashboard();
  const structureId = structureIdProp ?? activeStructure?.structure_id ?? null;
  const [status, setStatus] = useState<LifecycleStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [runId, setRunId] = useState("master_cold_v1");
  const [lineageId, setLineageId] = useState<"v6" | "v6.5">("v6.5");
  const [checkpointPath, setCheckpointPath] = useState("");
  const [corpusManifest, setCorpusManifest] = useState(
    "manifests/v6_corpus_stage_a_small_v1.json",
  );
  const [message, setMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const next = await api.getLifecycleStatus();
      setStatus(next);
    } catch (err) {
      setError(formatErr(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(id);
  }, [refresh]);

  const withBusy = async (fn: () => Promise<void>) => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      await fn();
      await refresh();
    } catch (err) {
      setError(formatErr(err));
    } finally {
      setBusy(false);
    }
  };

  const production = status?.production;
  const mlflow = status?.mlflow;
  const activeJob = status?.active_job;
  const gates = status?.gates;

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto p-3 text-sm text-zinc-200">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 font-medium text-zinc-100">
          <FlaskConical className="h-4 w-4 text-cyan-400" />
          Model Lifecycle
        </div>
        <button
          type="button"
          className="rounded border border-zinc-700 p-1 hover:bg-zinc-800"
          onClick={() => void refresh()}
          title="Refresh status"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-800 bg-red-950/40 px-2 py-1 text-xs text-red-300">
          {error}
        </div>
      )}
      {message && (
        <div className="rounded border border-emerald-800 bg-emerald-950/40 px-2 py-1 text-xs text-emerald-300">
          {message}
        </div>
      )}

      <section className="space-y-1 rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
        <div className="text-xs uppercase tracking-wide text-zinc-500">360° status</div>
        <Row label="Champion" value={production?.model_version ?? "—"} />
        <Row
          label="Checkpoint"
          value={
            production?.checkpoint_id
              ? `${production.checkpoint_id}`
              : production?.checkpoint_path ?? "—"
          }
        />
        <Row
          label="SHA"
          value={
            production?.checkpoint_sha256
              ? `${production.checkpoint_sha256.slice(0, 12)}…`
              : "—"
          }
        />
        <Row
          label="MLflow"
          value={
            mlflow?.available
              ? `up (${mlflow.mlflow_version ?? "?"})`
              : `down${mlflow?.error ? `: ${mlflow.error}` : ""}`
          }
        />
        <Row
          label="P_FEATURE_01"
          value={gates?.p_feature_01_passed ? "passed" : "missing"}
        />
        <Row
          label="Train job"
          value={
            activeJob
              ? `${activeJob.status} · ${activeJob.job_id}`
              : "idle"
          }
        />
      </section>

      {status?.lineages?.map((lin) => (
        <section
          key={lin.lineage_id}
          className="space-y-1 rounded-lg border border-zinc-800 bg-zinc-950/40 p-2"
        >
          <div className="flex items-center gap-2 text-xs font-medium text-cyan-300">
            <Activity className="h-3 w-3" />
            {lin.lineage_id} · {lin.model_version}
          </div>
          <Row
            label="Champion"
            value={
              lin.champion
                ? `v${lin.champion.version}${lin.champion.checkpoint_path ? ` · ${lin.champion.checkpoint_path}` : ""}`
                : "unset"
            }
          />
          <Row
            label="Challenger"
            value={
              lin.challenger
                ? `v${lin.challenger.version}`
                : "unset"
            }
          />
          <div className="text-[11px] text-zinc-500">
            experiment: {lin.mlflow_experiment}
          </div>
        </section>
      ))}

      <section className="space-y-2 rounded-lg border border-zinc-800 p-2">
        <div className="text-xs uppercase tracking-wide text-zinc-500">Actions</div>
        <label className="block text-xs text-zinc-400">
          Lineage
          <select
            className="mt-1 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1"
            value={lineageId}
            onChange={(e) => setLineageId(e.target.value as "v6" | "v6.5")}
          >
            <option value="v6.5">v6.5 (active fork)</option>
            <option value="v6">v6 (baseline)</option>
          </select>
        </label>
        <label className="block text-xs text-zinc-400">
          Run ID
          <input
            className="mt-1 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
          />
        </label>
        <button
          type="button"
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded bg-cyan-700 px-2 py-1.5 text-xs font-medium hover:bg-cyan-600 disabled:opacity-50"
          onClick={() =>
            void withBusy(async () => {
              const job = await api.enqueueLifecycleTrain({
                lineage_id: lineageId,
                run_id: runId,
                preset: "master_cold",
                no_warm_start: true,
              });
              setMessage(
                job.suggested_command
                  ? `Queued ${job.job_id}. Host: ${job.suggested_command}`
                  : `Queued ${job.job_id}`,
              );
            })
          }
        >
          {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Rocket className="h-3 w-3" />}
          Queue cold start
        </button>

        <label className="block text-xs text-zinc-400">
          Checkpoint path
          <input
            className="mt-1 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px]"
            placeholder="checkpoints/v65/runs/.../v65_best.pt"
            value={checkpointPath}
            onChange={(e) => setCheckpointPath(e.target.value)}
          />
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            disabled={busy || !checkpointPath}
            className="rounded border border-zinc-600 px-2 py-1.5 text-xs hover:bg-zinc-800 disabled:opacity-50"
            onClick={() =>
              void withBusy(async () => {
                await api.registerLifecycleModel({
                  lineage_id: lineageId,
                  checkpoint_path: checkpointPath,
                  alias: "challenger",
                  sync_contract: true,
                });
                setMessage("Registered as challenger");
              })
            }
          >
            Set challenger
          </button>
          <button
            type="button"
            disabled={busy || !checkpointPath}
            className="flex items-center justify-center gap-1 rounded border border-amber-700 bg-amber-950/40 px-2 py-1.5 text-xs hover:bg-amber-900/40 disabled:opacity-50"
            onClick={() =>
              void withBusy(async () => {
                await api.promoteLifecycleModel({
                  lineage_id: lineageId,
                  checkpoint_path: checkpointPath,
                  alias: "champion",
                  sync_contract: true,
                });
                setMessage("Promoted champion + contract sync");
              })
            }
          >
            <ShieldCheck className="h-3 w-3" />
            Promote champion
          </button>
        </div>

        <label className="block text-xs text-zinc-400">
          Corpus manifest
          <input
            className="mt-1 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-[11px]"
            value={corpusManifest}
            onChange={(e) => setCorpusManifest(e.target.value)}
          />
        </label>
        <button
          type="button"
          disabled={busy || !structureId}
          className="w-full rounded border border-zinc-600 px-2 py-1.5 text-xs hover:bg-zinc-800 disabled:opacity-50"
          onClick={() =>
            void withBusy(async () => {
              if (!structureId) return;
              const preview = await api.getCheckpointPreview({
                structure_id: structureId,
                alias: "challenger",
                lineage_id: lineageId,
              });
              setMessage(
                `Preview ready for ${structureId}. Open TripleViewport — ${JSON.stringify(preview.hydrate_hint ?? "")}`,
              );
            })
          }
        >
          Preview challenger on current structure
        </button>

        <button
          type="button"
          disabled={busy || !structureId}
          className="w-full rounded border border-zinc-600 px-2 py-1.5 text-xs hover:bg-zinc-800 disabled:opacity-50"
          onClick={() =>
            void withBusy(async () => {
              if (!structureId) return;
              const res = await api.addStructureToCorpus({
                structure_id: structureId,
                manifest_path: corpusManifest,
              });
              setMessage(`${res.action} ${res.pdb_id} → ${res.manifest}`);
            })
          }
        >
          Add current structure to corpus
          {!structureId ? " (select a structure)" : ""}
        </button>

        <button
          type="button"
          className="w-full rounded border border-violet-700 px-2 py-1.5 text-xs text-violet-200 hover:bg-violet-950/40"
          onClick={() => {
            emitPort(manifest, "open-panel", { panelId: "model_lifecycle" });
            window.open("/mlflow/", "_blank", "noopener,noreferrer");
          }}
        >
          Open MLflow UI
        </button>
      </section>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2 text-xs">
      <span className="shrink-0 text-zinc-500">{label}</span>
      <span className="truncate text-right text-zinc-200" title={value}>
        {value}
      </span>
    </div>
  );
}
