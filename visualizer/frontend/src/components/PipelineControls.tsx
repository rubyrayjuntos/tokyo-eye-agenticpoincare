/**
 * PipelineControls — left sidebar with ingest + pipeline execution.
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */
import { useState, useCallback, useEffect, useRef } from "react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { Structure, PipelineJob } from "../lib/types";

const PIPELINE_MODULES = [
  { id: "gnn_forward", label: "GNN Forward" },
  { id: "dtie_decomposition", label: "DTIE Decomposition" },
  { id: "vulnerability_scan", label: "Vulnerability Scan" },
  { id: "leak_flow", label: "Leak Flow" },
] as const;

const PIPELINE_STEPS = [
  "ingestion",
  "graph_build",
  "gnn_forward_pass",
  "dtie_decomposition",
  "complete",
];

export default function PipelineControls() {
  const { activeStructure, setActiveStructure, triggerRefresh } = useDashboard();

  // Ingest state
  const [pdbId, setPdbId] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  // Target selector state
  const [structures, setStructures] = useState<Structure[]>([]);
  const [selectedStructureId, setSelectedStructureId] = useState<string>("");

  // Pipeline modules state
  const [enabledModules, setEnabledModules] = useState<Record<string, boolean>>(
    Object.fromEntries(PIPELINE_MODULES.map((m) => [m.id, true]))
  );

  // Pipeline execution state
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [executing, setExecuting] = useState(false);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load structures for target selector
  const loadStructures = useCallback(() => {
    api.getStructures().then(setStructures).catch(() => {});
  }, []);

  useEffect(() => {
    loadStructures();
  }, [loadStructures]);

  // Sync selected structure with active structure
  useEffect(() => {
    if (activeStructure) {
      setSelectedStructureId(activeStructure.structure_id);
    }
  }, [activeStructure]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // --- Ingest handler ---
  const handleIngest = useCallback(async () => {
    const trimmed = pdbId.trim().toUpperCase();
    if (!trimmed) return;

    setIngesting(true);
    setIngestError(null);

    try {
      const result = await api.ingest(trimmed);
      // Build a Structure object from the ingest response
      const newStructure: Structure = {
        structure_id: result.structure_id,
        pdb_id: result.pdb_id,
        title: result.title,
        resolution: null,
        method: "",
        source: "rcsb",
        chains: result.chains,
        residue_count: result.residue_count,
        has_embeddings: false,
        last_run_id: null,
        ingested_at: new Date().toISOString(),
      };
      setStructures((prev) => [newStructure, ...prev]);
      setSelectedStructureId(result.structure_id);
      setActiveStructure(newStructure);
      setPdbId("");
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Ingestion failed";
      setIngestError(msg);
    } finally {
      setIngesting(false);
    }
  }, [pdbId, setActiveStructure]);

  // --- Pipeline execution handler ---
  const handleExecutePipeline = useCallback(async () => {
    if (!selectedStructureId) return;

    setExecuting(true);
    setPipelineError(null);
    setPipelineJob(null);

    const modules = Object.entries(enabledModules)
      .filter(([, enabled]) => enabled)
      .map(([id]) => id);

    try {
      const job = await api.runPipeline({
        structure_id: selectedStructureId,
        modules,
      });
      setPipelineJob(job);

      // Start polling for status
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getPipelineStatus(job.job_id);
          setPipelineJob(status);

          if (status.status === "complete" || status.status === "failed") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setExecuting(false);

            if (status.status === "failed") {
              setPipelineError(status.error || "Pipeline failed");
            } else {
              // Refresh structures list and all visualizations after completion
              loadStructures();
              triggerRefresh();
            }
          }
        } catch {
          // Polling error — keep trying
        }
      }, 2000);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Pipeline execution failed";
      setPipelineError(msg);
      setExecuting(false);
    }
  }, [selectedStructureId, enabledModules, loadStructures, triggerRefresh]);

  // --- Module toggle handler ---
  const toggleModule = (moduleId: string) => {
    setEnabledModules((prev) => ({ ...prev, [moduleId]: !prev[moduleId] }));
  };

  // --- Target selection handler ---
  const handleTargetChange = (structureId: string) => {
    setSelectedStructureId(structureId);
    const found = structures.find((s) => s.structure_id === structureId);
    if (found) setActiveStructure(found);
  };

  // --- Status display helpers ---
  const getStepIndex = (step: string) => {
    const idx = PIPELINE_STEPS.indexOf(step);
    return idx >= 0 ? idx : 0;
  };

  const isJobActive =
    pipelineJob &&
    (pipelineJob.status === "queued" || pipelineJob.status === "running");

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/60 flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="px-4 py-3 border-b border-zinc-800">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Pipeline Controls
        </h2>
      </div>

      {/* Ingest section */}
      <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          PDB ID
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="e.g. 4OBE"
            value={pdbId}
            onChange={(e) => setPdbId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleIngest();
            }}
            disabled={ingesting}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 disabled:opacity-50"
          />
          <button
            onClick={handleIngest}
            disabled={ingesting || !pdbId.trim()}
            className="px-3 py-1.5 bg-cyan-700/30 border border-cyan-600/40 rounded text-xs text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
          >
            {ingesting ? "..." : "Ingest"}
          </button>
        </div>
        {ingestError && (
          <p className="text-[10px] text-red-400">{ingestError}</p>
        )}
      </div>

      {/* Target selector */}
      <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          Target
        </label>
        <select
          value={selectedStructureId}
          onChange={(e) => handleTargetChange(e.target.value)}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-cyan-600"
        >
          <option value="">Select structure...</option>
          {structures.map((s) => (
            <option key={s.structure_id} value={s.structure_id}>
              {s.pdb_id} — {s.title || s.structure_id}
            </option>
          ))}
        </select>
      </div>

      {/* Pipeline modules */}
      <div className="px-4 py-3 border-b border-zinc-800 space-y-2">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          Modules
        </label>
        <div className="space-y-1.5">
          {PIPELINE_MODULES.map((mod) => (
            <label
              key={mod.id}
              className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={enabledModules[mod.id]}
                onChange={() => toggleModule(mod.id)}
                disabled={executing}
                className="rounded border-zinc-600 bg-zinc-800 accent-cyan-500"
              />
              {mod.label}
            </label>
          ))}
        </div>
      </div>

      {/* Execute button */}
      <div className="px-4 py-3 border-b border-zinc-800">
        <button
          onClick={handleExecutePipeline}
          disabled={executing || !selectedStructureId}
          className="w-full py-2 bg-cyan-700/30 border border-cyan-600/40 rounded text-xs font-medium text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-40"
        >
          {executing ? "Running..." : "Execute Pipeline"}
        </button>
      </div>

      {/* Status area */}
      <div className="px-4 py-3 flex-1">
        <label className="text-[10px] uppercase tracking-wider text-zinc-500">
          Status
        </label>

        {!pipelineJob && !pipelineError && (
          <p className="text-xs text-zinc-600 mt-1">Idle</p>
        )}

        {pipelineError && (
          <p className="text-xs text-red-400 mt-1">{pipelineError}</p>
        )}

        {pipelineJob && (
          <div className="mt-2 space-y-1.5">
            {/* Progress bar */}
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  pipelineJob.status === "failed"
                    ? "bg-red-500"
                    : pipelineJob.status === "complete"
                    ? "bg-emerald-500"
                    : "bg-cyan-500"
                }`}
                style={{ width: `${pipelineJob.progress}%` }}
              />
            </div>

            {/* Step indicators */}
            <div className="space-y-1">
              {PIPELINE_STEPS.map((step, idx) => {
                const currentIdx = getStepIndex(
                  pipelineJob.current_step || ""
                );
                const isDone = idx < currentIdx || pipelineJob.status === "complete";
                const isCurrent =
                  idx === currentIdx && pipelineJob.status === "running";

                return (
                  <div
                    key={step}
                    className={`flex items-center gap-1.5 text-[10px] ${
                      isDone
                        ? "text-emerald-400"
                        : isCurrent
                        ? "text-cyan-400"
                        : "text-zinc-600"
                    }`}
                  >
                    <span className="w-3 text-center">
                      {isDone ? "✓" : isCurrent ? "●" : "○"}
                    </span>
                    <span>{step.replace(/_/g, " ")}</span>
                  </div>
                );
              })}
            </div>

            {/* Status badge */}
            <div className="mt-2">
              <span
                className={`text-[10px] px-2 py-0.5 rounded ${
                  pipelineJob.status === "complete"
                    ? "bg-emerald-900/30 text-emerald-400"
                    : pipelineJob.status === "failed"
                    ? "bg-red-900/30 text-red-400"
                    : pipelineJob.status === "running"
                    ? "bg-cyan-900/30 text-cyan-400"
                    : "bg-zinc-800 text-zinc-400"
                }`}
              >
                {pipelineJob.status.toUpperCase()}
              </span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
