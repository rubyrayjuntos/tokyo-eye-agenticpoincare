/**
 * useAutoIngestPipeline — ingest → readiness poll → complete/error flow.
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { api } from "./api";
import type { Structure, PipelineJob, StructureReadiness } from "./types";
import { ingestAndTriggerPipeline } from "./autoIngestPipelineLogic";
import {
  actProgressLabel,
  errorMessageFromUnknown,
  MAX_READINESS_POLL_FAILURES,
  READINESS_POLL_START_LABEL,
  runReadinessPollCycle,
} from "./readinessPollingLogic";

export interface UseAutoIngestPipelineOptions {
  onComplete: (structure: Structure, readiness?: StructureReadiness) => void;
  onError: (error: string) => void;
}

export interface UseAutoIngestPipelineReturn {
  ingestAndRun: (pdbId: string) => Promise<void>;
  isIngesting: boolean;
  isRunningPipeline: boolean;
  pipelineJob: PipelineJob | null;
  readiness: StructureReadiness | null;
  progressLabel: string | null;
  error: string | null;
  reset: () => void;
}

const POLL_INTERVAL_MS = 2000;

export function useAutoIngestPipeline(
  options: UseAutoIngestPipelineOptions
): UseAutoIngestPipelineReturn {
  const [isIngesting, setIsIngesting] = useState(false);
  const [isRunningPipeline, setIsRunningPipeline] = useState(false);
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [readiness, setReadiness] = useState<StructureReadiness | null>(null);
  const [progressLabel, setProgressLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const ingestedStructureRef = useRef<Structure | null>(null);
  const pipelineJobIdRef = useRef<string | null>(null);
  const pollFailureCountRef = useRef(0);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const reset = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setIsIngesting(false);
    setIsRunningPipeline(false);
    setPipelineJob(null);
    setReadiness(null);
    setProgressLabel(null);
    setError(null);
    ingestedStructureRef.current = null;
    pipelineJobIdRef.current = null;
    pollFailureCountRef.current = 0;
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const failPipeline = useCallback(
    (message: string) => {
      stopPolling();
      setIsRunningPipeline(false);
      setError(message);
      optionsRef.current.onError(message);
    },
    [stopPolling]
  );

  const ingestAndRun = useCallback(
    async (pdbId: string) => {
      reset();
      setIsIngesting(true);

      try {
        const { structure, pipelineJob: job } = await ingestAndTriggerPipeline(pdbId, api);
        ingestedStructureRef.current = structure;
        pipelineJobIdRef.current = job?.job_id ?? null;

        setIsIngesting(false);
        setIsRunningPipeline(true);
        setPipelineJob(job);
        setProgressLabel(READINESS_POLL_START_LABEL);

        const pollReadiness = async () => {
          const structureSnapshot = ingestedStructureRef.current;
          if (!structureSnapshot) return;

          try {
            const result = await runReadinessPollCycle({
              structure: structureSnapshot,
              getReadiness: () => api.getStructureReadiness(structureSnapshot.structure_id),
              getPipelineStatus: (jobId) => api.getPipelineStatus(jobId),
              pipelineJobId: pipelineJobIdRef.current,
            });

            pollFailureCountRef.current = 0;

            if (result.type === "continue") {
              setReadiness(result.readiness);
              setProgressLabel(
                actProgressLabel(result.readiness) ?? "Discovery compute running…"
              );
              return;
            }

            if (result.type === "complete") {
              stopPolling();
              setIsRunningPipeline(false);
              setReadiness(result.readiness);
              setProgressLabel(actProgressLabel(result.readiness));
              setPipelineJob((prev) =>
                prev
                  ? {
                      ...prev,
                      status: "complete",
                      progress: 100,
                      completed_at: new Date().toISOString(),
                    }
                  : prev
              );
              optionsRef.current.onComplete(result.structure, result.readiness);
              return;
            }

            setReadiness(result.readiness);
            failPipeline(result.error);
          } catch (err: unknown) {
            pollFailureCountRef.current += 1;
            if (pollFailureCountRef.current >= MAX_READINESS_POLL_FAILURES) {
              failPipeline(errorMessageFromUnknown(err));
            }
          }
        };

        pollRef.current = setInterval(() => {
          void pollReadiness();
        }, POLL_INTERVAL_MS);

        await pollReadiness();
      } catch (err: unknown) {
        setIsIngesting(false);
        setIsRunningPipeline(false);
        const msg = errorMessageFromUnknown(err) || "Ingestion failed";
        setError(msg);
        optionsRef.current.onError(msg);
      }
    },
    [failPipeline, reset, stopPolling]
  );

  return {
    ingestAndRun,
    isIngesting,
    isRunningPipeline,
    pipelineJob,
    readiness,
    progressLabel,
    error,
    reset,
  };
}
