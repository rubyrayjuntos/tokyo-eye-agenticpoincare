/**
 * useAutoIngestPipeline — React hook wrapping the ingest → pipeline → poll → complete/error flow.
 * Requirements: 3.1, 4.1, 4.2, 4.3, 4.4
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { api } from "./api";
import type { Structure, PipelineJob } from "./types";
import {
  ingestAndTriggerPipeline,
  evaluatePipelineStatus,
} from "./autoIngestPipelineLogic";

export interface UseAutoIngestPipelineOptions {
  onComplete: (structure: Structure) => void;
  onError: (error: string) => void;
}

export interface UseAutoIngestPipelineReturn {
  ingestAndRun: (pdbId: string) => Promise<void>;
  isIngesting: boolean;
  isRunningPipeline: boolean;
  pipelineJob: PipelineJob | null;
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
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const ingestedStructureRef = useRef<Structure | null>(null);

  // Cleanup polling on unmount
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
    setError(null);
    ingestedStructureRef.current = null;
  }, []);

  const ingestAndRun = useCallback(async (pdbId: string) => {
    reset();
    setIsIngesting(true);

    try {
      const { structure, pipelineJob: job } = await ingestAndTriggerPipeline(
        pdbId,
        api
      );
      ingestedStructureRef.current = structure;

      setIsIngesting(false);
      setIsRunningPipeline(true);
      setPipelineJob(job);

      // Poll for pipeline status
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.getPipelineStatus(job.job_id);
          setPipelineJob(status);

          const action = evaluatePipelineStatus(status, ingestedStructureRef.current!);

          if (action.type === "complete") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setIsRunningPipeline(false);
            optionsRef.current.onComplete(action.structure);
          } else if (action.type === "failed") {
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setIsRunningPipeline(false);
            setError(action.error);
            optionsRef.current.onError(action.error);
          }
          // "continue" — keep polling
        } catch {
          // Polling network error — keep trying (resilient)
        }
      }, POLL_INTERVAL_MS);
    } catch (err: unknown) {
      setIsIngesting(false);
      setIsRunningPipeline(false);
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Ingestion failed";
      setError(msg);
      optionsRef.current.onError(msg);
    }
  }, [reset]);

  return {
    ingestAndRun,
    isIngesting,
    isRunningPipeline,
    pipelineJob,
    error,
    reset,
  };
}
