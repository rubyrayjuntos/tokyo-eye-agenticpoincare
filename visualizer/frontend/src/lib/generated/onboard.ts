// AUTO-GENERATED from science/contracts/onboard_contract.yaml — do not edit by hand.
// Regenerate: python -m science.contracts.generate_typescript



export interface PipelineJob {
  job_id: string;
  structure_id: string;
  status: "queued" | "running" | "complete" | "failed" | "timed_out" | "skipped";
  current_step?: string | null;
  progress?: number | null;
  modules?: unknown[] | null;
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}


export interface ActReadiness {
  act_id: string;
  number: number;
  title: string;
  question: string;
  color: string;
  status: "running" | "complete" | "degraded" | "failed" | "pending" | "not_implemented" | string;
  jobs_complete: number;
  jobs_total: number;
  required_artifacts: Record<string, boolean>;
  optional_artifacts: Record<string, boolean>;
}


export interface StructureReadiness {
  structure_id: string;
  readiness_status: "running" | "ready" | "degraded" | "failed";
  computation_run_id?: string | null;
  pathway: string;
  current_act: number;
  foundation: Record<string, boolean>;
  acts: Record<string, ActReadiness>;
  artifacts: Record<string, boolean>;
  tier1: Record<string, boolean>;
  tier2: Record<string, boolean>;
  missing_artifacts: string[];
  degraded_reasons: string[];
  probe_errors?: Record<string, string>;
  pipeline_job?: PipelineJob | null;
  geometric_readiness?: {
    requires_hyperbolic: boolean;
    required_hyperbolic_artifacts: Record<string, boolean>;
    hyperbolic_ready: boolean;
    learned_curvature?: number | null;
    curvature_ready?: boolean | null;
  };
}


export interface IngestResponse {
  structure_id: string;
  pdb_id: string;
  residues: number;
  residue_count: number;
  chains: string[] | number;
  chain_count?: number;
  atoms: number;
  audit_only?: boolean;
  audit_run_id?: string | null;
  pipeline_status?: "queued" | "running" | "complete" | "failed" | "skipped" | null;
  pipeline_job_id?: string | null;
  pipeline_status_url?: string | null;
  readiness_url?: string | null;
  source?: string;
  already_existed?: boolean;
}

export interface ArtifactAvailabilityEntry {
  present: boolean;
  tier: number | null;
  reason: string | null;
}

export interface HydrateMeta {
  contract_version: string;
  degraded: boolean;
  missing_tier1_count: number;
  missing_keys: string[];
}


export interface HydrationResponse {
  structure_id: string;
  structure_snapshot?: unknown | null;
  embeddings?: unknown | null;
  graph_metrics?: unknown | null;
  allosteric_sites?: unknown | null;
  binding_scan?: unknown | null;
  source_leaks?: unknown | null;
  hypotheses?: unknown | null;
  provenance_runs?: unknown | null;
  annotations?: unknown | null;
  phase2_vulnerability?: unknown | null;
  phase4_resistance?: unknown | null;
  phase5_pharmacophore?: unknown | null;
  phase6_drug_candidates?: unknown | null;
  resistance_data?: unknown | null;
  persistence_status?: unknown | null;
  context_summary?: unknown | null;
  buffering_atlas?: unknown | null;
  artifact_availability?: Record<string, ArtifactAvailabilityEntry>;
  hydrate_meta?: HydrateMeta;
}
