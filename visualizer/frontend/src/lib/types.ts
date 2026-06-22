// Core domain types for the Tokyo Eye visualizer

export type PoincareColorMode =
  | "cone_depth"
  | "epistemic"
  | "aleatoric"
  | "centrality"
  | "domain";

export type StructureColorModeType = "spectrum" | "chain" | "bfactor" | "hydrophobicity";

export type ActivePanelName =
  | "structure"
  | "topology"
  | "hypotheses"
  | "agent"
  | null;

// ---------- Viewport Directives ----------

export interface ViewportDirective {
  action: "highlight" | "set_metric" | "focus" | "clear" | "annotate" | "set_color_mode";
  residue_ids?: string[];
  metric?: string;
  color_mode?: PoincareColorMode;
  message?: string;
  annotation?: string;
  focus_residue?: string;
}

// ---------- Structures ----------

export interface Structure {
  structure_id: string;
  pdb_id: string;
  title?: string | null;
  resolution?: number | null;
  method?: string | null;
  source?: string | null;
  chains?: string[] | null;
  residue_count?: number | null;
  has_embeddings?: boolean;
  last_run_id?: string | null;
  ingested_at?: string | null;
}

// ---------- Pipeline ----------

export interface PipelineJob {
  job_id: string;
  structure_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress?: number | null;
  error?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
}

export interface PipelineRunRequest {
  structure_id: string;
  modules?: string[];
}

// ---------- Ingestion ----------

export interface IngestRequest {
  pdb_id: string;
}

export interface IngestResponse {
  structure_id: string;
  pdb_id: string;
  title?: string | null;
  chains?: string[];
  residue_count?: number | null;
  already_exists?: boolean;
}

// ---------- Embeddings ----------

export interface EmbeddingResidueData {
  residue_id: string;
  residue_name?: string | null;
  chain_label?: string | null;
  x?: number | null;
  y?: number | null;
  cone_depth?: number | null;
  epistemic_uncertainty?: number | null;
  aleatoric_uncertainty?: number | null;
}

export interface EmbeddingData {
  structure_id: string;
  curvature?: number | null;
  model_version?: string | null;
  residues: EmbeddingResidueData[];
}

// ---------- Graph ----------

export interface GraphMetrics {
  structure_id: string;
  edge_count?: number | null;
  mean_degree?: number | null;
  bridges?: string[];
}

export interface GraphMetricsData {
  structure_id: string;
  node_count: number;
  edge_count: number;
  mean_degree: number;
  bridge_count: number;
  density: number;
}

export interface GraphCompareResult {
  structure_a: string;
  structure_b: string;
  added_edges: Array<{ source: string; target: string }>;
  removed_edges: Array<{ source: string; target: string }>;
  shared_edges: number;
}

export interface CompareGraphDiffResult {
  structure_a: string;
  structure_b: string;
  diff: Record<string, unknown>;
}

export interface ShortestPathResult {
  path: string[];
  length: number;
}

// ---------- KPIs ----------

export interface KPIs {
  structure_count: number;
  embedding_count: number;
  pipeline_runs: number;
  hypothesis_count: number;
}

// ---------- Agent Chat ----------

export interface AgentChatRequest {
  message: string;
  session_id?: string | null;
  structure_id?: string | null;
  viewport_state?: Record<string, unknown> | null;
}

export interface AgentChatResponse {
  response: string;
  session_id: string;
  viewport_directives?: ViewportDirective[] | null;
  tool_results?: Record<string, unknown>[] | null;
  telemetry?: {
    tools_used?: string[];
    reasoning_steps?: number;
    confidence?: number;
  } | null;
}

export interface AgentTelemetrySnapshot {
  session_id: string;
  tool_calls: number;
  last_activity?: string | null;
}

// ---------- Hydration ----------

export interface HydrationResponse {
  structure_id: string;
  hydration_sites: number;
  status: string;
}

// ---------- RCSB ----------

export interface RCSBSearchRequest {
  query: string;
  limit?: number;
}

export interface RCSBSearchResult {
  pdb_id: string;
  title?: string | null;
  resolution?: number | null;
  method?: string | null;
  chains?: string[] | null;
  residue_count?: number | null;
}

// ---------- Hypotheses ----------

export type HypothesisStatus = "gathering" | "supported" | "contradicted" | "inconclusive";

export interface Hypothesis {
  hypothesis_id: string;
  structure_id?: string | null;
  statement: string;
  status: HypothesisStatus;
  confidence?: number | null;
  evidence?: Evidence[];
  created_at?: string | null;
  updated_at?: string | null;
}

export interface HypothesisCreateRequest {
  structure_id?: string | null;
  statement: string;
}

export interface Evidence {
  evidence_id: string;
  hypothesis_id: string;
  content: string;
  supports: boolean;
  created_at?: string | null;
}

export interface EvidenceCreateRequest {
  content: string;
  supports: boolean;
}

// ---------- Data Tools ----------

export interface ResidueSearchRequest {
  query?: string | null;
  min_cone_depth?: number | null;
  max_epistemic?: number | null;
}

export interface ResidueSearchResult {
  residue_id: string;
  residue_name?: string | null;
  chain_label?: string | null;
  cone_depth?: number | null;
  epistemic_uncertainty?: number | null;
}

export interface AllostericSitesData {
  structure_id: string;
  sites: Array<{
    site_id: number;
    residue_ids: string[];
    score: number;
  }>;
}

export interface Annotation {
  annotation_id: string;
  structure_id: string;
  residue_id?: string | null;
  content: string;
  created_at?: string | null;
}

export interface AnnotationCreateRequest {
  residue_id?: string | null;
  content: string;
}

export interface ProvenanceRun {
  run_id: string;
  structure_id: string;
  modules: string[];
  status: string;
  created_at?: string | null;
}

export interface RunSummary {
  run_id: string;
  structure_id: string;
  status: string;
  metrics?: Record<string, number>;
}

export interface RunCompareResult {
  run_a: string;
  run_b: string;
  diff: Record<string, unknown>;
}

export interface ExportRequest {
  format: "json" | "csv" | "pdb";
  include_embeddings?: boolean;
}

export interface ExportResponse {
  download_url: string;
  filename: string;
}

export interface PlotRequest {
  plot_type: string;
  structure_id: string;
  params?: Record<string, unknown>;
}

export interface PlotResponse {
  filename: string;
  plot_type: string;
}

export interface DisplacementResult {
  structure_a: string;
  structure_b: string;
  residue_displacements: Array<{
    residue_id: string;
    displacement: number;
  }>;
  mean_displacement: number;
}

// ---------- Compare State ----------

export interface CompareState {
  active: boolean;
  secondaryStructure: Structure | null;
  displacements: DisplacementResult | null;
  graphDiff: CompareGraphDiffResult | null;
  loading: boolean;
  error: string | null;
}

// ---------- API Error ----------

export interface ApiError {
  error: string;
  message: string;
}
