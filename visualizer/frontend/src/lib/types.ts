/** API response types matching the backend design */

export interface Structure {
  structure_id: string;
  pdb_id: string;
  title: string;
  resolution: number | null;
  method: string;
  source: string;
  chains: string[];
  residue_count: number;
  has_embeddings: boolean;
  last_run_id: string | null;
  ingested_at: string;
}

export interface PipelineJob {
  job_id: string;
  structure_id: string;
  status: "queued" | "running" | "complete" | "failed";
  current_step: string;
  progress: number;
  started_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface ResidueEmbedding {
  residue_id: string;
  residue_index: number;
  chain_label: string;
  x: number;
  y: number;
  cone_depth: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
}

export interface EmbeddingData {
  structure_id: string;
  curvature: number;
  residues: ResidueEmbedding[];
}

export interface GraphMetrics {
  structure_id: string;
  residues: Array<{
    residue_id: string;
    centrality: number;
    clustering_coefficient: number;
    is_bridge: boolean;
    dehydron_score: number;
  }>;
}

export interface KPIs {
  total_structures: number;
  mean_uncertainty: number;
  avg_inference_seconds: number;
  active_jobs: number;
  model_status: "ready" | "training" | "error";
  db_connected: boolean;
  science_container_available: boolean;
}

export interface IngestRequest {
  pdb_id: string;
}

export interface IngestResponse {
  structure_id: string;
  pdb_id: string;
  title: string;
  residue_count: number;
  chains: string[];
}

export interface PipelineRunRequest {
  structure_id: string;
  modules?: string[];
  target?: string;
}

export interface ResistanceSummaryContext {
  lambda_2: number;
  hinge_count: number;
  high_sensitivity_count: number;
  moderate_count: number;
  stable_count: number;
}

export interface TopResidueInfo {
  residue_id: string;
  epistemic_uncertainty: number;
}

/** KRAS Buffering Atlas + ASAR (Allosteric Structure-Activity Relationship) types */
export interface AtlasAllele {
  allele: string;
  ic50: number;           // nM - user editable for the current chemotype profile
  x: number;              // Core Frustration
  y: number;              // Relay Flux
}

export interface ASARFit {
  betaX: number;          // coefficient for Core Frustration
  betaY: number;          // coefficient for Relay Flux
  intercept: number;
  r2: number;
  nPairs: number;
  angleDeg: number;       // angle of the (betaX, betaY) vector from X-axis
  magnitude: number;
}

export interface ViewportState {
  // Structure identity
  structure_id: string | null;
  structure_title: string | null;
  is_radar_active?: boolean;
  selected_pocket_id?: number | null;

  // Poincaré disc
  poincare: {
    color_mode: string;
    mobius_focus_enabled: boolean;
    mobius_focus_residue: string | null;
    selected_residue: SelectedResidueInfo | null;
    brush_selected_ids: string[];
  };

  // 3D Molecular Viewer
  viewer_3d: {
    color_mode: string;
    risk_threshold: number;
    highlighted_residue_ids: string[];
  };

  // Hydration data summary
  data_summary: {
    residue_count: number;
    source_leak_count: number;
    hypothesis_count: number;
    hypothesis_status_distribution: Record<string, number> | null;
    provenance_run_count: number;
    annotation_count: number;
    top_uncertainty_residues: TopResidueInfo[];
    persistence_status: Record<string, boolean>;
    resistance_summary: ResistanceSummaryContext | null;
  } | null;

  // Active tool panel
  active_panel: string | null;

  // Pipeline status
  pipeline: {
    status: string;
    current_step: string | null;
    progress: number | null;
  };

  // Compare mode context (present only when compare mode is active)
  compare?: {
    secondary_structure_id: string;
    top_movers: Array<{ residue_id: string; displacement: number }>;
    edge_diff: { gained: number; lost: number; changed: number };
  } | null;

  // Data inspector context (present only when data_inspector panel is active)
  data_inspector?: {
    phases_computed: string[];
    phase_counts: Record<string, number>;
    top_druggability_pocket: number | null;
    top_drug_candidate_score: number | null;
    admet_pass_rate: number | null;
  } | null;
}

export interface AgentChatRequest {
  message: string;
  session_id: string;
  context: ViewportState;
}

export interface AgentChatResponse {
  response: string;
  session_id: string;
  referenced_residues?: string[];
  viewport_directives?: ViewportDirective[];
  tool_calls?: string[];
  telemetry?: {
    trace_id: string;
    tool_calls: Array<{
      tool_name: string;
      arguments?: Record<string, unknown>;
      result_summary?: string | null;
      is_error: boolean;
      duration_ms: number;
    }>;
    ground_truth_facts?: Record<string, unknown>;
    total_duration_ms: number;
    llm_calls: number;
    tokens: {
      input: number;
      output: number;
      total: number;
    };
    token_attribution?: TokenAttribution;
    llm_call_breakdown?: LLMCallBreakdown[];
  };
}

export interface TokenAttributionEntry {
  chars?: number;
  estimated_tokens: number;
  percent: number;
}

export interface TokenAttribution {
  input?: Record<string, TokenAttributionEntry>;
  output?: Record<string, TokenAttributionEntry>;
  total?: Record<string, TokenAttributionEntry>;
}

export interface LLMCallBreakdown {
  call_index: number;
  input_tokens: number;
  output_tokens: number;
  input: Record<string, TokenAttributionEntry>;
  output: Record<string, TokenAttributionEntry>;
}

export interface AgentTelemetrySnapshot {
  session_id: string;
  trace_count: number;
  latest_trace: {
    trace_id: string;
    total_duration_ms: number;
    llm_calls: number;
    tokens: { input: number; output: number; total: number };
    token_attribution?: TokenAttribution;
    llm_call_breakdown?: LLMCallBreakdown[];
    tool_calls: Array<{
      tool_name: string;
      is_error: boolean;
      duration_ms: number;
    }>;
  } | null;
  recent_traces: Array<{
    trace_id: string;
    total_duration_ms: number;
    llm_calls: number;
    tokens: { input: number; output: number; total: number };
    token_attribution?: TokenAttribution;
    llm_call_breakdown?: LLMCallBreakdown[];
    tool_calls: Array<{
      tool_name: string;
      is_error: boolean;
      duration_ms: number;
    }>;
  }>;
  viewport: {
    active_connections: number;
    active_sessions: number;
    delivery_success_count: number;
    delivery_failure_count: number;
    delivery_failure_rate: number;
  };
}

export interface ApiError {
  error: string;
  message: string;
}

// --- RCSB Search ---

export interface RCSBSearchRequest {
  mode: "text" | "sequence" | "structure" | "functional";
  query: string;
  organism?: string;
  max_resolution?: number;
  min_identity?: number;
  evalue_cutoff?: number;
  max_results?: number;
}

export interface RCSBSearchResult {
  pdb_id: string;
  title: string;
  resolution: number | null;
  method: string;
  organism: string | null;
  similarity_score?: number;
}

// --- Graph Topology ---

export interface GraphMetricsData {
  structure_id: string;
  metrics: Array<{
    residue_id: string;
    degree: number;
    betweenness: number;
    clustering_coefficient: number;
    closeness: number;
    eigenvector_centrality: number;
    is_bridge: boolean;
    conductance: number;
  }>;
}

export interface GraphCompareResult {
  structure_a: string;
  structure_b: string;
  edge_diff: {
    gained: Array<{ source: string; target: string; edge_type: string }>;
    lost: Array<{ source: string; target: string; edge_type: string }>;
    changed: Array<{ source: string; target: string; weight_delta: number }>;
  };
  metric_diff: Array<{
    position: string;
    betweenness_delta: number;
    degree_delta: number;
  }>;
}

export interface ShortestPathResult {
  source: string;
  target: string;
  path: string[];
  path_length: number;
  total_distance: number;
  disconnected: boolean;
}

// --- Hypothesis Engine ---

export interface Prediction {
  prediction_id: string;
  statement: string;
  test_tool: string | null;
  test_params: Record<string, unknown> | null;
  threshold: string | null;
  passed: boolean | null;
  tested_at: string | null;
}

export interface Evidence {
  evidence_id: string;
  source_tool: string;
  supports: boolean;
  strength: number;
  description: string;
  gathered_at: string;
}

export interface Hypothesis {
  hypothesis_id: string;
  structure_id: string;
  statement: string;
  mechanism: string | null;
  status: "proposed" | "gathering" | "supported" | "contradicted" | "inconclusive";
  confidence: number;
  predictions: Prediction[];
  evidence_supporting: number;
  evidence_contradicting: number;
  created_at: string;
  updated_at: string;
}

export interface HypothesisCreateRequest {
  structure_id: string;
  statement: string;
  predictions: Array<{
    statement: string;
    test_tool?: string;
    test_params?: Record<string, unknown>;
    threshold?: string;
  }>;
  mechanism?: string;
}

export interface EvidenceCreateRequest {
  source_tool: string;
  supports: boolean;
  strength: number;
  description: string;
}

// --- Data Tools ---

export interface ResidueSearchRequest {
  chain?: string;
  residue_name?: string;
  min_uncertainty?: number;
  max_uncertainty?: number;
  min_cone_depth?: number;
  max_cone_depth?: number;
  uncertainty_type?: "epistemic" | "aleatoric" | "total";
  limit?: number;
}

export interface ResidueSearchResult {
  residue_id: string;
  chain_label: string;
  residue_name: string;
  cone_depth: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
}

export interface AllostericSite {
  site_id: string;
  confidence: number;
  residue_ids: string[];
}

export interface AllostericSitesData {
  structure_id: string;
  sites: AllostericSite[];
}

export interface Annotation {
  annotation_id: string;
  structure_id: string;
  residue_ids: string[] | null;
  annotation: string;
  annotation_type: "finding" | "hypothesis" | "note" | "warning";
  created_at: string;
}

export interface AnnotationCreateRequest {
  residue_ids?: string[];
  annotation: string;
  annotation_type: "finding" | "hypothesis" | "note" | "warning";
}

export interface ProvenanceRun {
  run_id: string;
  structure_id: string;
  model_version: string;
  pipeline_name: string;
  run_type: string;
  parent_run_id: string | null;
  started_at: string;
  asset_count?: number;
}

export interface RunSummary {
  run_id: string;
  phases: string[];
  duration_seconds: number;
  assets_created: number;
  errors: string[];
}

export interface RunCompareResult {
  run_a: string;
  run_b: string;
  residue_deltas: Array<{
    residue_id: string;
    cone_depth_delta: number;
    uncertainty_delta: number;
  }>;
}

// --- Plots ---

export interface PlotRequest {
  structure_id: string;
  plot_type:
    | "poincare_disc"
    | "uncertainty_profile"
    | "cone_depth_histogram"
    | "wt_vs_mutant"
    | "persistence_barcode"
    | "source_leak_map";
  parameters?: Record<string, unknown>;
}

export interface PlotResponse {
  success: boolean;
  file_path: string | null;
  plot_type: string;
  message: string;
  download_url: string | null;
}

// --- Viewport Directives ---

export interface ViewportDirective {
  action: "highlight" | "set_metric" | "focus" | "clear" | "annotate" | "compare_runs" | "open_tab" | "close_tab" | "set_color_mode";
  structure_id?: string;
  highlight_groups?: Array<{
    residue_ids: string[];
    color: string;
    style: "glow" | "pulse" | "outline" | "color";
    label?: string;
  }>;
  focus_residues?: string[];
  metric?: string;
  message?: string;
  component_id?: string;
  color_mode?: string;
}

// --- Hydration ---

export interface SourceLeakData {
  structure_id: string;
  leaks: Array<{
    residue_id: string;
    leak_score: number;
    source: string;
  }>;
  /** Raw field name from API (backend returns 'source_leaks' not 'leaks') */
  source_leaks?: Array<{
    residue_id: string;
    leak_score: number;
    source: string;
  }>;
  count?: number;
}

// --- Resistance Sensitivity ---

export interface ResistanceResidue {
  residue_id: string;
  sensitivity_score: number;
  coupling_count: number;
  is_hinge: boolean;
  classification: "high_sensitivity" | "moderate" | "stable";
}

export interface ResistanceData {
  residues: ResistanceResidue[];
  spectral: {
    lambda_2: number;
    hinge_count: number;
  };
  pathways?: Array<Record<string, unknown>>;
}

export interface HydrationResponse {
  structure_id: string;
  embeddings: EmbeddingData | null;
  graph_metrics: GraphMetricsData | null;
  allosteric_sites: AllostericSitesData | null;
  source_leaks: SourceLeakData | null;
  resistance_data: ResistanceData | null;
  hypotheses: Hypothesis[] | null;
  provenance_runs: ProvenanceRun[] | null;
  annotations: Annotation[] | null;
  pharmacophore_pockets: PharmacophoreData | null;
  drug_candidates: DrugCandidateData | null;
  phase5_pharmacophore?: {
    structure_id: string;
    pockets?: PharmacophoreRow[];
    pharmacophores?: PharmacophoreRow[];
    count?: number;
  } | null;
  phase6_drug_candidates?: {
    structure_id: string;
    candidates?: DrugCandidateRow[];
    drug_candidates?: DrugCandidateRow[];
    count?: number;
    admet_passed_count?: number;
    state_selective_count?: number;
  } | null;
  phase4_resistance?: ResistanceData | null;
  buffering_atlas?: Record<string, unknown> | null;
  persistence_status: {
    embeddings_persisted: boolean;
    graph_persisted: boolean;
    sites_persisted: boolean;
    phase5_persisted?: boolean;
    phase6_persisted?: boolean;
  };
}

// --- Phase 5: Pharmacophore Pockets ---

export interface PharmacophoreRow {
  pocket_index: number;
  druggability_score: number;
  residue_count: number;
  volume_estimate: number;
  allosteric_coupling: number;
  center_x: number;
  center_y: number;
  center_z: number;
  residue_ids: string[];
  // Derived for allosteric remote control viz (from hydration enrichment)
  connected_allosteric_locks?: string[];
  connecting_pathway_count?: number;
  connecting_coupling_sum?: number;
  connecting_pathways?: Array<{
    source_residue?: string;
    target_residue?: string;
    coupling_strength?: number;
    [key: string]: unknown;
  }>;
}

export interface PharmacophoreData {
  structure_id: string;
  pockets: PharmacophoreRow[];
  count?: number;
}

// --- Phase 6: Drug Candidates ---

export interface DrugCandidateRow {
  pocket_index: number;
  combined_druggability: number;
  accessibility_score: number;
  binding_potential: number;
  admet_pass: boolean;
  selectivity_ratio: number;
  is_state_selective: boolean;
}

export interface DrugCandidateData {
  structure_id: string;
  candidates: DrugCandidateRow[];
  count?: number;
  admet_passed_count?: number;
  state_selective_count?: number;
}

// --- Viewport Context (for Agent Chat) ---

export interface SelectedResidueInfo {
  residue_id: string;
  residue_name: string | null;
  chain_label: string | null;
  epistemic_uncertainty: number | null;
  cone_depth: number | null;
}

export type PoincareColorMode = "cone_depth" | "uncertainty";
export type StructureColorModeType = "spectrum" | "cone_depth" | "epistemic" | "aleatoric" | "plasticity" | "allosteric" | "resistance" | "pockets" | "drug_candidates";
export type ActivePanelName = "rcsb_search" | "graph_topology" | "hypothesis" | "data_tools" | "provenance" | "plot_generator" | "compare" | "data_inspector" | null;

// --- Compare Mode ---

export interface DisplacementRow {
  residue_id: string;
  chain: string;
  index: number;
  primary_depth: number;
  secondary_depth: number;
  depth_delta: number;
  displacement: number;
}

export interface DisplacementResult {
  structure_a: string;
  structure_b: string;
  displacements: DisplacementRow[];
  total_matched: number;
  summary: {
    mean_displacement: number;
    max_displacement: number;
    movers_above_threshold: number;
    threshold: number;
  };
}

export interface CompareGraphDiffResult {
  structure_a: string;
  structure_b: string;
  edge_diff: {
    gained_count: number;
    lost_count: number;
    changed_count: number;
    gained: Array<{ source: string; target: string; edge_type: string }>;
    lost: Array<{ source: string; target: string; edge_type: string }>;
    changed: Array<{ source: string; target: string; weight_delta: number }>;
  };
  hbond: {
    gained: number;
    lost: number;
  };
  metric_deltas: Array<{
    residue_id: string;
    chain: string;
    betweenness_delta: number;
    degree_delta: number;
    clustering_delta: number;
  }>;
  total_matched_residues: number;
}

export interface CompareState {
  active: boolean;
  secondaryStructure: Structure | null;
  displacements: DisplacementResult | null;
  graphDiff: CompareGraphDiffResult | null;
  loading: boolean;
  error: string | null;
}

export interface ViewportStateProps {
  poincareColorMode: PoincareColorMode;
  poincareSelectedResidue: SelectedResidueInfo | null;
  mobiusFocus: boolean;
  brushSelection: string[];
  viewerColorMode: StructureColorModeType;
  riskThreshold: number;
  activePanel: ActivePanelName;
}

// --- Export ---

export interface ExportRequest {
  format: "csv" | "json";
}

export interface ExportResponse {
  download_url: string;
  format: string;
  structure_id: string;
}
