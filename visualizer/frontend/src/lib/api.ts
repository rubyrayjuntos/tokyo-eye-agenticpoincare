import type {
  Structure,
  PipelineJob,
  EmbeddingData,
  GraphMetrics,
  KPIs,
  LifecycleStatus,
  IngestRequest,
  IngestResponse,
  PipelineRunRequest,
  AgentChatRequest,
  AgentChatResponse,
  AgentTelemetrySnapshot,
  RCSBSearchRequest,
  RCSBSearchResult,
  GraphMetricsData,
  GraphCompareResult,
  ShortestPathResult,
  Hypothesis,
  HypothesisCreateRequest,
  EvidenceCreateRequest,
  Evidence,
  ResidueSearchRequest,
  ResidueSearchResult,
  AllostericSitesData,
  Annotation,
  AnnotationCreateRequest,
  ProvenanceRun,
  RunSummary,
  RunCompareResult,
  ExportRequest,
  ExportResponse,
  PlotRequest,
  PlotResponse,
  HydrationResponse,
  DisplacementResult,
  CompareGraphDiffResult,
  StructureReadiness,
  StructureAuditResponse,
} from "./types";

// In development, use relative paths so requests go through Vite's proxy.
// In production (or when VITE_API_URL is set), use the explicit URL.
const DEFAULT_API_BASE =
  typeof window !== "undefined" && import.meta.env.DEV
    ? ""  // relative — Vite proxy handles /api → http://localhost:8000
    : typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}:8000`
      : "http://localhost:8000";

const BASE_URL = import.meta.env.VITE_API_URL || DEFAULT_API_BASE;

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({} as Record<string, unknown>));
      const detail = body.detail;
      const detailMessage =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail
                .map((item) =>
                  typeof item === "string"
                    ? item
                    : item && typeof item === "object" && "msg" in item
                      ? String((item as { msg: unknown }).msg)
                      : JSON.stringify(item),
                )
                .join("; ")
            : detail && typeof detail === "object"
              ? JSON.stringify(detail)
              : null;
      const message =
        detailMessage ||
        (typeof body.message === "string" ? body.message : null) ||
        res.statusText;
      throw new Error(message || `Request failed (${res.status})`);
    }

    return res.json();
  }

  // --- Structures ---

  async getStructures(): Promise<Structure[]> {
    const res = await this.request<{ structures: Structure[]; count: number }>("/api/structures");
    return res.structures;
  }

  async getEmbeddings(structureId: string): Promise<EmbeddingData> {
    return this.request<EmbeddingData>(
      `/api/structures/${structureId}/embeddings`
    );
  }

  async getMetrics(structureId: string): Promise<GraphMetrics> {
    return this.request<GraphMetrics>(
      `/api/structures/${structureId}/metrics`
    );
  }

  // --- Ingestion ---

  async ingest(pdbId: string): Promise<IngestResponse> {
    const body: IngestRequest = { pdb_id: pdbId };
    return this.request<IngestResponse>("/api/ingest", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async getStructureReadiness(structureId: string): Promise<StructureReadiness> {
    return this.request<StructureReadiness>(
      `/api/structures/${structureId}/readiness`
    );
  }

  async getStructureAudit(
    structureId: string,
    options?: { limit?: number; severity?: "info" | "warning" | "error" },
  ): Promise<StructureAuditResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set("limit", String(options.limit));
    if (options?.severity) params.set("severity", options.severity);
    const qs = params.toString();
    return this.request<StructureAuditResponse>(
      `/api/structures/${structureId}/audit${qs ? `?${qs}` : ""}`,
    );
  }

  // --- Pipeline ---

  async runPipeline(req: PipelineRunRequest): Promise<PipelineJob> {
    return this.request<PipelineJob>("/api/pipeline/run", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async getPipelineStatus(jobId: string): Promise<PipelineJob> {
    return this.request<PipelineJob>(`/api/pipeline/status/${jobId}`);
  }

  // --- KPIs ---

  async getKPIs(): Promise<KPIs> {
    return this.request<KPIs>("/api/kpis");
  }

  // --- GNN Model Lifecycle ---

  async getLifecycleStatus(): Promise<LifecycleStatus> {
    return this.request<LifecycleStatus>("/api/lifecycle/status");
  }

  async getLifecycleLineages(): Promise<{ lineages: Record<string, unknown>[] }> {
    return this.request("/api/lifecycle/lineages");
  }

  async getLifecycleRuns(params?: {
    experiment?: string;
    lineage_id?: string;
    max_results?: number;
  }): Promise<{ experiment: string; runs: Record<string, unknown>[]; error?: string }> {
    const q = new URLSearchParams();
    if (params?.experiment) q.set("experiment", params.experiment);
    if (params?.lineage_id) q.set("lineage_id", params.lineage_id);
    if (params?.max_results) q.set("max_results", String(params.max_results));
    const suffix = q.toString() ? `?${q}` : "";
    return this.request(`/api/lifecycle/runs${suffix}`);
  }

  async enqueueLifecycleTrain(body: {
    lineage_id?: "v6" | "v6.5";
    run_id?: string;
    preset?: string;
    corpus?: string;
    device?: string;
    no_warm_start?: boolean;
  }): Promise<{
    job_id: string;
    status: string;
    suggested_command?: string;
    note?: string;
  }> {
    return this.request("/api/lifecycle/train", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async registerLifecycleModel(body: {
    lineage_id: "v6" | "v6.5";
    checkpoint_path: string;
    alias?: "champion" | "challenger";
    run_id?: string;
    sync_contract?: boolean;
  }): Promise<Record<string, unknown>> {
    return this.request("/api/lifecycle/register", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async promoteLifecycleModel(body: {
    lineage_id: "v6" | "v6.5";
    checkpoint_path?: string;
    version?: string;
    alias?: "champion" | "challenger";
    run_id?: string;
    sync_contract?: boolean;
  }): Promise<Record<string, unknown>> {
    return this.request("/api/lifecycle/promote", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async addStructureToCorpus(body: {
    structure_id: string;
    manifest_path?: string;
    pdb_id?: string;
    chain?: string;
    enabled?: boolean;
  }): Promise<{ action: string; pdb_id: string; manifest: string }> {
    return this.request("/api/lifecycle/corpus/add", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async getCheckpointPreview(params: {
    structure_id: string;
    checkpoint_path?: string;
    alias?: string;
    lineage_id?: string;
  }): Promise<Record<string, unknown>> {
    const q = new URLSearchParams({ structure_id: params.structure_id });
    if (params.checkpoint_path) q.set("checkpoint_path", params.checkpoint_path);
    if (params.alias) q.set("alias", params.alias);
    if (params.lineage_id) q.set("lineage_id", params.lineage_id);
    return this.request(`/api/lifecycle/checkpoint-preview?${q}`);
  }

  // --- Agent Chat ---

  async chat(req: AgentChatRequest): Promise<AgentChatResponse> {
    return this.request<AgentChatResponse>("/api/agent/chat", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async getAgentTelemetry(sessionId: string): Promise<AgentTelemetrySnapshot> {
    return this.request<AgentTelemetrySnapshot>(`/api/agent/telemetry/${sessionId}`);
  }

  // --- Hydration ---

  async hydrate(structureId: string): Promise<HydrationResponse> {
    return this.request<HydrationResponse>(
      `/api/structures/${structureId}/hydrate`
    );
  }

  // --- RCSB Search ---

  private async rcsbPost(
    path: string,
    body: Record<string, unknown>,
  ): Promise<RCSBSearchResult[]> {
    const res = await this.request<{
      results?: RCSBSearchResult[];
      count?: number;
    }>(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return Array.isArray(res.results) ? res.results : [];
  }

  async rcsbSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.rcsbPost("/api/rcsb/search", {
      query: req.query,
      organism: req.organism,
      max_resolution: req.max_resolution,
      max_results: req.max_results ?? 10,
    });
  }

  async rcsbSequenceSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.rcsbPost("/api/rcsb/sequence-search", {
      sequence: req.query,
      evalue_cutoff: req.evalue_cutoff ?? 0.1,
      min_identity:
        req.min_identity != null
          ? req.min_identity > 1
            ? req.min_identity / 100
            : req.min_identity
          : 0,
      max_results: req.max_results ?? 10,
    });
  }

  async rcsbStructureSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.rcsbPost("/api/rcsb/structure-search", {
      pdb_id: req.query.trim().toUpperCase().slice(0, 4),
      max_results: req.max_results ?? 10,
    });
  }

  async rcsbInfo(pdbId: string): Promise<RCSBSearchResult> {
    return this.request<RCSBSearchResult>(`/api/rcsb/info/${pdbId}`);
  }

  // --- Graph Topology ---

  async getGraphMetrics(structureId: string): Promise<GraphMetricsData> {
    return this.request<GraphMetricsData>(
      `/api/graph/${structureId}/metrics`
    );
  }

  async getGraphBridges(structureId: string): Promise<{ residues: string[] }> {
    return this.request<{ residues: string[] }>(
      `/api/graph/${structureId}/bridges`
    );
  }

  async getGraphHbonds(structureId: string): Promise<{ edges: Array<{ source: string; target: string }> }> {
    return this.request<{ edges: Array<{ source: string; target: string }> }>(
      `/api/graph/${structureId}/hbonds`
    );
  }

  async getShortestPath(
    structureId: string,
    source: string,
    target: string
  ): Promise<ShortestPathResult> {
    return this.request<ShortestPathResult>(
      `/api/graph/${structureId}/shortest-path`,
      {
        method: "POST",
        body: JSON.stringify({ source, target }),
      }
    );
  }

  async compareGraphs(
    structureA: string,
    structureB: string
  ): Promise<GraphCompareResult> {
    return this.request<GraphCompareResult>("/api/graph/compare", {
      method: "POST",
      body: JSON.stringify({ structure_a: structureA, structure_b: structureB }),
    });
  }

  // --- Hypotheses ---

  async getHypotheses(
    structureId?: string,
    status?: string
  ): Promise<Hypothesis[]> {
    const params = new URLSearchParams();
    if (structureId) params.set("structure_id", structureId);
    if (status) params.set("status", status);
    const qs = params.toString();
    return this.request<Hypothesis[]>(
      `/api/hypotheses${qs ? `?${qs}` : ""}`
    );
  }

  async createHypothesis(req: HypothesisCreateRequest): Promise<Hypothesis> {
    return this.request<Hypothesis>("/api/hypotheses", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async testHypothesis(hypothesisId: string): Promise<Hypothesis> {
    return this.request<Hypothesis>(
      `/api/hypotheses/${hypothesisId}/test`,
      { method: "POST" }
    );
  }

  async addEvidence(
    hypothesisId: string,
    req: EvidenceCreateRequest
  ): Promise<Evidence> {
    return this.request<Evidence>(
      `/api/hypotheses/${hypothesisId}/evidence`,
      {
        method: "POST",
        body: JSON.stringify(req),
      }
    );
  }

  async evaluateHypothesis(hypothesisId: string): Promise<Hypothesis> {
    return this.request<Hypothesis>(
      `/api/hypotheses/${hypothesisId}/evaluate`,
      { method: "POST" }
    );
  }

  // --- Data Tools ---

  async searchResidues(
    structureId: string,
    req: ResidueSearchRequest
  ): Promise<ResidueSearchResult[]> {
    return this.request<ResidueSearchResult[]>(
      `/api/data/${structureId}/search-residues`,
      {
        method: "POST",
        body: JSON.stringify(req),
      }
    );
  }

  async getAllostericSites(structureId: string): Promise<AllostericSitesData> {
    return this.request<AllostericSitesData>(
      `/api/data/${structureId}/allosteric-sites`
    );
  }

  async getProvenance(structureId: string): Promise<ProvenanceRun[]> {
    return this.request<ProvenanceRun[]>(
      `/api/data/${structureId}/provenance`
    );
  }

  async exportStructureData(
    structureId: string,
    req: ExportRequest
  ): Promise<ExportResponse> {
    return this.request<ExportResponse>(
      `/api/data/${structureId}/export`,
      {
        method: "POST",
        body: JSON.stringify(req),
      }
    );
  }

  async getAnnotations(structureId: string): Promise<Annotation[]> {
    return this.request<Annotation[]>(
      `/api/data/${structureId}/annotations`
    );
  }

  async createAnnotation(
    structureId: string,
    req: AnnotationCreateRequest
  ): Promise<Annotation> {
    return this.request<Annotation>(
      `/api/data/${structureId}/annotations`,
      {
        method: "POST",
        body: JSON.stringify(req),
      }
    );
  }

  async getRunSummary(runId: string): Promise<RunSummary> {
    return this.request<RunSummary>(`/api/data/runs/${runId}/summary`);
  }

  async compareRuns(runA: string, runB: string): Promise<RunCompareResult> {
    return this.request<RunCompareResult>("/api/data/runs/compare", {
      method: "POST",
      body: JSON.stringify({ run_a: runA, run_b: runB }),
    });
  }

  // --- Plots ---

  async generatePlot(req: PlotRequest): Promise<PlotResponse> {
    return this.request<PlotResponse>("/api/plots/generate", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  getPlotUrl(filename: string): string {
    return `${this.baseUrl}/api/plots/${filename}`;
  }

  // --- Compare ---

  async compareEmbeddings(
    idA: string,
    idB: string,
    threshold?: number
  ): Promise<DisplacementResult> {
    const qs = threshold != null ? `?threshold=${threshold}` : "";
    return this.request<DisplacementResult>(
      `/api/compare/embeddings/${idA}/${idB}${qs}`
    );
  }

  async compareGraphsDirect(
    idA: string,
    idB: string
  ): Promise<CompareGraphDiffResult> {
    return this.request<CompareGraphDiffResult>(
      `/api/compare/graphs/${idA}/${idB}`
    );
  }

  // --- Workbench session ---

  async getOrchestrationSnapshot(sessionId: string): Promise<Record<string, unknown>> {
    return this.request(`/api/session/${sessionId}/orchestration`);
  }

  async postOrchestrationEvent(
    sessionId: string,
    event: { type: string; phase?: string; tool?: string },
  ): Promise<Record<string, unknown>> {
    return this.request(`/api/session/${sessionId}/orchestration/event`, {
      method: "POST",
      body: JSON.stringify(event),
    });
  }

  async getWorkspaceLayout(sessionId: string): Promise<{
    session_id: string;
    workspace_id: string;
    layout_json: Record<string, unknown>;
    active_phase_group: string;
    updated_at: string | null;
  }> {
    return this.request(`/api/session/${sessionId}/workspace-layout`);
  }

  async putWorkspaceLayout(
    sessionId: string,
    payload: {
      workspace_id?: string;
      layout_json: Record<string, unknown>;
      active_phase_group: string;
    },
  ): Promise<Record<string, unknown>> {
    return this.request(`/api/session/${sessionId}/workspace-layout`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  }
}

export const api = new ApiClient(BASE_URL);
