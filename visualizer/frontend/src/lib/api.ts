import type {
  Structure,
  PipelineJob,
  EmbeddingData,
  GraphMetrics,
  KPIs,
  IngestRequest,
  IngestResponse,
  PipelineRunRequest,
  AgentChatRequest,
  AgentChatResponse,
  AgentTelemetrySnapshot,
  ApiError,
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
      const err: ApiError = await res.json().catch(() => ({
        error: "unknown",
        message: res.statusText,
      }));
      throw err;
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

  async rcsbSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.request<RCSBSearchResult[]>("/api/rcsb/search", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async rcsbSequenceSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.request<RCSBSearchResult[]>("/api/rcsb/sequence-search", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  async rcsbStructureSearch(req: RCSBSearchRequest): Promise<RCSBSearchResult[]> {
    return this.request<RCSBSearchResult[]>("/api/rcsb/structure-search", {
      method: "POST",
      body: JSON.stringify(req),
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
}

export const api = new ApiClient(BASE_URL);
