# Design Document

## Overview

This spec surfaces all existing agent tools as user-interactive frontend panels while maintaining agent accessibility. The architecture adds new REST endpoints that wrap the existing tool functions, a structure hydration system that loads all available data on structure selection, and frontend panels for each tool group.

Key principle: every tool is accessible both via direct user interaction (buttons, forms) AND via the agent (which calls the same endpoints). All writes go through the Normalizer. The frontend receives viewport directives from both user actions and agent responses.

## Architecture

```mermaid
graph TB
    subgraph Frontend [React Dashboard :3000]
        Hydrator[Structure Hydrator]
        RCSB[RCSB Search Panel]
        Graph[Graph Topology Panel]
        Hyp[Hypothesis Panel]
        Data[Data Tools Panel]
        Plot[Plot Generator]
        VizCtrl[Visualization Controls]
        Viewer[Molecular Viewer + Poincaré]
    end

    subgraph Backend [FastAPI :8000]
        HydAPI[GET /api/structures/{id}/hydrate]
        RCSBAPI[/api/rcsb/*]
        GraphAPI[/api/graph/*]
        HypAPI[/api/hypotheses/*]
        DataAPI[/api/data/*]
        PlotAPI[/api/plots/*]
        AgentAPI[POST /api/agent/chat]
    end

    subgraph Tools [Existing Tool Layer]
        RCSBTools[rcsb.py]
        GraphTools[graph_tools.py]
        HypTools[hypothesis/tools.py]
        DataTools[data_tools.py]
        PlotTools[plotting/tools.py]
    end

    subgraph Data [PostgreSQL + Normalizer]
        DB[(Governed Tables)]
        Norm[Normalizer]
    end

    Hydrator -->|GET /hydrate| HydAPI
    RCSB -->|/api/rcsb/*| RCSBAPI
    Graph -->|/api/graph/*| GraphAPI
    Hyp -->|/api/hypotheses/*| HypAPI
    Data -->|/api/data/*| DataAPI
    Plot -->|/api/plots/*| PlotAPI

    HydAPI --> DataTools
    RCSBAPI --> RCSBTools
    GraphAPI --> GraphTools
    HypAPI --> HypTools
    DataAPI --> DataTools
    PlotAPI --> PlotTools
    AgentAPI --> Tools

    Tools --> Norm
    Norm --> DB
    HydAPI --> DB
```

## Components and Interfaces

### New Backend Routers

#### Router: `agent/coordinator/routers/rcsb.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/rcsb/search` | POST | Text/keyword search with organism/resolution filters |
| `/api/rcsb/sequence-search` | POST | Sequence similarity search (BLAST-like) |
| `/api/rcsb/structure-search` | POST | Structure similarity search (3D alignment) |
| `/api/rcsb/info/{pdb_id}` | GET | Fetch detailed metadata for a PDB entry |

#### Router: `agent/coordinator/routers/graph.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/graph/{structure_id}/metrics` | GET | Per-residue graph metrics (full set) |
| `/api/graph/{structure_id}/bridges` | GET | Bridge/articulation-point residues |
| `/api/graph/{structure_id}/hbonds` | GET | H-bond subgraph |
| `/api/graph/{structure_id}/shortest-path` | POST | Shortest path between two residues |
| `/api/graph/compare` | POST | Compare topology of two structures |

#### Router: `agent/coordinator/routers/hypotheses.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/hypotheses` | GET | List hypotheses (filter by structure_id, status) |
| `/api/hypotheses` | POST | Propose new hypothesis (falsifiability guardrail) |
| `/api/hypotheses/{id}/test` | POST | Execute predictions and update confidence |
| `/api/hypotheses/{id}/evidence` | POST | Add evidence to hypothesis |
| `/api/hypotheses/{id}/evaluate` | POST | Recalculate confidence with decay |

#### Router: `agent/coordinator/routers/data.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/data/{structure_id}/search-residues` | POST | Flexible residue search/filter |
| `/api/data/{structure_id}/allosteric-sites` | GET | Allosteric site clusters |
| `/api/data/{structure_id}/provenance` | GET | Provenance run history |
| `/api/data/{structure_id}/export` | POST | Export CSV/JSON |
| `/api/data/{structure_id}/annotations` | GET | List annotations |
| `/api/data/{structure_id}/annotations` | POST | Add annotation |
| `/api/data/runs/{run_id}/summary` | GET | Run summary |
| `/api/data/runs/compare` | POST | Compare two runs |

#### Router: `agent/coordinator/routers/plots.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/plots/generate` | POST | Generate matplotlib figure, return PNG path |
| `/api/plots/{filename}` | GET | Serve generated plot image |

#### Hydration Endpoint (added to dashboard router)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/structures/{id}/hydrate` | GET | Return all available data for a structure in one call |

### Hydration Response Shape

```typescript
interface HydrationResponse {
  structure_id: string;
  embeddings: EmbeddingData | null;
  graph_metrics: GraphMetricsData | null;
  allosteric_sites: AllostericSitesData | null;
  source_leaks: SourceLeakData | null;
  hypotheses: HypothesisData[] | null;
  provenance_runs: ProvenanceRun[] | null;
  annotations: Annotation[] | null;
  persistence_status: {
    embeddings_persisted: boolean;
    graph_persisted: boolean;
    sites_persisted: boolean;
  };
}
```

### Frontend Components (New)

```
src/components/
  panels/
    RCSBSearchPanel.tsx        # 4-mode search with results cards
    GraphTopologyPanel.tsx     # Metrics table, bridges, H-bonds, paths
    HypothesisPanel.tsx        # Hypothesis cards, propose form, evidence
    DataToolsPanel.tsx         # Residue search, allosteric sites, provenance
    PlotGeneratorPanel.tsx     # Plot type selector, params, inline display
    AnnotationsTimeline.tsx    # Chronological annotation display
    ProvenanceTree.tsx         # Run lineage visualization
  controls/
    VisualizationToolbar.tsx   # Highlight, color metric, focus, clear
    ResidueSelector.tsx        # Multi-select residues across panels
    PersistenceIndicator.tsx   # Green/red status per panel
  context/
    HydrationProvider.tsx      # React context for hydrated structure data
```

### RCSB Search Modes

**Text Search** — Uses `rcsbapi.search.TextQuery` + `AttributeQuery` for organism/resolution filters. Already implemented in `search_rcsb()`.

**Sequence Similarity** — Uses `rcsbapi.search.SequenceQuery` with evalue_cutoff and identity_cutoff parameters. Accepts a FASTA sequence or a PDB chain reference (e.g., "4OBE_A"). Returns hits ranked by sequence identity.

**Structure Similarity** — Uses the RCSB Structure Alignment API (`alignment.rcsb.org`). Accepts a PDB ID + chain, returns structurally similar entries ranked by TM-score or RMSD.

**Functional Annotation** — Uses `AttributeQuery` on GO terms, EC numbers, or Pfam domains. Allows searching by biological function rather than sequence/structure.

## Data Models

### New TypeScript Interfaces

```typescript
interface RCSBSearchRequest {
  mode: "text" | "sequence" | "structure" | "functional";
  query: string;
  organism?: string;
  max_resolution?: number;
  min_identity?: number;  // sequence mode
  evalue_cutoff?: number; // sequence mode
  max_results?: number;
}

interface RCSBSearchResult {
  pdb_id: string;
  title: string;
  resolution: number | null;
  method: string;
  organism: string | null;
  similarity_score?: number;  // identity % or TM-score
}

interface GraphMetricsData {
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

interface GraphCompareResult {
  structure_a: string;
  structure_b: string;
  edge_diff: {
    gained: Array<{source: string; target: string; edge_type: string}>;
    lost: Array<{source: string; target: string; edge_type: string}>;
    changed: Array<{source: string; target: string; weight_delta: number}>;
  };
  metric_diff: Array<{
    position: string;
    betweenness_delta: number;
    degree_delta: number;
  }>;
}

interface ShortestPathResult {
  source: string;
  target: string;
  path: string[];
  path_length: number;
  total_distance: number;
  disconnected: boolean;
}

interface Hypothesis {
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

interface Prediction {
  prediction_id: string;
  statement: string;
  test_tool: string | null;
  test_params: Record<string, unknown> | null;
  threshold: string | null;
  passed: boolean | null;
  tested_at: string | null;
}

interface Evidence {
  evidence_id: string;
  source_tool: string;
  supports: boolean;
  strength: number;
  description: string;
  gathered_at: string;
}

interface Annotation {
  annotation_id: string;
  structure_id: string;
  residue_ids: string[] | null;
  annotation: string;
  annotation_type: "finding" | "hypothesis" | "note" | "warning";
  created_at: string;
}

interface ProvenanceRun {
  run_id: string;
  structure_id: string;
  model_version: string;
  pipeline_name: string;
  run_type: string;
  parent_run_id: string | null;
  started_at: string;
  asset_count?: number;
}

interface PlotRequest {
  structure_id: string;
  plot_type: "poincare_disc" | "uncertainty_profile" | "cone_depth_histogram" | "wt_vs_mutant" | "persistence_barcode" | "source_leak_map";
  parameters?: Record<string, unknown>;
}

interface PlotResponse {
  success: boolean;
  file_path: string | null;
  plot_type: string;
  message: string;
  download_url: string | null;
}

interface ViewportDirective {
  action: "highlight" | "set_metric" | "focus" | "clear" | "annotate" | "compare_runs";
  structure_id?: string;
  highlight_groups?: Array<{
    residue_ids: string[];
    color: string;
    style: "glow" | "pulse" | "outline" | "color";
    label?: string;
  }>;
  metric?: string;
  message?: string;
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


Property 1: Hydration returns all available data types
*For any* structure_id that has computed embeddings, graph metrics, and hypotheses, the hydration endpoint SHALL return non-null values for each corresponding field in the response
**Validates: Requirements 1.1**

Property 2: Agent context reflects hydrated data
*For any* hydration response with non-zero residue count and source leak count, the context builder SHALL produce a context payload containing those exact counts
**Validates: Requirements 1.4**

Property 3: RCSB search results are ordered by mode-appropriate metric
*For any* RCSB search response with multiple results, the results SHALL be ordered by: relevance (text mode), identity percentage descending (sequence mode), or similarity score descending (structure mode)
**Validates: Requirements 2.2, 2.3, 2.4**

Property 4: Graph metrics endpoint returns complete metric set
*For any* structure with graph data, the metrics endpoint SHALL return all 7 metric fields (degree, betweenness, clustering_coefficient, closeness, eigenvector_centrality, is_bridge, conductance) for each residue
**Validates: Requirements 3.1**

Property 5: Bridge endpoint returns only bridge residues
*For any* structure with graph data, all residues returned by the bridges endpoint SHALL have is_bridge=true
**Validates: Requirements 3.2**

Property 6: H-bond endpoint returns only H-bond edges
*For any* structure with graph data, all edges returned by the hbond endpoint SHALL have edge_type='h_bond'
**Validates: Requirements 3.3**

Property 7: Shortest path is valid
*For any* two connected residues in a structure's contact graph, the shortest path response SHALL contain a path where each consecutive pair of residues shares an edge, and total_distance equals the sum of individual edge weights along the path
**Validates: Requirements 3.4**

Property 8: Graph compare edge sets are consistent
*For any* two structures with graph data, the gained edges + common edges SHALL equal the edge set of structure B, and the lost edges + common edges SHALL equal the edge set of structure A
**Validates: Requirements 3.5**

Property 9: Hypothesis falsifiability guardrail
*For any* hypothesis submission with an empty predictions list, the backend SHALL reject it with an error response containing "falsifiability" in the message
**Validates: Requirements 4.3**

Property 10: Confidence is bounded after testing
*For any* hypothesis that has been tested, the confidence value SHALL be between 0.0 and 1.0 inclusive, and the status SHALL be one of the valid enum values
**Validates: Requirements 4.4**

Property 11: Residue search filter correctness
*For any* search with min_uncertainty=X, all returned residues SHALL have the specified uncertainty type >= X; for any search with chain=C, all returned residues SHALL have chain_label=C
**Validates: Requirements 5.2**

Property 12: Export round-trip
*For any* structure with embedding data, exporting as JSON and then parsing the resulting file SHALL produce a valid JSON object containing the structure_id and a residues array with length equal to the structure's residue count
**Validates: Requirements 5.4**

Property 13: Annotation persistence round-trip
*For any* annotation created via the POST endpoint, immediately querying the GET annotations endpoint for the same structure SHALL return an annotation with matching annotation_id, text, and type
**Validates: Requirements 3.6, 4.7, 5.5**

Property 14: Plot generation produces a file
*For any* valid structure with embedding data and valid plot_type, the generate endpoint SHALL return success=true and a file_path pointing to an existing PNG file with size > 0 bytes
**Validates: Requirements 6.2**

Property 15: Write operations return provenance metadata
*For any* successful write operation (hypothesis creation, evidence addition, annotation), the response SHALL contain a non-empty run_id or equivalent provenance identifier
**Validates: Requirements 8.1**

## Error Handling

| Scenario | Handling |
|----------|----------|
| RCSB unreachable | Return 502 with `{"error": "rcsb_unavailable", "message": "..."}` |
| Invalid sequence for similarity search | Return 422 with validation details |
| Structure has no graph data | Return 404 with `{"error": "no_graph_data", "message": "Run pipeline first"}` |
| Shortest path between disconnected residues | Return 200 with `disconnected: true` and empty path |
| Hypothesis without predictions | Return 422 with falsifiability error |
| Plot generation missing data | Return 404 with `{"error": "missing_data", "message": "..."}` |
| Export for structure with no embeddings | Return 404 with guidance on what to run |
| Normalizer write failure | Return 503 with error details and `retry: true` flag |
| DB connection lost during hydration | Return partial hydration with null fields and `db_connected: false` |

## Testing Strategy

**Property-Based Testing Library:** Hypothesis (Python) for backend

**Unit Tests:**
- Each new router endpoint tested with mock DB (QueryValidatingMockDB)
- Hydration endpoint tested with various data availability scenarios
- RCSB endpoints tested with mocked rcsbapi responses
- Hypothesis engine tested with various prediction/evidence combinations

**Property Tests:**
- Properties 1-15 implemented using Hypothesis library
- Each property generates random valid inputs and verifies the stated invariant
- Minimum 100 iterations per property test
- Tag format: **Feature: workbench-tool-surface, Property {number}: {property_text}**

**Integration Tests:**
- Full hydration flow with real DB (docker compose)
- Hypothesis lifecycle: propose → test → add evidence → evaluate
- Annotation round-trip with Normalizer verification
- Plot generation with real matplotlib rendering

**Configuration:**
- Property tests use `@settings(max_examples=100)`
- Each test tagged with design property reference
- Tests use the existing `QueryValidatingMockDB` fixture for unit tests
- Integration tests use `integration_db` fixture with transaction rollback
