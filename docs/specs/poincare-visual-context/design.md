# Design Document: Poincaré Visual Context

## Overview

This feature gives the agent dual-mode perceptual awareness of the Poincaré disc: a structured topological interpretation always present in context (clusters, neighborhoods, hubs, density), and an on-demand visual snapshot that lets the LLM see exactly what the user sees. Together, these close the perception gap between user and agent — the user looks at the disc and sees patterns; the agent can now reason about those same patterns.

The structured interpretation is computed server-side from the persisted `hyp_projection_2d` VECTOR(2) coordinates using true Poincaré distance and HDBSCAN clustering. Results are cached per (structure_id, run_id) pair. The visual snapshot is a 512×512 PNG captured from the frontend canvas on demand and passed as a multimodal content block.

## Architecture

```mermaid
flowchart TB
    subgraph "Database"
        EMB[fact_gnn_node_embedding<br/>hyp_projection_2d VECTOR(2)]
    end

    subgraph "Backend (agent/tools/)"
        TOPO[disc_topology_tool<br/>HDBSCAN + k-NN + sectors]
        NEIGH[disc_neighborhood_tool<br/>k-nearest on disc]
        CACHE[TopologyCache<br/>per structure_id + run_id]
        TOPO --> CACHE
        NEIGH --> CACHE
        EMB --> TOPO
        EMB --> NEIGH
    end

    subgraph "Orchestration"
        CTX[Context Builder<br/>includes Disc_Summary]
        ORCH[SessionOrchestrator]
        CTX --> ORCH
        CACHE --> CTX
    end

    subgraph "Frontend"
        CANVAS[Three.js Poincaré Canvas]
        SNAP[Snapshot Capture<br/>canvas.toDataURL()]
        CANVAS --> SNAP
    end

    subgraph "Chat Flow"
        REQ[Chat Request<br/>+ optional image base64]
        LLM[LLM Provider<br/>multimodal message]
        SNAP -->|on-demand| REQ
        CTX -->|always| LLM
        REQ -->|if image present + vision supported| LLM
    end
```

## Components and Interfaces

### 1. Disc Topology Computer

Core computation module that transforms raw VECTOR(2) coordinates into structured spatial awareness.

```python
# agent/tools/disc_topology.py

@dataclass
class DiscCluster:
    cluster_id: int
    residue_ids: list[str]
    centroid_angle_deg: float    # 0-360, angular position on disc
    centroid_radius: float       # 0-1, distance from center
    hub_residue_id: str          # highest disc-degree in cluster
    size: int

@dataclass
class DiscTopologyResult:
    structure_id: str
    run_id: str
    curvature_c: float
    total_residues: int
    cluster_count: int
    clusters: list[DiscCluster]
    bridge_residues: list[str]   # residues connecting clusters
    peripheral_residues: list[str]  # isolated at disc edge
    radial_density: dict[str, int]  # "core"/"mid"/"periphery" counts

@dataclass
class DiscNeighborhood:
    target_residue_id: str
    target_cluster_id: int | None
    is_hub: bool
    is_peripheral: bool
    neighbors: list[NeighborInfo]

@dataclass
class NeighborInfo:
    residue_id: str
    hyperbolic_distance: float
    cluster_id: int | None
    cone_depth: float
    epistemic_uncertainty: float


def compute_disc_topology(
    coordinates: list[tuple[str, float, float]],  # (residue_id, x, y)
    curvature_c: float = 1.0,
    min_cluster_size: int = 5,
) -> DiscTopologyResult:
    """Compute full disc topology from Poincaré coordinates.

    Uses true Poincaré distance for the distance matrix,
    then HDBSCAN for natural cluster discovery.
    """
    ...

def compute_disc_neighborhood(
    target_residue_id: str,
    coordinates: list[tuple[str, float, float]],
    topology: DiscTopologyResult,
    curvature_c: float = 1.0,
    k: int = 8,
) -> DiscNeighborhood:
    """Get k-nearest neighbors on the disc for a specific residue."""
    ...
```

### 2. Poincaré Distance Function

True hyperbolic distance on the Poincaré disc model:

```python
def poincare_distance(p: tuple[float, float], q: tuple[float, float], c: float = 1.0) -> float:
    """Compute geodesic distance on the Poincaré disc.

    Formula: d(p, q) = (1/√c) * arcosh(1 + 2c * ||p-q||² / ((1 - c*||p||²)(1 - c*||q||²)))

    Args:
        p, q: Points on the disc (x, y) with ||p|| < 1/√c
        c: Curvature parameter (default 1.0)
    """
    ...
```

### 3. Topology Cache

```python
class TopologyCache:
    """LRU cache for computed disc topologies, keyed by (structure_id, run_id)."""

    def __init__(self, max_size: int = 50):
        self._cache: dict[tuple[str, str], DiscTopologyResult] = {}

    def get(self, structure_id: str, run_id: str) -> DiscTopologyResult | None: ...
    def put(self, result: DiscTopologyResult) -> None: ...
    def invalidate(self, structure_id: str) -> None: ...
```

### 4. Disc Summary Formatter

Converts `DiscTopologyResult` into a compact text block for context injection:

```python
def format_disc_summary(topology: DiscTopologyResult) -> str:
    """Format disc topology into a compact context block.

    Output format (bounded to ~150 tokens):
    [Disc Topology] 4 clusters, 187 residues
    Cluster 0 (45°, core, 52 residues, hub: A:G12) — P-loop domain
    Cluster 1 (180°, mid, 41 residues, hub: A:T35) — Switch-I
    Cluster 2 (270°, periphery, 38 residues, hub: A:D57) — Switch-II
    Cluster 3 (315°, mid, 29 residues, hub: A:Y71) — α3-helix
    Bridges: A:E31, A:D33 | Peripheral: A:K147, A:R149
    """
    ...
```

### 5. Frontend Snapshot Capture

```typescript
// visualizer/frontend/src/lib/snapshotCapture.ts

export async function capturePoincareSnapshot(
  canvas: HTMLCanvasElement,
  maxSize: number = 512
): Promise<string> {
  // Resize if needed, export as base64 PNG
  // Returns: "data:image/png;base64,..."
}
```

Triggered by:
- User clicks "Show agent" button
- Agent requests snapshot via WebSocket `request_snapshot` message
- Chat component detects user's question references visual patterns

### 6. Chat Request Extension

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str
    viewport_state: ViewportState | None = None
    poincare_snapshot: str | None = None  # base64 PNG, max 512x512
```

### 7. LLM Message Construction

```python
def build_llm_messages(
    context_block: str,
    user_message: str,
    poincare_snapshot: str | None = None,
    provider_supports_vision: bool = True,
) -> list[dict]:
    """Build multimodal message array for LLM.

    If snapshot present and vision supported:
    - System message with context block
    - User message with text + image content blocks
    """
    ...
```

## Data Models

### Disc Topology Result (Pydantic)

```python
class DiscClusterModel(BaseModel):
    cluster_id: int
    residue_ids: list[str]
    centroid_angle_deg: float
    centroid_radius: float
    hub_residue_id: str
    size: int
    angular_sector: str  # "NE", "E", "SE", "S", "SW", "W", "NW", "N"

class DiscTopologyModel(BaseModel):
    structure_id: str
    run_id: str
    curvature_c: float
    total_residues: int
    cluster_count: int
    clusters: list[DiscClusterModel]
    bridge_residues: list[str]
    peripheral_residues: list[str]
    radial_density: dict[str, int]  # "core", "mid", "periphery"
    computed_at: str

class DiscNeighborhoodModel(BaseModel):
    target_residue_id: str
    target_cluster_id: int | None
    is_hub: bool
    is_peripheral: bool
    neighbors: list[NeighborInfoModel]

class NeighborInfoModel(BaseModel):
    residue_id: str
    hyperbolic_distance: float
    cluster_id: int | None
    cone_depth: float
    epistemic_uncertainty: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Topology completeness

*For any* set of valid 2D Poincaré disc coordinates (all within the disc boundary), the computed topology result SHALL contain: a cluster assignment for every input residue, at least one cluster, centroid angle and radius for each cluster, a hub residue for each cluster, and counts matching the input size.

**Validates: Requirements 1.1, 1.2**

### Property 2: Disc summary content completeness

*For any* valid DiscTopologyResult, the formatted Disc_Summary text SHALL contain: the cluster count, angular position description for each cluster, residue count per cluster, and bridge residue identifiers.

**Validates: Requirements 1.3, 1.4**

### Property 3: Disc summary size bound

*For any* DiscTopologyResult (including pathological cases with 20+ clusters and 500+ residues), the formatted Disc_Summary SHALL not exceed 150 tokens (approximately 600 characters).

**Validates: Requirements 1.5**

### Property 4: Neighborhood query completeness

*For any* residue in a set of disc embeddings with at least k+1 residues, the neighborhood query SHALL return exactly k neighbors, each with a valid hyperbolic distance > 0, cluster membership, cone_depth, and epistemic_uncertainty values.

**Validates: Requirements 2.1, 2.2**

### Property 5: Topology annotation correctness

*For any* residue that qualifies as a hub (highest disc-degree in its cluster) or peripheral (radial distance > threshold), the neighborhood result SHALL include the appropriate annotation (is_hub=True or is_peripheral=True) with the correct cluster or nearest-cluster reference.

**Validates: Requirements 2.3, 2.4**

### Property 6: Visual snapshot conditional inclusion

*For any* chat request with a non-null poincare_snapshot field, the LLM message SHALL include an image content block if and only if the provider supports vision. When the provider does not support vision, the image SHALL be omitted and no error SHALL be raised.

**Validates: Requirements 3.2, 3.5**

### Property 7: Poincaré distance correctness

*For any* two points on the Poincaré disc (within the boundary defined by curvature c), the computed distance SHALL satisfy: d(p, p) = 0, d(p, q) = d(q, p), d(p, q) ≥ 0, and the triangle inequality d(p, r) ≤ d(p, q) + d(q, r).

**Validates: Requirements 4.1**

### Property 8: Curvature parameterization

*For any* pair of disc points and two different curvature values c₁ ≠ c₂, the Poincaré distances computed with c₁ and c₂ SHALL differ (demonstrating that curvature affects computation).

**Validates: Requirements 4.3**

### Property 9: Topology cache idempotence

*For any* (structure_id, run_id) pair, computing disc topology twice SHALL return identical results, and the second call SHALL be served from cache (not recomputed).

**Validates: Requirements 4.4**

### Property 10: Summary freshness on structure change

*For any* orchestrator with a loaded structure and computed disc summary, changing the active structure SHALL cause the next context block to reflect the new structure's topology (not the stale cached data from the previous structure).

**Validates: Requirements 5.4**

## Error Handling

- **Empty embeddings**: If no `hyp_projection_2d` data exists for a structure, return an empty topology with cluster_count=0 and skip the Disc_Summary in context
- **Degenerate clustering**: If HDBSCAN assigns all points to noise (cluster=-1), treat the entire set as one cluster centered at the mean position
- **Points outside disc boundary**: Clamp to disc boundary (||p|| < 1/√c) before computing distances; log a warning
- **Snapshot too large**: If base64 image exceeds 1MB, reject with error and ask frontend to reduce resolution
- **Vision not supported**: Gracefully skip image content block; the structured summary provides fallback
- **Cache invalidation race**: On GNN run completion, invalidate cache before any new chat requests can read stale data

## Testing Strategy

### Property-Based Testing

Use **Hypothesis** (Python) for all property tests. Each property maps to one `@given` test function.

Configuration:
- Minimum 100 examples per property test
- Use `@settings(max_examples=200)` for distance and clustering properties
- Tag format: `# Feature: poincare-visual-context, Property N: <title>`

Library: `hypothesis` with custom strategies for generating valid Poincaré disc point sets.

### Unit Tests

Unit tests cover:
- HDBSCAN producing expected clusters on known geometric arrangements
- Poincaré distance against known analytical values (e.g., distance from origin)
- Snapshot base64 validation and size checking
- Context builder integration with disc summary
- Cache hit/miss/invalidation scenarios

### Test Organization

```
tests/
├── test_disc_topology_properties.py    # Properties 1-5, 7-10
├── test_disc_topology_unit.py          # Edge cases, known values
├── test_visual_snapshot.py             # Property 6, integration
└── strategies/
    └── disc_strategies.py              # Generators for valid disc point sets
```

### Key Test Strategies

```python
@st.composite
def valid_disc_points(draw, min_points=10, max_points=200, curvature=1.0):
    """Generate a set of points strictly within the Poincaré disc boundary."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    max_radius = 1.0 / math.sqrt(curvature) - 0.01  # stay within boundary
    points = []
    for i in range(n):
        angle = draw(st.floats(min_value=0, max_value=2*math.pi))
        radius = draw(st.floats(min_value=0, max_value=max_radius))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((f"R{i}", x, y))
    return points, curvature
```
