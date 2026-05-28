# Feature Brief: Graph Topology Compare

## Goal

Give the agent tools to analyze and compare the molecular contact graph itself — not just GNN embeddings, but the underlying edges (H-bonds, contacts, covalent bonds) and per-node graph metrics. This enables WT/mutant comparison at the structural level: "which edges changed, which residues became more/less connected."

## Why This Matters

The GNN operates on a graph. Currently we only persist the *output* (embeddings, cone depth). But the *input graph* contains critical structural information:
- Lost H-bonds in a mutant → destabilized region
- Gained contacts → new allosteric coupling
- Betweenness centrality changes → altered signal propagation paths
- Bridge residues → single points of failure in the conformational network

## Data Model

### New migration: `031_graph_topology.sql`

```sql
CREATE TABLE fact_graph_edge (
    edge_id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    source_residue_id TEXT NOT NULL REFERENCES dim_residue(residue_id),
    target_residue_id TEXT NOT NULL REFERENCES dim_residue(residue_id),
    edge_type       TEXT NOT NULL,  -- 'h_bond', 'contact', 'covalent', 'disulfide', 'salt_bridge'
    distance_angstrom DOUBLE PRECISION,
    hyperbolic_distance DOUBLE PRECISION,
    weight          DOUBLE PRECISION DEFAULT 1.0,
    metadata        JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fact_graph_node_metrics (
    metric_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id      TEXT NOT NULL REFERENCES dim_residue(residue_id),
    degree          INTEGER,
    betweenness     DOUBLE PRECISION,
    clustering_coefficient DOUBLE PRECISION,
    closeness       DOUBLE PRECISION,
    eigenvector_centrality DOUBLE PRECISION,
    is_bridge       BOOLEAN DEFAULT FALSE,
    conductance     DOUBLE PRECISION,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_graph_edge_natural ON fact_graph_edge(run_id, source_residue_id, target_residue_id, edge_type);
CREATE UNIQUE INDEX uq_graph_metrics_natural ON fact_graph_node_metrics(run_id, residue_id);
```

## Tools to Implement

### `agent/tools/graph_tools.py`

1. **`get_graph_metrics(structure_id, residue_ids?, metric_type?)`**
   - Returns per-residue: degree, betweenness, clustering_coeff, conductance, is_bridge
   - Highlights high-betweenness residues in the viewer

2. **`compare_graphs(structure_id_a, structure_id_b)`**
   - Edge diff: gained edges, lost edges, changed weights
   - Node metric diff: Δbetweenness, Δdegree per residue
   - Highlights gained/lost H-bonds in the viewer

3. **`get_hbond_network(structure_id, residue_ids?)`**
   - Returns the H-bond subgraph (edges of type 'h_bond')
   - Optionally filtered to a region

4. **`find_graph_bridges(structure_id)`**
   - Residues whose removal disconnects the graph
   - Critical for identifying allosteric communication bottlenecks

5. **`get_shortest_paths(structure_id, source_residue_id, target_residue_id)`**
   - Shortest path in the contact graph between two residues
   - Useful for "how does signal propagate from site A to site B?"

## Integration Points

- `science/dtie/common/graph_builder.py` already builds the PyG graph — extend it to persist edges via the Normalizer
- Add a `normalize_graph_topology` path to `data/normalizer/core.py`
- Graph metrics computation (networkx) should run in `asyncio.to_thread`
- Register tools in `agent/llm/agents.py` under a new `GRAPH_TOOLS` list

## Compute Flow

```
Structure ingested → graph_builder builds contact graph → 
  edges persisted via Normalizer → 
  networkx computes metrics (betweenness, bridges, etc.) → 
  metrics persisted via Normalizer →
  agent can query via tools
```

## Testing

- Unit test: mock DB, verify edge diff logic
- Integration test: build graph for test structure, persist, query back
- Compare test: two structures with known edge differences
