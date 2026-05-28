-- GNN vector embedding storage for AlloyDB/pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS fact_gnn_node_embedding (
    embedding_id         TEXT PRIMARY KEY,
    node_id              TEXT NOT NULL UNIQUE REFERENCES fact_gnn_node_output(node_id) ON DELETE CASCADE,
    gnn_run_id           TEXT NOT NULL REFERENCES fact_gnn_inference(gnn_run_id) ON DELETE CASCADE,
    structure_id         TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id           TEXT NOT NULL REFERENCES dim_residue(residue_id),
    model_version        TEXT NOT NULL,
    projection_dim       INTEGER NOT NULL,
    projection_embedding VECTOR NOT NULL,
    cone_depth           DOUBLE PRECISION,
    cone_width           DOUBLE PRECISION,
    routing_entropy      DOUBLE PRECISION,
    expert_winner        INTEGER,
    source_type          TEXT NOT NULL DEFAULT 'probabilistic',
    computed_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gnn_embedding_run ON fact_gnn_node_embedding(gnn_run_id);
CREATE INDEX IF NOT EXISTS idx_gnn_embedding_structure ON fact_gnn_node_embedding(structure_id);
CREATE INDEX IF NOT EXISTS idx_gnn_embedding_residue ON fact_gnn_node_embedding(residue_id);