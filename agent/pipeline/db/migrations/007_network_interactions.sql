-- ============================================================================
-- Migration 007: Network Interaction Layer
-- Persists STRING/DepMap/Reactome/community data under governance model.
-- Bridges network-level Poincaré disc to GNN residue-level embeddings.
-- ============================================================================

-- ── External Data Source Registry ────────────────────────────────────────────
-- Governs all external data fetches with provenance and TTL.

CREATE TABLE IF NOT EXISTS dim_external_source (
    source_id       TEXT PRIMARY KEY,
    source_name     TEXT NOT NULL,
    source_version  TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL,
    expires_at      TIMESTAMPTZ,
    checksum        TEXT,
    record_count    INTEGER,
    parameters      JSONB
);

CREATE INDEX IF NOT EXISTS idx_external_source_name ON dim_external_source(source_name);

-- ── Protein-Protein Interactions (PROTEIN level) ─────────────────────────────
-- Source: STRING API
-- Correlates to: dim_structure (via gene_symbol), fact_dtie_run.gene_symbol

CREATE TABLE IF NOT EXISTS fact_protein_interaction (
    interaction_id  TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES dim_external_source(source_id),
    protein_a       TEXT NOT NULL,
    protein_b       TEXT NOT NULL,
    combined_score  INTEGER NOT NULL,
    experimental    INTEGER,
    coexpression    INTEGER,
    database_score  INTEGER,
    textmining      INTEGER,
    pathway_context TEXT,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, protein_a, protein_b)
);

CREATE INDEX IF NOT EXISTS idx_ppi_protein_a ON fact_protein_interaction(protein_a);
CREATE INDEX IF NOT EXISTS idx_ppi_protein_b ON fact_protein_interaction(protein_b);
CREATE INDEX IF NOT EXISTS idx_ppi_source    ON fact_protein_interaction(source_id);

-- ── Gene Essentiality (PROTEIN level) ────────────────────────────────────────
-- Source: DepMap CRISPR Gene Dependency
-- Correlates to: dim_structure (via gene_symbol), fact_dtie_run.gene_symbol

CREATE TABLE IF NOT EXISTS fact_gene_essentiality (
    essentiality_id     TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL REFERENCES dim_external_source(source_id),
    gene_symbol         TEXT NOT NULL,
    mean_dependency     DOUBLE PRECISION NOT NULL,
    n_cell_lines        INTEGER,
    percentile_rank     DOUBLE PRECISION,
    pathway_centrality  DOUBLE PRECISION,
    computed_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, gene_symbol)
);

CREATE INDEX IF NOT EXISTS idx_essentiality_gene ON fact_gene_essentiality(gene_symbol);

-- ── Community Detection Results (PROTEIN level) ──────────────────────────────
-- Source: Louvain on STRING graph
-- Correlates to: fact_protein_interaction (same proteins)

CREATE TABLE IF NOT EXISTS fact_community_detection (
    detection_id    TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES dim_external_source(source_id),
    algorithm       TEXT NOT NULL DEFAULT 'louvain',
    resolution      DOUBLE PRECISION DEFAULT 1.0,
    n_communities   INTEGER NOT NULL,
    modularity      DOUBLE PRECISION,
    input_nodes     INTEGER NOT NULL,
    input_edges     INTEGER NOT NULL,
    parameters      JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_community_membership (
    detection_id    TEXT NOT NULL REFERENCES fact_community_detection(detection_id),
    gene_symbol     TEXT NOT NULL,
    community_id    INTEGER NOT NULL,
    community_label TEXT,
    intra_degree    INTEGER,
    inter_degree    INTEGER,
    PRIMARY KEY (detection_id, gene_symbol)
);

CREATE INDEX IF NOT EXISTS idx_community_gene ON fact_community_membership(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_community_id   ON fact_community_membership(community_id);

-- ── Pathway Membership (PROTEIN level) ───────────────────────────────────────
-- Source: Reactome API
-- Correlates to: fact_protein_interaction, fact_community_membership

CREATE TABLE IF NOT EXISTS fact_pathway_membership (
    membership_id   TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES dim_external_source(source_id),
    gene_symbol     TEXT NOT NULL,
    pathway_id      TEXT NOT NULL,
    pathway_name    TEXT NOT NULL,
    evidence_code   TEXT,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, gene_symbol, pathway_id)
);

CREATE INDEX IF NOT EXISTS idx_pathway_gene ON fact_pathway_membership(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_pathway_name ON fact_pathway_membership(pathway_name);

-- ── Protein-Level GNN Aggregate (PROTEIN level, derived from RESIDUE level) ──
-- Materialized from fact_gnn_node_output per structure.
-- Bridges network disc to residue-level GNN embeddings.
-- Correlates to: fact_gnn_inference, fact_gnn_node_output, dim_structure,
--                fact_dtie_run, fact_dtie_doorway, fact_dehydron

CREATE TABLE IF NOT EXISTS fact_protein_gnn_summary (
    summary_id              TEXT PRIMARY KEY,
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    gnn_run_id              TEXT NOT NULL REFERENCES fact_gnn_inference(gnn_run_id),
    gene_symbol             TEXT,
    condition               TEXT NOT NULL,
    n_residues              INTEGER NOT NULL,
    mean_cone_depth         DOUBLE PRECISION NOT NULL,
    max_cone_depth          DOUBLE PRECISION NOT NULL,
    mean_cone_width         DOUBLE PRECISION NOT NULL,
    mean_epistemic          DOUBLE PRECISION NOT NULL,
    mean_aleatoric          DOUBLE PRECISION NOT NULL,
    mean_total_uncertainty  DOUBLE PRECISION NOT NULL,
    embedding_centroid      JSONB,
    n_doorways              INTEGER,
    n_dehydrons             INTEGER,
    vulnerability_score     DOUBLE PRECISION,
    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (gnn_run_id, condition)
);

CREATE INDEX IF NOT EXISTS idx_protein_gnn_gene      ON fact_protein_gnn_summary(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_protein_gnn_structure  ON fact_protein_gnn_summary(structure_id);
CREATE INDEX IF NOT EXISTS idx_protein_gnn_vuln       ON fact_protein_gnn_summary(vulnerability_score DESC NULLS LAST);

-- ── Views for the network↔GNN bridge query ───────────────────────────────────

CREATE OR REPLACE VIEW v_network_gnn_bridge AS
SELECT
    cm.gene_symbol,
    cm.community_id,
    cm.community_label,
    cm.intra_degree,
    cm.inter_degree,
    ge.mean_dependency      AS essentiality,
    ge.percentile_rank      AS essentiality_rank,
    pgs.mean_cone_depth,
    pgs.max_cone_depth,
    pgs.mean_total_uncertainty,
    pgs.vulnerability_score,
    pgs.n_doorways,
    pgs.n_dehydrons,
    pgs.n_residues,
    pgs.embedding_centroid,
    pgs.structure_id,
    pgs.condition
FROM fact_community_membership cm
LEFT JOIN fact_gene_essentiality ge
    ON ge.gene_symbol = cm.gene_symbol
LEFT JOIN fact_protein_gnn_summary pgs
    ON pgs.gene_symbol = cm.gene_symbol
WHERE cm.detection_id = (
    SELECT detection_id FROM fact_community_detection
    ORDER BY computed_at DESC LIMIT 1
);

CREATE OR REPLACE VIEW v_pathway_gnn_bridge AS
SELECT
    pm.gene_symbol,
    pm.pathway_name,
    pm.pathway_id,
    ge.mean_dependency      AS essentiality,
    pgs.mean_cone_depth,
    pgs.vulnerability_score,
    pgs.n_doorways,
    pgs.embedding_centroid,
    pgs.structure_id,
    pgs.condition
FROM fact_pathway_membership pm
LEFT JOIN fact_gene_essentiality ge
    ON ge.gene_symbol = pm.gene_symbol
LEFT JOIN fact_protein_gnn_summary pgs
    ON pgs.gene_symbol = pm.gene_symbol
WHERE pm.source_id = (
    SELECT source_id FROM dim_external_source
    WHERE source_name = 'Reactome'
    ORDER BY fetched_at DESC LIMIT 1
);
