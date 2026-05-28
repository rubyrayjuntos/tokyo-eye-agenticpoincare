-- ============================================================================
-- Migration 006: DTIE Database Persistence
-- Complete persistence layer for all DTIE pipeline phases.
-- Requirements: 1.1, 2.1, 3.1, 3.2, 4.2, 5.1, 5.3, 5.4, 5.6, 5.7, 5.8, 7.1, 7.2, 7.3
-- ============================================================================

-- Ensure pgvector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- FACT TABLE: Per-Residue Ingestion Features
-- Stores all computed features from PDB ingestion (rho, SASA, coords, etc.)
-- Requirements: 1.1, 1.2, 1.3
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_ingestion_features (
    feature_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    structure_id    TEXT NOT NULL,
    condition       TEXT NOT NULL,  -- 'gdp' or 'gtp'
    residue_id      TEXT NOT NULL,
    residue_index   INTEGER NOT NULL,
    ca_x            DOUBLE PRECISION,
    ca_y            DOUBLE PRECISION,
    ca_z            DOUBLE PRECISION,
    no_mid_x        DOUBLE PRECISION,
    no_mid_y        DOUBLE PRECISION,
    no_mid_z        DOUBLE PRECISION,
    rho             DOUBLE PRECISION,
    tau_flag        DOUBLE PRECISION,
    sasa            DOUBLE PRECISION,
    ss_type         DOUBLE PRECISION,
    resname         TEXT,
    volume          DOUBLE PRECISION,
    hydrophobicity  DOUBLE PRECISION,
    charge          DOUBLE PRECISION,
    b_factor        DOUBLE PRECISION,
    packing_density DOUBLE PRECISION,
    aa_index        INTEGER,
    version         INTEGER NOT NULL DEFAULT 1,
    is_current      BOOLEAN NOT NULL DEFAULT TRUE,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_features_run
    ON fact_ingestion_features(run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_features_structure
    ON fact_ingestion_features(structure_id, condition);
CREATE INDEX IF NOT EXISTS idx_ingestion_features_current
    ON fact_ingestion_features(structure_id, condition, is_current)
    WHERE is_current = TRUE;

-- ============================================================================
-- FACT TABLE: Graph Edges (Cα radius graph)
-- Stores the protein contact graph edges with spatial attributes
-- Requirements: 3.1, 3.3
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_graph_edge (
    edge_id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL,
    structure_id        TEXT NOT NULL,
    condition           TEXT NOT NULL,
    source_residue_id   TEXT NOT NULL,
    target_residue_id   TEXT NOT NULL,
    source_index        INTEGER NOT NULL,
    target_index        INTEGER NOT NULL,
    rel_x               DOUBLE PRECISION,
    rel_y               DOUBLE PRECISION,
    rel_z               DOUBLE PRECISION,
    distance            DOUBLE PRECISION,
    conductance         DOUBLE PRECISION,  -- NULL until Phase 4b populates it
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_graph_edge_structure
    ON fact_graph_edge(structure_id, condition);
CREATE INDEX IF NOT EXISTS idx_graph_edge_run
    ON fact_graph_edge(run_id);
CREATE INDEX IF NOT EXISTS idx_graph_edge_source
    ON fact_graph_edge(structure_id, condition, source_index);

-- ============================================================================
-- FACT TABLE: Phase 1 Output (Witness Embedding)
-- Stores witness coordinates, landmarks, curvature
-- Requirements: 5.1
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase1_output (
    phase1_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    n_witnesses     INTEGER,
    n_landmarks     INTEGER,
    curvature_c     DOUBLE PRECISION,
    mapping_function TEXT,
    witnesses_json  JSONB,
    landmarks_json  JSONB,
    landmark_to_residue_map JSONB,
    landmark_indices JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase1_output_run
    ON fact_phase1_output(run_id);

-- ============================================================================
-- FACT TABLE: Phase 3 Output (Persistence Barcodes)
-- Stores topological persistence results
-- Requirements: 5.3
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase3_output (
    phase3_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    n_terminal_leaks INTEGER,
    barcode_length  INTEGER,
    max_alpha_used  DOUBLE PRECISION,
    barcode_json    JSONB,
    terminal_leaks_json JSONB,
    h1_edges_json   JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase3_output_run
    ON fact_phase3_output(run_id);

-- ============================================================================
-- FACT TABLE: Phase 3.5 Lifted Sites
-- Stores topologically lifted sites with barycenter coordinates
-- Requirements: 5.4
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase35_lifted_site (
    site_id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    site_index      INTEGER,
    barycenter_x    DOUBLE PRECISION,
    barycenter_y    DOUBLE PRECISION,
    barycenter_z    DOUBLE PRECISION,
    birth           DOUBLE PRECISION,
    num_vertices    INTEGER,
    cycle_vertices  JSONB,
    generator_simplices JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase35_lifted_site_run
    ON fact_phase35_lifted_site(run_id);

-- ============================================================================
-- FACT TABLE: Phase 4b Output (Flux Betweenness)
-- Stores spectral graph analysis results
-- Requirements: 5.6
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase4b_output (
    phase4b_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    lambda_2_gdp    DOUBLE PRECISION,
    lambda_2_gtp    DOUBLE PRECISION,
    density_gdp     DOUBLE PRECISION,
    density_gtp     DOUBLE PRECISION,
    gdp_hubs_json   JSONB,
    gtp_hubs_json   JSONB,
    delta_hubs_json JSONB,
    commute_pairs_json JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase4b_output_run
    ON fact_phase4b_output(run_id);

-- ============================================================================
-- FACT TABLE: Phase 5 Pharmacophores
-- Stores pharmacophore models with spatial and chemical properties
-- Requirements: 5.7
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase5_pharmacophore (
    pharmacophore_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    pharm_index     INTEGER,
    center_x        DOUBLE PRECISION,
    center_y        DOUBLE PRECISION,
    center_z        DOUBLE PRECISION,
    druggability_score DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    pocket_source   TEXT,
    allosteric_coupling DOUBLE PRECISION,
    action          TEXT,
    atom_type_constraints JSONB,
    hbond_donors    JSONB,
    hbond_acceptors JSONB,
    charge_complementarity JSONB,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase5_pharmacophore_run
    ON fact_phase5_pharmacophore(run_id);

-- ============================================================================
-- FACT TABLE: Phase 6 Candidates (Screening/Docking/ADMET/Selectivity)
-- Stores drug candidates from phases 6a through 6d
-- Requirements: 5.8
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase6_candidate (
    candidate_id    TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id          TEXT NOT NULL,
    smiles          TEXT,
    phase6a_score   DOUBLE PRECISION,
    delta_g_gdp     DOUBLE PRECISION,
    delta_g_gtp     DOUBLE PRECISION,
    selectivity_ratio DOUBLE PRECISION,
    admet_passed    BOOLEAN,
    admet_fail_reasons JSONB,
    pharmacophore_idx INTEGER,
    docking_method  TEXT,
    phase           TEXT,  -- 'phase6a', 'phase6b', 'phase6c', 'phase6d'
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase6_candidate_run
    ON fact_phase6_candidate(run_id);
CREATE INDEX IF NOT EXISTS idx_phase6_candidate_phase
    ON fact_phase6_candidate(run_id, phase);


