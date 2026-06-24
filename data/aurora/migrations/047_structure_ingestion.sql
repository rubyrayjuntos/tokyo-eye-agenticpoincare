-- =============================================================================
-- Migration 047: Structure Ingestion Schema Extensions
-- Date: 2026-06-22
-- Purpose: Extend core dimensions for full BinaryCIF ingest support.
--          Add computation scope, residue alignment, structural alignment,
--          covalent bond, and normalization protocol tables.
-- Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8
-- =============================================================================

-- ---------------------------------------------------------------------------
-- dim_structure extensions (Requirement 10.3)
-- ---------------------------------------------------------------------------
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS organism TEXT;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS release_date DATE;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS polymer_composition TEXT;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS r_factor DOUBLE PRECISION;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS r_free DOUBLE PRECISION;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS model_count INTEGER DEFAULT 1;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS assembly_id TEXT;

-- ---------------------------------------------------------------------------
-- dim_chain extensions (Requirement 10.1)
-- ---------------------------------------------------------------------------
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS label_asym_id TEXT;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS entity_id TEXT;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS sequence_length INTEGER;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS uniprot_accession TEXT;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS uniprot_start INTEGER;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS uniprot_end INTEGER;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS is_entity_duplicate BOOLEAN DEFAULT FALSE;
ALTER TABLE dim_chain ADD COLUMN IF NOT EXISTS is_representative BOOLEAN DEFAULT TRUE;

-- ---------------------------------------------------------------------------
-- dim_residue extensions (Requirement 10.2)
-- ---------------------------------------------------------------------------
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS label_seq_id INTEGER;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS insertion_code TEXT;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS comp_id TEXT;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS parent_comp_id TEXT;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS is_resolved BOOLEAN DEFAULT TRUE;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS is_modified BOOLEAN DEFAULT FALSE;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS max_b_factor DOUBLE PRECISION;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS low_confidence_coords BOOLEAN DEFAULT FALSE;
ALTER TABLE dim_residue ADD COLUMN IF NOT EXISTS partial_backbone BOOLEAN DEFAULT FALSE;

-- ---------------------------------------------------------------------------
-- dim_atom extensions
-- ---------------------------------------------------------------------------
ALTER TABLE dim_atom ADD COLUMN IF NOT EXISTS altloc TEXT;
ALTER TABLE dim_atom ADD COLUMN IF NOT EXISTS is_hetero BOOLEAN DEFAULT FALSE;
ALTER TABLE dim_atom ADD COLUMN IF NOT EXISTS model_id INTEGER DEFAULT 1;

-- ---------------------------------------------------------------------------
-- structure_computation_scope (Requirement 10.4)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS structure_computation_scope (
    structure_id        TEXT PRIMARY KEY REFERENCES dim_structure(structure_id),
    primary_chain_ids   TEXT[] NOT NULL,
    reference_chain     TEXT NOT NULL,
    exclude_chain_ids   TEXT[] DEFAULT '{}',
    scope_source        TEXT NOT NULL DEFAULT 'auto',
    selection_reason    TEXT,
    normalization_protocol TEXT NOT NULL DEFAULT 'graph_default',
    quality_filters     JSONB,
    model_index         INTEGER DEFAULT 1,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- fact_residue_alignment (Requirement 10.5)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_residue_alignment (
    id                  BIGSERIAL PRIMARY KEY,
    residue_id          TEXT NOT NULL REFERENCES dim_residue(residue_id),
    uniprot_accession   TEXT NOT NULL,
    uniprot_position    INTEGER,
    isoform_id          TEXT,
    mapping_source      TEXT NOT NULL DEFAULT 'sifts',
    mapping_confidence  DOUBLE PRECISION NOT NULL,
    reason_code         TEXT,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (residue_id, uniprot_accession, run_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_residue_alignment_uniprot
    ON fact_residue_alignment(uniprot_accession, uniprot_position);
CREATE INDEX IF NOT EXISTS idx_fact_residue_alignment_residue
    ON fact_residue_alignment(residue_id);

-- ---------------------------------------------------------------------------
-- fact_structural_alignment (Requirement 10.6)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_structural_alignment (
    id                      BIGSERIAL PRIMARY KEY,
    query_structure_id      TEXT NOT NULL REFERENCES dim_structure(structure_id),
    reference_structure_id  TEXT NOT NULL REFERENCES dim_structure(structure_id),
    uniprot_accession       TEXT NOT NULL,
    rotation_matrix         DOUBLE PRECISION[9] NOT NULL,
    translation             DOUBLE PRECISION[3] NOT NULL,
    rmsd                    DOUBLE PRECISION NOT NULL,
    aligned_residue_count   INTEGER NOT NULL,
    comparable_core         JSONB NOT NULL,
    protocol_version        TEXT NOT NULL,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (query_structure_id, reference_structure_id, uniprot_accession, run_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_structural_alignment_query
    ON fact_structural_alignment(query_structure_id);
CREATE INDEX IF NOT EXISTS idx_fact_structural_alignment_ref
    ON fact_structural_alignment(reference_structure_id);

-- ---------------------------------------------------------------------------
-- fact_covalent_bond (Requirement 10.7)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_covalent_bond (
    id              BIGSERIAL PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id_1    TEXT NOT NULL REFERENCES dim_residue(residue_id),
    residue_id_2    TEXT NOT NULL REFERENCES dim_residue(residue_id),
    atom_name_1     TEXT NOT NULL,
    atom_name_2     TEXT NOT NULL,
    bond_type       TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES provenance_run(run_id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (structure_id, residue_id_1, residue_id_2, bond_type)
);

-- ---------------------------------------------------------------------------
-- normalization_protocol (Requirement 10.8)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS normalization_protocol (
    protocol_name   TEXT PRIMARY KEY,
    version         INTEGER NOT NULL DEFAULT 1,
    parameters      JSONB NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Seed 4 default protocols
INSERT INTO normalization_protocol (protocol_name, version, parameters, description)
VALUES
    ('graph_default', 1,
     '{"atoms":"protein_only","altloc":"highest_occupancy","modified_residues":"harmonize_to_parent","unresolved":"exclude","projection":"ca_only","partial_backbone":"exclude"}',
     'Default for GNN graph building'),
    ('family_compare', 1,
     '{"residues":"sifts_mapped_only","intersection":"uniprot_positions","superposition":"apply_reference","tags_tails":"exclude","mutations":"retain"}',
     'For cross-structure family comparison'),
    ('binding_site', 1,
     '{"ligands":"within_cutoff","cutoff_angstrom":5.0,"waters":"exclude_unless_bridging","frame":"local_pocket","protonation":"untouched"}',
     'For binding site analysis'),
    ('interface', 1,
     '{"chains":"multi_chain","assembly":"biological","entity_collapse":false}',
     'For protein-protein interface analysis')
ON CONFLICT (protocol_name) DO NOTHING;
