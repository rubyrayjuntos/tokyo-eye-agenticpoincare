-- Migration 004: Structure Normalization Layer
-- Adds canonical residue identity, pairwise alignment, and multi-structure
-- alignment group tables for the normalization layer.
-- Idempotent: safe to run multiple times (IF NOT EXISTS / DO blocks).
--
-- Requirements: 5.2, 9.1

-- ============================================================================
-- CANONICAL RESIDUE IDENTITY
-- ============================================================================

-- Canonical residue identity mapping (UniProt-based)
CREATE TABLE IF NOT EXISTS dim_residue_canonical (
    canonical_id        TEXT PRIMARY KEY,       -- "P01116:12"
    uniprot_accession   TEXT NOT NULL,          -- "P01116"
    uniprot_position    INTEGER NOT NULL,       -- 12
    uniprot_residue     TEXT,                   -- "G" (from UniProt)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonical_uniprot ON dim_residue_canonical(uniprot_accession);

-- Per-structure residue → canonical mapping
CREATE TABLE IF NOT EXISTS bridge_residue_canonical (
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT NOT NULL REFERENCES dim_residue(residue_id),
    canonical_id        TEXT NOT NULL REFERENCES dim_residue_canonical(canonical_id),
    mapping_confidence  TEXT NOT NULL,          -- "exact", "substitution", "mutation", "gap_adjacent"
    residue_status      TEXT NOT NULL,          -- "resolved", "disordered", "truncated", "artifact"
    observed_residue    TEXT,                   -- actual AA in structure (may differ from UniProt)
    PRIMARY KEY (structure_id, residue_id)
);

CREATE INDEX IF NOT EXISTS idx_bridge_canonical ON bridge_residue_canonical(canonical_id);
CREATE INDEX IF NOT EXISTS idx_bridge_structure ON bridge_residue_canonical(structure_id);

-- ============================================================================
-- PAIRWISE STRUCTURE ALIGNMENT
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_structure_alignment (
    alignment_id        TEXT PRIMARY KEY,
    structure_a_id      TEXT NOT NULL REFERENCES dim_structure(structure_id),
    structure_b_id      TEXT NOT NULL REFERENCES dim_structure(structure_id),
    reference_id        TEXT NOT NULL,          -- which structure is the reference frame
    chain_a_label       TEXT NOT NULL,
    chain_b_label       TEXT NOT NULL,

    -- Alignment quality metrics
    rmsd                DOUBLE PRECISION NOT NULL,
    sequence_identity   DOUBLE PRECISION NOT NULL,
    coverage            DOUBLE PRECISION NOT NULL,
    n_matched_residues  INTEGER NOT NULL,
    n_common_core       INTEGER NOT NULL,
    quality_class       TEXT NOT NULL,          -- "excellent", "good", "acceptable", "poor"

    -- Transformation (mobile → reference)
    rotation_matrix     JSONB NOT NULL,         -- [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]
    translation_vector  JSONB NOT NULL,         -- [tx, ty, tz]

    -- Provenance
    alignment_method    TEXT NOT NULL,          -- "uniprot_guided", "seqres_fallback"
    scoring_matrix      TEXT NOT NULL DEFAULT 'BLOSUM62',
    uniprot_accession   TEXT,                  -- NULL if fallback was used
    fallback_reason     TEXT,                  -- reason if UniProt mapping failed
    is_verified         BOOLEAN NOT NULL DEFAULT FALSE,
    version             INTEGER NOT NULL DEFAULT 1,
    is_current          BOOLEAN NOT NULL DEFAULT TRUE,

    -- Metadata
    computed_at         TIMESTAMPTZ DEFAULT NOW(),
    parameters          JSONB                  -- scoring params, cutoffs, etc.
);

CREATE INDEX IF NOT EXISTS idx_alignment_pair ON fact_structure_alignment(structure_a_id, structure_b_id);
CREATE INDEX IF NOT EXISTS idx_alignment_current ON fact_structure_alignment(is_current) WHERE is_current = TRUE;

-- Per-residue alignment details (common core and deviation tracking)
CREATE TABLE IF NOT EXISTS fact_alignment_residue (
    alignment_id        TEXT NOT NULL REFERENCES fact_structure_alignment(alignment_id),
    canonical_id        TEXT NOT NULL,
    residue_a_id        TEXT,                  -- NULL if disordered in A
    residue_b_id        TEXT,                  -- NULL if disordered in B
    ca_distance         DOUBLE PRECISION,      -- after superposition (NULL if either missing)
    in_common_core      BOOLEAN NOT NULL,
    exclusion_reason    TEXT,                  -- NULL, "gap", "disordered_a", "disordered_b", "high_deviation", "artifact"
    conditional_disorder TEXT,                 -- NULL, "disordered_in_a", "disordered_in_b"
    PRIMARY KEY (alignment_id, canonical_id)
);

CREATE INDEX IF NOT EXISTS idx_align_residue_core ON fact_alignment_residue(alignment_id, in_common_core);

-- ============================================================================
-- MULTI-STRUCTURE ALIGNMENT GROUPS
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_alignment_group (
    group_id            TEXT PRIMARY KEY,
    reference_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    uniprot_accession   TEXT,
    n_structures        INTEGER NOT NULL,
    consensus_core_size INTEGER NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bridge_alignment_group_member (
    group_id            TEXT NOT NULL REFERENCES fact_alignment_group(group_id),
    alignment_id        TEXT NOT NULL REFERENCES fact_structure_alignment(alignment_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    PRIMARY KEY (group_id, structure_id)
);

-- ============================================================================
-- DOWNSTREAM FK COLUMNS (nullable for backward compatibility)
-- ============================================================================

DO $$ BEGIN
    ALTER TABLE fact_gnn_inference
        ADD COLUMN IF NOT EXISTS alignment_id TEXT REFERENCES fact_structure_alignment(alignment_id);
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE fact_dtie_run
        ADD COLUMN IF NOT EXISTS alignment_id TEXT REFERENCES fact_structure_alignment(alignment_id);
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
