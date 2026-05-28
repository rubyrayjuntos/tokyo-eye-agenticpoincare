-- ============================================================================
-- Migration 026: Canonical Key Generation Functions
-- Date: 2026-05-27
-- Purpose: SQL-side canonical key generation for use in backfill queries,
--          views, and data validation. Mirrors the Python implementation
--          in science/dtie/common/keys.py.
-- Related: data/RESIDUE_ID_KEY_STRATEGY.md
-- ============================================================================

-- Canonical residue_id construction
CREATE OR REPLACE FUNCTION canonical_residue_id(
    p_structure_id TEXT,
    p_chain_label TEXT,
    p_residue_index INTEGER,
    p_insertion_code TEXT DEFAULT NULL
) RETURNS TEXT AS $$
BEGIN
    IF p_insertion_code IS NOT NULL AND p_insertion_code != '' THEN
        RETURN p_structure_id || ':' || p_chain_label || ':' || p_residue_index || ':' || p_insertion_code;
    ELSE
        RETURN p_structure_id || ':' || p_chain_label || ':' || p_residue_index;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Canonical chain_id construction
CREATE OR REPLACE FUNCTION canonical_chain_id(
    p_structure_id TEXT,
    p_chain_label TEXT
) RETURNS TEXT AS $$
BEGIN
    RETURN p_structure_id || ':' || p_chain_label;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Validation: check if a residue_id matches canonical format
CREATE OR REPLACE FUNCTION is_valid_residue_id(p_residue_id TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN p_residue_id ~ '^[a-zA-Z0-9_]+:[A-Za-z0-9]+:\d+(:[A-Za-z])?$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Add CHECK constraint to dim_residue (validates format on insert)
-- Note: This uses a permissive pattern to avoid blocking legitimate edge cases
-- during early migration. Can be tightened later.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_residue_id_format'
    ) THEN
        ALTER TABLE dim_residue
            ADD CONSTRAINT chk_residue_id_format
            CHECK (residue_id ~ '^[a-zA-Z0-9_]+:[A-Za-z0-9]+:\d+(:[A-Za-z])?$');
    END IF;
END $$;

-- ============================================================================
-- NOTES
-- ============================================================================
-- - These functions are IMMUTABLE (safe for use in indexes and views).
-- - The Python implementation in science/dtie/common/keys.py is authoritative.
-- - These SQL functions exist for convenience in backfill and validation queries.
-- - The CHECK constraint on dim_residue enforces format at the database level.
