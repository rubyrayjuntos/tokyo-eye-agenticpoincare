# Residue ID Key Strategy (v1.0 — LOCKED)

**Status:** Accepted  
**Date:** May 27, 2026  
**Related:** ADR-001, DIMENSIONAL_MODEL_DRAFT.md, Migration 001

---

## Decision

The `residue_id` primary key uses a **deterministic canonical format** derived from the structure's identity and the residue's position within it.

### Format

```
{structure_id}:{chain_label}:{residue_index}:{insertion_code}
```

- `structure_id` — the governing `dim_structure.structure_id` (itself a deterministic key, see below)
- `chain_label` — the author-assigned chain identifier (e.g., `A`, `B`)
- `residue_index` — the PDB residue sequence number (author numbering)
- `insertion_code` — PDB insertion code if present, otherwise omitted (trailing colon dropped)

### Examples

```
4OBE:A:12       — residue 12 on chain A of structure 4OBE
7XKJ:B:145:A   — residue 145 with insertion code A on chain B
af2_KRAS:A:61  — AlphaFold-derived structure, chain A, residue 61
```

### Structure ID Strategy

`structure_id` follows a similar deterministic pattern:

| Source | Format | Example |
|--------|--------|---------|
| RCSB PDB | lowercase PDB ID | `4obe` |
| AlphaFold | `af2_{uniprot_id}` | `af2_P01116` |
| User upload | `user_{sha256_prefix8}_{name}` | `user_a3f8c1d2_kras_mutant` |
| Computed/derived | `derived_{parent_id}_{suffix}` | `derived_4obe_minimized` |

### Chain ID Strategy

```
{structure_id}:{chain_label}
```

Example: `4obe:A`

---

## Rationale

### Why Canonical (Deterministic) Over Surrogate (UUID)

1. **Cross-run joins without lookup tables.** The same residue in the same structure always has the same ID regardless of which pipeline produced the data. This is critical for comparing v3 vs v4 outputs on the same protein.

2. **Human readability.** A researcher can look at `4OBE:A:12` and immediately know what it refers to. UUIDs require a lookup.

3. **Idempotent ingestion.** Re-processing the same structure produces the same dimension keys, enabling clean upsert semantics without deduplication logic.

4. **External reference stability.** The ID can be reconstructed from any PDB file or mmCIF without needing access to our database.

5. **Backfill simplicity.** Historical data can be assigned correct `residue_id` values without needing to query existing records.

### Why Not Pure PDB Numbering

PDB author numbering has known issues (gaps, insertion codes, inconsistencies across depositions). However:

- For our use case (single-structure analysis with GNN), author numbering is the natural coordinate system.
- The `residue_index` field in `dim_residue` stores the raw number; renumbering or alignment can be handled via computed properties or separate mapping tables.
- Insertion codes are included to handle the edge cases.

### Trade-offs Accepted

- If a structure is re-deposited with different chain labels, it gets a different `structure_id` and therefore different `residue_id` values. This is acceptable — they are scientifically different depositions.
- Very long `residue_id` strings (vs compact UUIDs). Acceptable given the readability and join benefits.
- Requires consistent key generation across all ingestion paths. Mitigated by a single canonical function (see below).

---

## Implementation

### Canonical Key Generation Function (Python)

```python
def make_residue_id(
    structure_id: str,
    chain_label: str,
    residue_index: int,
    insertion_code: str | None = None,
) -> str:
    """Generate the canonical residue_id.
    
    This function MUST be used by all code that creates or references
    residue dimension records. No other key format is acceptable.
    """
    base = f"{structure_id}:{chain_label}:{residue_index}"
    if insertion_code:
        return f"{base}:{insertion_code}"
    return base


def make_chain_id(structure_id: str, chain_label: str) -> str:
    """Generate the canonical chain_id."""
    return f"{structure_id}:{chain_label}"


def make_structure_id(
    pdb_id: str | None = None,
    source: str = "rcsb",
    uniprot_id: str | None = None,
    name: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Generate the canonical structure_id based on source."""
    if source == "rcsb" and pdb_id:
        return pdb_id.lower()
    elif source == "alphafold" and uniprot_id:
        return f"af2_{uniprot_id}"
    elif source == "user" and content_hash and name:
        return f"user_{content_hash[:8]}_{name}"
    elif source == "derived":
        raise ValueError("Derived structures need explicit parent + suffix")
    else:
        raise ValueError(f"Cannot generate structure_id for source={source}")
```

### SQL Helper (for use in migrations/backfill)

```sql
-- Canonical residue_id construction (for use in backfill queries)
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
```

---

## Validation Rules

1. All `residue_id` values MUST match the regex: `^[a-zA-Z0-9_]+:[A-Za-z0-9]+:\d+(:[A-Za-z])?$`
2. The `structure_id` prefix of a `residue_id` MUST correspond to an existing `dim_structure` record.
3. The Normalizer MUST reject any payload that does not use canonical key generation.
4. Backfill scripts MUST use the canonical functions above — no ad-hoc key construction.

---

## Migration Impact

Migration 001 already defines `residue_id TEXT PRIMARY KEY`. This strategy is compatible — no schema change needed. The constraint is on the *values* written, enforced by the Normalizer and ingestion code.

A new migration (026) will add the SQL helper function and a CHECK constraint pattern.

---

**This decision is LOCKED for Phase 1 execution.**
