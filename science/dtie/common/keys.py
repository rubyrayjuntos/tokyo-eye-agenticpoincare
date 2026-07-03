# Migrated from: new (Phase 1 implementation) on 2026-05-27
"""Canonical key generation for the Tokyo Eye governed data model.

This module is the SINGLE SOURCE OF TRUTH for generating dimension keys
(structure_id, chain_id, residue_id). All code that creates or references
dimension records MUST use these functions.

See: data/RESIDUE_ID_KEY_STRATEGY.md for the full rationale.
"""

from __future__ import annotations

import re

# Validation patterns
_STRUCTURE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_]+$")
_CHAIN_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
_RESIDUE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+:[A-Za-z0-9]+:\d+(:[A-Za-z])?$")


def make_structure_id(
    *,
    pdb_id: str | None = None,
    source: str = "rcsb",
    uniprot_id: str | None = None,
    name: str | None = None,
    content_hash: str | None = None,
    parent_id: str | None = None,
    suffix: str | None = None,
) -> str:
    """Generate the canonical structure_id based on source type.

    Args:
        pdb_id: PDB identifier (for RCSB source).
        source: One of 'rcsb', 'alphafold', 'user', 'derived'.
        uniprot_id: UniProt accession (for AlphaFold source).
        name: Human-readable name (for user uploads).
        content_hash: SHA256 hash of structure content (for user uploads).
        parent_id: Parent structure_id (for derived structures).
        suffix: Descriptive suffix (for derived structures).

    Returns:
        Canonical structure_id string.

    Raises:
        ValueError: If required fields for the given source are missing.
    """
    if source == "rcsb":
        if not pdb_id:
            raise ValueError("pdb_id required for RCSB source")
        result = pdb_id.lower().strip()
    elif source == "alphafold":
        if not uniprot_id:
            raise ValueError("uniprot_id required for AlphaFold source")
        result = f"af2_{uniprot_id}"
    elif source == "user":
        if not content_hash or not name:
            raise ValueError("content_hash and name required for user source")
        safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
        result = f"user_{content_hash[:8]}_{safe_name}"
    elif source == "derived":
        if not parent_id or not suffix:
            raise ValueError("parent_id and suffix required for derived source")
        safe_suffix = re.sub(r"[^a-z0-9_]", "_", suffix.lower().strip())
        # Flatten nested derivations: strip any existing "derived_" prefix from parent_id
        # so we don't get derived_derived_derived_... nesting.
        flat_parent = re.sub(r"^derived_", "", parent_id)
        result = f"derived_{flat_parent}_{safe_suffix}"
    else:
        raise ValueError(f"Unknown source type: {source}")

    if not _STRUCTURE_ID_PATTERN.match(result):
        raise ValueError(f"Generated structure_id '{result}' does not match canonical pattern")

    return result


def make_chain_id(structure_id: str, chain_label: str) -> str:
    """Generate the canonical chain_id.

    Args:
        structure_id: The parent structure's canonical ID.
        chain_label: Author-assigned chain identifier (e.g., 'A', 'B').

    Returns:
        Canonical chain_id string.
    """
    if not _CHAIN_LABEL_PATTERN.match(chain_label):
        raise ValueError(f"Invalid chain_label: '{chain_label}'")
    return f"{structure_id}:{chain_label}"


def make_residue_id(
    structure_id: str,
    chain_label: str,
    residue_index: int,
    insertion_code: str | None = None,
) -> str:
    """Generate the canonical residue_id.

    This function MUST be used by all code that creates or references
    residue dimension records. No other key format is acceptable.

    Args:
        structure_id: The parent structure's canonical ID.
        chain_label: Author-assigned chain identifier.
        residue_index: PDB residue sequence number (author numbering).
        insertion_code: PDB insertion code if present, otherwise None.

    Returns:
        Canonical residue_id string.
    """
    base = f"{structure_id}:{chain_label}:{residue_index}"
    if insertion_code:
        if len(insertion_code) != 1 or not insertion_code.isalpha():
            raise ValueError(f"Invalid insertion_code: '{insertion_code}'")
        return f"{base}:{insertion_code}"
    return base


def make_atom_id(
    structure_id: str,
    chain_label: str,
    residue_index: int,
    atom_name: str,
    insertion_code: str | None = None,
    alt_loc: str | None = None,
) -> str:
    """Generate the canonical atom_id.

    Args:
        structure_id: The parent structure's canonical ID.
        chain_label: Author-assigned chain identifier.
        residue_index: PDB residue sequence number.
        atom_name: Atom name (e.g., 'CA', 'N', 'CB').
        insertion_code: PDB insertion code if present.
        alt_loc: Alternate location indicator if present.

    Returns:
        Canonical atom_id string.
    """
    residue_id = make_residue_id(structure_id, chain_label, residue_index, insertion_code)
    atom_part = atom_name.strip()
    if alt_loc:
        return f"{residue_id}:{atom_part}:{alt_loc}"
    return f"{residue_id}:{atom_part}"


def make_site_id(structure_id: str, site_type: str, index: int) -> str:
    """Generate the canonical site_id.

    Args:
        structure_id: The parent structure's canonical ID.
        site_type: Type of site (e.g., 'source_leak', 'allosteric').
        index: Numeric index for disambiguation within a structure+type.

    Returns:
        Canonical site_id string.
    """
    safe_type = re.sub(r"[^a-z0-9_]", "_", site_type.lower().strip())
    return f"{structure_id}:site:{safe_type}:{index}"


def validate_residue_id(residue_id: str) -> bool:
    """Check if a residue_id matches the canonical format."""
    return bool(_RESIDUE_ID_PATTERN.match(residue_id))


def coerce_residue_id(residue_id: str, structure_id: str) -> str:
    """Normalize partial residue identifiers to canonical ``structure:chain:index`` form."""
    raw = residue_id.strip()
    structure_id = structure_id.strip()
    if not raw:
        return raw
    if validate_residue_id(raw):
        return raw

    parts = raw.split(":")
    if len(parts) == 2:
        chain_label, index_str = parts
        try:
            residue_index = int(index_str)
        except ValueError:
            return raw
        return make_residue_id(structure_id, chain_label, residue_index)

    if len(parts) == 3:
        chain_label, index_str, insertion_code = parts
        try:
            residue_index = int(index_str)
        except ValueError:
            return raw
        if insertion_code.isalpha() and len(insertion_code) == 1:
            return make_residue_id(
                structure_id, chain_label, residue_index, insertion_code
            )
        return raw

    return raw


def coerce_residue_ids(residue_ids: list[str], structure_id: str) -> list[str]:
    """Normalize a list of residue identifiers, preserving order and deduplicating."""
    normalized: list[str] = []
    seen: set[str] = set()
    for residue_id in residue_ids:
        canonical = coerce_residue_id(residue_id, structure_id)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def validate_structure_id(structure_id: str) -> bool:
    """Check if a structure_id matches the canonical format."""
    return bool(_STRUCTURE_ID_PATTERN.match(structure_id))
