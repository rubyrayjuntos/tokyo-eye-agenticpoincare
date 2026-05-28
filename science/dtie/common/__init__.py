# Migrated from: new (Phase 1 implementation) on 2026-05-27
"""Common utilities shared across v3 and v4 DTIE science code.

This package contains:
- keys: Canonical key generation (residue_id, structure_id, etc.)
- normalizer_payloads: Pydantic schemas for the governed write path
"""

from science.dtie.common.keys import (
    make_atom_id,
    make_chain_id,
    make_residue_id,
    make_site_id,
    make_structure_id,
    validate_residue_id,
)

__all__ = [
    "make_atom_id",
    "make_chain_id",
    "make_residue_id",
    "make_site_id",
    "make_structure_id",
    "validate_residue_id",
]
