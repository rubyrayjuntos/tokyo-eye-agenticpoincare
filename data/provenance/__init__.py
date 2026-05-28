# Migrated from: new (Phase 1/2 implementation) on 2026-05-27
"""Provenance lineage tooling for the Tokyo Eye governed data layer.

This package provides:
- Lineage queries (full DAG traversal for any asset or residue)
- Provenance reporting (what runs contributed to a given output)
- Audit trail for normalization attempts
"""

from data.provenance.lineage import ProvenanceTracer

__all__ = ["ProvenanceTracer"]
