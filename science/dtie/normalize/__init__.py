"""Normalization layer — JIT protocol-based views for computation tasks.

This module provides named, versioned normalization protocols that transform
raw ingested ParsedStructure data into task-specific views. Raw ingest is
opinion-free; normalization applies domain-specific filtering and
transformation tailored to each downstream computation task.

Protocols:
  - graph_default: GNN graph building (protein only, Cα projection)
  - family_compare: Cross-structure family comparison (SIFTS-mapped)
  - binding_site: Binding site analysis (ligands + pocket)
  - interface: Protein-protein interface (multi-chain, assembly-aware)

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from science.dtie.normalize.protocols.base import (
    NormalizedResidue,
    NormalizedView,
    NormalizationProtocol,
    ProtocolName,
    PROTOCOL_REGISTRY,
    get_protocol,
)

__all__ = [
    "NormalizedResidue",
    "NormalizedView",
    "NormalizationProtocol",
    "ProtocolName",
    "PROTOCOL_REGISTRY",
    "get_protocol",
]
