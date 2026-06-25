"""Post-ingest alignment sidecar — UniProt residue mapping + structural superposition.

This package handles:
  1. SIFTS PDB↔UniProt residue mapping from RCSB (sifts_mapper.py)
  2. SVD-based Kabsch superposition for structure pairs (kabsch_aligner.py)
  3. Orchestration of alignment + persistence (alignment_engine.py)

Usage:
    from science.dtie.alignment.sifts_mapper import fetch_sifts_mapping
    from science.dtie.alignment.kabsch_aligner import compute_kabsch_superposition
    from science.dtie.alignment.alignment_engine import AlignmentEngine
"""

from __future__ import annotations
