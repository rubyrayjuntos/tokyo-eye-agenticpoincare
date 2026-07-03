# Implementation Plan: Structure Ingestion & Normalization

## Overview

Implements the full BinaryCIF-based ingestion pipeline: download, parse, enrich, score, persist, align. Built incrementally from schema → parser → normalizer path → scorer → protocols → endpoint → alignment sidecar. Directory layout follows `science/dtie/ingest/`, `science/dtie/normalize/`, and `science/dtie/alignment/`.

## Tasks

- [x] 1. Schema migration and payload models
  - [x] 1.1 Create migration 047_structure_ingestion.sql with all schema extensions (dim_structure, dim_chain, dim_residue, dim_atom columns + new tables: structure_computation_scope, fact_residue_alignment, fact_structural_alignment, fact_covalent_bond, normalization_protocol with version column). Seed 4 default protocols.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_
  - [x] 1.2 Create Pydantic payload models in `science/dtie/common/ingest_payloads.py` (IngestDimensionPayload, AlignmentPayload, and all sub-models: StructureDimension, ChainDimension, ResidueDimension, AtomDimension, CovalentBondFact, ResidueAlignmentRecord, StructuralAlignmentRecord)
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 6.2, 7.3, 8.1_

- [x] 2. BinaryCIF downloader and structure parser
  - [x] 2.1 Implement `science/dtie/ingest/__init__.py` and `science/dtie/ingest/downloader.py` — async BinaryCIF download with 3-retry exponential backoff, SHA-256 hash computation
    - URL pattern: https://files.rcsb.org/download/{pdb_id}.bcif.gz
    - _Requirements: 1.1, 1.7_
  - [x] 2.2 Implement `science/dtie/ingest/parser.py` — biotite-based BinaryCIF parser producing ParsedStructure dataclass
    - Extract all chains from _atom_site + _pdbx_poly_seq_scheme
    - Include unresolved residues (is_resolved=False from seq scheme)
    - Parse _pdbx_struct_mod_residue for parent_comp_id mapping
    - Parse _struct_conn for covalent bonds (disulf, covale)
    - Default to model 1 for NMR (store model_count)
    - Flag partial_backbone where any of {N, CA, C} missing
    - Compute max_b_factor per residue, set low_confidence_coords if > 100
    - _Requirements: 1.2, 1.9, 1.10, 1.11, 8.1, 8.3, 8.5_
  - [x] 2.3 Write property tests for parser: backbone detection, modified residue mapping, B-factor derivation
    - **Property 5: Modified residue parent mapping**
    - **Property 6: Partial backbone detection**
    - **Property 14: B-factor derived quality flags**
    - **Validates: Requirements 1.9, 1.11, 8.3, 8.5**

- [x] 3. Metadata enricher
  - [x] 3.1 Implement `science/dtie/ingest/metadata.py` — query RCSB Data API via rcsbapi.data.DataQuery for entry + entity metadata, graceful degradation if unreachable
    - Store organism, UniProt accession(s), entity type, reference sequence IDs per entity
    - Detect and flag duplicate entity instances (multiple chains same entity_id)
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6_

- [x] 4. Normalizer ingestion paths
  - [x] 4.1 Add `normalize_ingest_dimensions()` method to `data/normalizer/core.py`
    - Validates all canonical keys via validate_residue_id
    - Persists dim_structure, dim_chain, dim_residue, dim_atom, fact_covalent_bond
    - Partial-failure semantics: each table is sub-transaction (if atoms fail, dims still committed)
    - source_type='empirical' for all dimension/covalent writes
    - Provenance records: normalization_protocol, file_hash, rcsbapi_version, biotite_version
    - _Requirements: 3.5, 11.1, 11.3, 11.5, 11.6, 11.7_
  - [x] 4.2 Add `normalize_alignment()` method to `data/normalizer/core.py`
    - Persists fact_residue_alignment and fact_structural_alignment
    - source_type='deterministic'
    - _Requirements: 11.2_
  - [x] 4.3 Write property tests for Normalizer ingestion path
    - **Property 2: Idempotent re-ingest**
    - **Property 4: Key format rejection**
    - **Property 15: Source type classification**
    - **Property 16: Partial failure persistence**
    - **Validates: Requirements 1.8, 3.5, 9.4, 11.1, 11.2, 11.3, 11.5, 11.6**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Chain scorer and computation scope
  - [x] 6.1 Implement `science/dtie/ingest/chain_scorer.py`
    - Multi-factor scoring: protein type (10), UniProt coverage (3), resolved fraction (2), B-factor quality (1.5), not-duplicate (2), low mutation burden (1), sequence length (0.5)
    - Collapse duplicate entity instances → select one representative per entity_id
    - Output: ComputationScope with primary_chain_ids, reference_chain, exclude_chain_ids, scope_source, selection_reason, normalization_protocol
    - Support multi-chain scope for interface protocol
    - _Requirements: 4.1, 4.2, 4.5, 4.6_
  - [x] 6.2 Write property tests for chain scorer
    - **Property 7: Duplicate entity detection**
    - **Property 8: Chain scorer selects highest quality**
    - **Validates: Requirements 2.4, 4.1, 4.6**

- [x] 7. Normalization protocols
  - [x] 7.1 Implement `science/dtie/normalize/__init__.py`, `science/dtie/normalize/protocols/base.py`, and protocol implementations: `graph_default.py`, `family_compare.py`, `binding_site.py`, `interface.py`
    - Each protocol is versioned with version column tracked in normalization_protocol table
    - graph_default: protein atoms only, highest-occupancy altloc, harmonize modified residues, exclude unresolved, exclude partial_backbone, Cα projection
    - family_compare: SIFTS-mapped only, UniProt intersection, apply reference superposition, exclude tags/tails, retain mutations
    - binding_site: preserve ligands/cofactors within cutoff, exclude waters unless bridging, local pocket frame, protonation untouched
    - interface: multi-chain, biological assembly aware, entity-instance collapse disabled
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [x] 7.2 Implement `science/dtie/normalize/scope_selector.py` — reads structure_computation_scope and applies named protocol to filter parsed structure for GraphBuilder
    - _Requirements: 4.3_
  - [x] 7.3 Write property tests for protocol filtering
    - **Property 9: graph_default protocol filtering**
    - **Property 10: Provenance records protocol and versions**
    - **Validates: Requirements 5.2, 5.6, 11.7**

- [x] 8. Alignment engine
  - [x] 8.1 Implement `science/dtie/alignment/__init__.py` and `science/dtie/alignment/sifts_mapper.py` — fetch SIFTS PDB↔UniProt residue mapping from RCSB, populate ResidueAlignment records with reason codes for unmapped residues (tags, engineered)
    - Return empty list with warning if SIFTS unavailable
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [x] 8.2 Implement `science/dtie/alignment/kabsch_aligner.py` — SVD-based Kabsch superposition via scipy, compute rotation + translation + RMSD on comparable core (mapped UniProt positions, Cα present in both, acceptable occupancy, preferred altloc, tags/tails excluded)
    - Handle reflection case (det(R) = -1)
    - Skip if < 3 common residues
    - _Requirements: 7.1, 7.2, 7.3, 7.6_
  - [x] 8.3 Implement `science/dtie/alignment/alignment_engine.py` — orchestrates SIFTS fetch + residue alignment persistence + Kabsch computation for structure pairs sharing UniProt accession
    - Reference structure versioned and configurable
    - _Requirements: 7.4, 7.5_
  - [x] 8.4 Write property tests for alignment engine
    - **Property 11: Alignment completeness with reason codes**
    - **Property 12: Kabsch superposition validity**
    - **Validates: Requirements 6.2, 6.3, 7.1, 7.2, 7.3**

- [x] 9. Ingest endpoint and wiring
  - [x] 9.1 Add `POST /compute/ingest-full` to science container router (`science/api/routers/ingest.py`)
    - Orchestrates: idempotency check → download → parse → enrich → canonical keys → score → quality flags → normalize dimensions → scope upsert → fire alignment sidecar (background task, non-blocking)
    - Returns IngestResponse with structure_id, counts, scope, alignment_status
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [x] 9.2 Add `ingest_structure()` method to `agent/tools/science_client.py` calling `/compute/ingest-full`
    - _Requirements: 9.1_
  - [x] 9.3 Add `POST /api/ingest` agent-side endpoint in `agent/coordinator/routers/` that delegates to ScienceClient and triggers computation pipeline in parallel with alignment
    - _Requirements: 9.1, 9.2_
  - [x] 9.4 Write integration test: full ingest round-trip (submit PDB ID → verify all dimension tables populated, canonical keys valid, covalent bonds stored, test idempotency on double-submit)
    - **Property 1: Ingestion completeness**
    - **Property 3: Canonical key determinism**
    - **Property 13: Covalent bond extraction**
    - **Validates: Requirements 1.4, 1.5, 2.3, 3.1, 3.2, 3.3, 3.6, 8.1**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (no optional markers)
- Property tests written alongside each component for maximum correctness guarantees
- Directory layout: `science/dtie/ingest/` (download, parse, metadata, scorer), `science/dtie/normalize/` (protocols, scope selector), `science/dtie/alignment/` (sifts, kabsch, engine)
- Normalizer extensions in `data/normalizer/core.py` (two new methods)
- `data/normalizer/` remains the single write path for governed data; protocols live in science layer
- All canonical keys use `science/dtie/common/keys.py` exclusively
- Dehydron computation is NOT part of raw ingest — it belongs in normalization protocols (post-ingest enrichment)
