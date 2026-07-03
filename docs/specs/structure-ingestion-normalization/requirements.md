# Requirements Document

## Introduction

This feature establishes the complete structure ingestion pipeline for Tokyo Eye — from PDB ID submission through atom-level parsing, metadata enrichment, computation scope assignment, and cross-structure alignment. The design philosophy is: ingest everything RCSB offers faithfully (no opinions at ingest time), then normalize JIT before each computation task with named protocols tailored to what that task needs. A post-ingest alignment sidecar builds a standardized comparison library using UniProt canonical sequences and Kabsch superposition, enabling true 1:1 residue correspondence across structures in the same protein family.

Critical principle: raw ingest preserves deposited truth; normalization produces versioned task-specific views keyed by both deposited IDs and external canonical mappings.

## Glossary

- **Raw_Ingest**: The process of downloading BinaryCIF from RCSB and populating dimensional tables with all chains, all residues (including unresolved), all atoms (including altlocs), preserving both author and label identifiers
- **Computation_Scope**: A per-structure configuration specifying which chain(s) to use, with a scored default-chain selector and user override capability
- **Alignment_Sidecar**: A post-ingest enrichment job that maps residues to UniProt canonical positions and computes structural superpositions for cross-structure comparison
- **Normalization_Protocol**: A named preset (e.g., `graph_default`, `family_compare`, `binding_site`, `interface`) that defines the exact transform applied JIT before a specific computation task
- **Canonical_Key**: The deterministic residue_id format `{structure_id}:{chain_label}:{residue_index}:{insertion_code}` per RESIDUE_ID_KEY_STRATEGY.md
- **Deposited_Identity**: The raw mmCIF identifiers as deposited: (auth_asym_id, auth_seq_id, pdbx_PDB_ins_code, label_asym_id, label_seq_id, comp_id)
- **Comparison_Identity**: The external canonical mapping: (uniprot_accession, uniprot_position, isoform_id, mapping_source, mapping_confidence)
- **SIFTS_Mapping**: The authoritative PDB-UniProt residue correspondence maintained by RCSB/PDBe
- **Kabsch_Superposition**: Least-squares rotation + translation that minimizes RMSD between two aligned coordinate sets
- **BinaryCIF**: The compressed binary format of mmCIF served by RCSB (smaller, faster to parse than text CIF)
- **Comparable_Core**: The subset of residues used for superposition: mapped UniProt positions, present Cα in both structures, acceptable occupancy, excluded tags/tails
- **Reference_Structure**: The structure in a protein family that all others are superposed onto (versioned, configurable)

## Requirements

### Requirement 1: BinaryCIF Download and Full Atom-Level Parsing

**User Story:** As a researcher, I want structures ingested with full atom-level detail from BinaryCIF, so that downstream computations have access to coordinates, altlocs, occupancies, B-factors, SSE, and modified residue mappings without re-fetching.

#### Acceptance Criteria

1. WHEN a PDB ID is submitted for ingestion, THE Science_API SHALL download the BinaryCIF file from RCSB
2. WHEN the BinaryCIF is downloaded, THE Science_API SHALL parse it using biotite to extract all chains, all residues (including unresolved from _pdbx_poly_seq_scheme), and all atoms
3. THE Science_API SHALL populate dim_structure with: structure_id (canonical), pdb_id, source, resolution, r_factor, r_free, method, title, organism, release_date, polymer_composition, model_count, assembly_id
4. THE Science_API SHALL populate dim_chain for ALL polymer chains preserving BOTH auth_asym_id AND label_asym_id, plus entity_id, entity_type, sequence_length, is_entity_duplicate, is_representative
5. THE Science_API SHALL populate dim_residue for ALL residues including unresolved ones (flagged is_resolved=false), storing: auth_seq_id, label_seq_id, insertion_code, residue_name, residue_name_3, comp_id, parent_comp_id (for modified residues), sse_code, is_resolved, is_modified, max_b_factor, low_confidence_coords
6. THE Science_API SHALL populate dim_atom with ALL atoms: atom_name, element, x, y, z, occupancy, b_factor, altloc (nullable), is_hetero, model_id
7. IF the BinaryCIF download fails, THEN THE Science_API SHALL retry 3 times with exponential backoff before returning an error
8. IF a structure already exists, THEN THE Science_API SHALL skip re-ingestion and return existing metadata (idempotent)
9. THE Science_API SHALL parse _pdbx_struct_mod_residue to store parent standard residue mapping (MSE-MET, SEP-SER, etc.)
10. WHEN the structure is NMR, THE Science_API SHALL default to model 1, store model_count, and expose model_index in Computation_Scope
11. THE Science_API SHALL flag residues where backbone atoms are incomplete as partial_backbone=true for downstream exclusion decisions

### Requirement 2: Metadata Enrichment via RCSB Data API

**User Story:** As a researcher, I want rich metadata (organism, UniProt cross-references, entity hierarchy, quality metrics) available immediately after ingestion.

#### Acceptance Criteria

1. WHEN a structure is ingested, THE Science_API SHALL query RCSB Data API (via rcsbapi.data.DataQuery) for entry-level and entity-level metadata
2. THE Science_API SHALL store per-entity: organism, UniProt accession(s), entity type, reference sequence identifiers
3. THE Science_API SHALL store the auth_asym_id to label_asym_id mapping explicitly on dim_chain
4. THE Science_API SHALL detect duplicate entity instances (multiple chains same entity_id) and flag them
5. WHEN the RCSB Data API is unreachable, THE Science_API SHALL proceed with BinaryCIF-only metadata
6. THE metadata enrichment SHALL use rcsbapi's built-in batching and rate limiting

### Requirement 3: Canonical Key Generation

**User Story:** As a system architect, I want deterministic canonical keys for all ingested data guaranteeing cross-run joins, idempotent upserts, and human readability.

#### Acceptance Criteria

1. THE Ingest_Engine SHALL generate structure_id via make_structure_id(pdb_id, source="rcsb") producing lowercase PDB ID
2. THE Ingest_Engine SHALL generate chain_id via make_chain_id(structure_id, auth_asym_id) — author chain label is canonical
3. THE Ingest_Engine SHALL generate residue_id via make_residue_id(structure_id, auth_asym_id, auth_seq_id, insertion_code) preserving author numbering including negatives
4. ALL key generation SHALL use science/dtie/common/keys.py functions exclusively
5. THE Normalizer SHALL reject payloads where residue_id does not match format regex
6. THE key functions SHALL treat (auth_seq_id, insertion_code) as the uniqueness pair per chain — not auth_seq_id alone

### Requirement 4: Computation Scope with Scored Chain Selector

**User Story:** As a researcher, I want intelligent chain selection using multiple quality signals, with override capability.

#### Acceptance Criteria

1. WHEN ingested, THE Science_API SHALL compute default scope using a multi-factor scorer: polymer type, UniProt coverage, resolved residue count, B-factor distribution, entity duplicate status, mutation burden
2. THE scope SHALL be stored with: primary_chain_ids, reference_chain, exclude_chain_ids, scope_source, selection_reason, normalization_protocol
3. THE GraphBuilder SHALL read scope and filter chains applying the named normalization protocol
4. THE user SHALL be able to override scope via dashboard with scope_source='user'
5. THE scope SHALL support multi-chain for interface analysis
6. THE scorer SHALL collapse duplicate entity instances selecting ONE representative

### Requirement 5: Named Normalization Protocols

**User Story:** As a system architect, I want normalization applied via named versioned presets so every computed result points to a reproducible protocol.

#### Acceptance Criteria

1. THE System SHALL define protocols: graph_default, family_compare, binding_site, interface
2. graph_default: scoped chains, protein atoms only, highest-occupancy altloc, modified residue harmonization, unresolved excluded, Cα projection
3. family_compare: SIFTS-mapped only, UniProt intersection, reference superposition applied, tags excluded, mutation sites retained
4. binding_site: preserve ligands/cofactors within cutoff, waters by policy, local pocket frame, protonation untouched
5. interface: multi-chain, biological assembly aware, entity-instance collapse disabled
6. EACH protocol SHALL be versioned; provenance_run SHALL record protocol name + version

### Requirement 6: Post-Ingest UniProt Alignment

**User Story:** As a researcher, I want each residue mapped to its UniProt canonical position for true cross-structure correspondence.

#### Acceptance Criteria

1. WHEN UniProt accession is available, THE Alignment_Engine SHALL fetch SIFTS mapping from RCSB
2. THE Alignment_Engine SHALL populate fact_residue_alignment with: residue_id, uniprot_accession, uniprot_position, isoform_id, mapping_source, mapping_confidence, reason_code
3. WHEN residues have no mapping (tags, engineered), THE Alignment_Engine SHALL record uniprot_position=NULL with reason_code
4. THE alignment SHALL use RCSB SIFTS data (authoritative) not local sequence alignment
5. WHEN SIFTS unavailable, THE Alignment_Engine SHALL log warning and skip (structure remains usable)
6. Cross-structure comparison SHALL join on uniprot_position — never on author numbering alone

### Requirement 7: Structural Superposition for Family Comparison

**User Story:** As a researcher, I want family structures superposed onto a common reference frame for meaningful geometric comparison.

#### Acceptance Criteria

1. WHEN two structures share UniProt accession, THE Alignment_Engine SHALL compute Kabsch superposition on the comparable core
2. THE comparable core SHALL be: mapped UniProt positions, Cα present in both, acceptable occupancy, preferred altloc, tags/tails excluded
3. THE result SHALL be stored: rotation_matrix, translation, RMSD, aligned_residue_count, comparable_core definition (JSON)
4. THE reference structure SHALL be versioned and configurable; changing it triggers re-superposition
5. compare_wt_mutant SHALL apply stored superposition before computing displacement vectors
6. THE alignment SHALL store both result AND comparable_core definition for reproducibility

### Requirement 8: Covalent Connectivity and Quality Gates

**User Story:** As a researcher, I want disulfide bonds, covalent modifications, and quality metrics stored at ingest time.

#### Acceptance Criteria

1. THE Ingest_Engine SHALL parse _struct_conn to extract covalent bonds into fact_covalent_bond
2. THE Ingest_Engine SHALL store resolution, r_factor, r_free in dim_structure
3. THE Ingest_Engine SHALL compute per-residue max B-factor on dim_residue
4. THE Computation_Scope SHALL expose quality filters (max_b_factor_threshold, min_resolution)
5. WHEN residue B-factor > 100, THE System SHALL flag low_confidence_coords=true

### Requirement 9: Ingestion Endpoint Integration

**User Story:** As the dashboard, I want a single ingest call that triggers the full pipeline automatically.

#### Acceptance Criteria

1. THE POST /api/ingest endpoint SHALL call science container /compute/ingest-full for BinaryCIF-based ingestion
2. WHEN ingestion completes, THE System SHALL trigger: computation pipeline AND alignment sidecar in parallel
3. THE ingest response SHALL include: structure_id, chain_count, residue_count, atom_count, computation_scope, alignment_status
4. THE same PDB ID submitted twice SHALL return existing metadata (idempotent)
5. THE alignment sidecar SHALL NOT block the computation pipeline

### Requirement 10: Schema Extensions

**User Story:** As a system architect, I want the database schema to support dual identity, computation scope, alignment, covalent bonds, and quality data.

#### Acceptance Criteria

1. Extend dim_chain with: label_asym_id, uniprot_accession, uniprot_start, uniprot_end, sequence_length, entity_id, is_entity_duplicate, is_representative
2. Extend dim_residue with: label_seq_id, comp_id, parent_comp_id, is_resolved, is_modified, max_b_factor, low_confidence_coords, partial_backbone
3. Extend dim_structure with: title, organism, release_date, polymer_composition, r_factor, r_free, model_count, assembly_id
4. Add structure_computation_scope table
5. Add fact_residue_alignment table
6. Add fact_structural_alignment table with comparable_core JSONB and protocol_version
7. Add fact_covalent_bond table
8. Add normalization_protocol table (protocol_name PK, version, parameters JSONB)

### Requirement 11: Normalizer Compliance

**User Story:** As a system architect, I want all ingestion writes governed with provenance.

#### Acceptance Criteria

1. Dimension writes SHALL use source_type='empirical'
2. Alignment writes SHALL use source_type='deterministic'
3. Covalent bond writes SHALL use source_type='empirical'
4. Computation scope is configuration (direct upsert, no Normalizer)
5. Partial failures SHALL persist what succeeded without rolling back
6. Normalizer SHALL validate canonical key format on all residue_id values
7. Provenance SHALL record: normalization_protocol used, BinaryCIF file hash, rcsbapi version, biotite version
