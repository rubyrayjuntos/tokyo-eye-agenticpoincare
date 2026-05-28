# Aurora Migrations Index

**Location:** `data/aurora/migrations/`

This directory contains the ordered SQL migrations for the governed data layer.

## Current Migrations (as of 2026-05-27)

| File | Purpose | Key Elements |
|------|---------|--------------|
| `001_core_dimensions.sql` | Establish dimensional foundation | `dim_structure`, `dim_chain`, `dim_residue`, `dim_atom`, `dim_site` + bridge + `embedding_space` registry |
| `002_example_fact_tables.sql` | Demonstrate residue-centric fact table patterns | `fact_gnn_node_output`, `fact_phase_output`, `fact_site_embedding` |
| `003_provenance.sql` | Provenance spine | `provenance_run`, `provenance_event` |
| `004_gnn_embeddings.sql` | GNN embeddings & vector facts | `fact_gnn_node_embedding`, `fact_residue_graph_features` (residue-anchored, multi-space) |
| `005_dtie_phase_outputs.sql` | Representative DTIE phase outputs | `fact_phase1_witness_embedding`, `fact_phase3_persistence`, `fact_dehydron` (residue-linked where appropriate) |
| `006_dehydron_facts.sql` | Dehydron data (residue-linked) | `fact_dehydron` with donor/acceptor residue FKs |
| `007_void_and_topological_facts.sql` | Void detection and topological outputs | Void, phase 3/3.5, and bridge tables |
| `008_immunogenicity_and_redzone.sql` | Immunogenicity and metabolic liability | `fact_immunogenicity_run`, `fact_immunogenic_epitope`, `fact_metabolism_site` |
| `009_folding_and_validation.sql` | Folding paths and validation mining | `fact_folding_path`, validation runs, glue sites, wrapper suggestions, bridges |
| `010_folding_paths.sql` | Refined folding trajectory metadata | Production-grade folding path table with GCS pointers |
| `011_energy_and_hdx.sql` | Energy calculations and HDX correlation | `fact_energy_provenance_step`, `fact_hdx_correlation` |
| `012_annotations_and_cdd.sql` | External and custom annotations | `fact_cdd_annotation`, `fact_generic_annotation` |
| `013_embedding_space_registry.sql` | Embedding space registry | First-class registry for Euclidean, Hyperbolic, and future spaces |
| `014_flexible_computed_properties.sql` | Extensibility escape hatch | `fact_computed_property` for rapid addition of new metrics |
| `015_site_level_facts.sql` | Site-level scientific outputs | `fact_site_output` and contribution bridge for higher-order constructs |
| `016_provenance_helpers.sql` | Provenance helper functions | PL/pgSQL functions for lineage queries (starter set) |
| `017_helper_views.sql` | Common helper views | Residue-centric views for embeddings, site membership, and provenance summaries |
| `018_site_and_residue_bridges.sql` | Site/residue relationships + summary | Enhanced bridges and `fact_residue_summary` for fast common queries |
| `019_common_residue_views.sql` | Common helper views | Residue-centric views for embeddings, site membership, and provenance |
| `020_materialized_view_candidates.sql` | Performance layer candidates | Example materialized views for high-frequency queries |
| `021_residue_summary_materialized.sql` | Residue current state view | Broad materialized view combining embeddings, dehydrons, and sites |
| `022_governed_asset_catalog.sql` | Central governed asset registry | Core table for tracking all governed assets with provenance |
| `023_governed_write_path_notes.sql` | Future normalizer design notes | Placeholder + design intent for the single authorized write path |
| `024_rag_readiness_scaffolding.sql` | RAG readiness marker | Design notes confirming the model is shaped for future RAG work |
| `025_phase1_core_refinement.sql` | Phase 1 hardening marker | Intent to refine and harden the core model based on Phase 0.2 learnings |
| `026_canonical_key_functions.sql` | Canonical key generation | SQL functions for residue_id/chain_id construction + CHECK constraint |
| `027_hyp_projection_vector.sql` | Native hyperbolic projection | VECTOR(2) column for 2D Poincaré disc projections + IVFFlat index |
| `028_normalization_audit.sql` | Normalization audit trail | `normalization_audit` table for tracking all write attempts |
| `029_production_views.sql` | Production access views | Agent, visualizer, and governance views (purpose-built for consumers) |

## Principles

- All migrations are ordered and additive.
- Every fact table should carry a `run_id` referencing the provenance system (see 003).
- Residue is the preferred grain for scientific outputs (per ADR-001).
- Support for multiple embedding spaces is built in from the start.

## Next Expected Work (Phase 2+)

- Performance testing of materialized views under realistic load
- Additional domain-specific fact tables as science code is integrated
- RAG-specific views and embedding spaces (Phase 5)
- Monitoring and alerting on normalization_audit failures

---

**This file should be kept up to date as new migrations are added.**