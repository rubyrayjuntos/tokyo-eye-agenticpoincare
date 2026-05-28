# GOSP Data Lineage Map
**Date:** 2026-03-21
**Status:** Under reconciliation

This document traces every piece of data from its origin through computation,
storage, and delivery to its consumer. It is the single source of truth for
what the system does and how data flows through it.

---

## Trigger: POST /ingest

```
Tokyo Eye (browser)
    │  POST /ingest { "pdb_id": "7XKJ" }
    ▼
Cloud Run: api_v2/ingest.py
    │  ingestion_broker.ingest_structure_broker(pdb_id, fmt="bcif")
    │      → RCSB → PDBe → AlphaFold fallback (BinaryCIF only)
    │  normalizer/structure.normalize_structure()
    ▼
AlloyDB (written)
    dim_structure      structure_id (UUID v4), pdb_id, resolution, method, ...
    dim_chain          chain_id → structure_id, chain_label, entity_type, ...
    dim_residue        residue_id → chain_id, residue_index, residue_name, sse_code, sasa=NULL
    dim_atom           atom_id → residue_id, atom_name, element, x, y, z
    fact_job_status    queued rows for: fast_path, void, cdd, immunogenicity, energy,
                       metabolism, lerp, validation_mining
    ▼
Cloud Tasks
    Enqueues: fast_path + void + cdd + immunogenicity + metabolism + lerp + validation_mining
    ▼
Returns immediately: { structure_id, pdb_id, job_ids }
```

---

## TIER 1 — Fast Path (sequential, ~10–15s)

*Triggered by Cloud Tasks after ingest. Steps are sequential — each must complete before the next starts.*

### Step 1 — Per-Residue SASA

| | |
|---|---|
| **Service** | `services/physics_kernel.calculate_per_residue_sasa(structure)` |
| **Method** | FreeSASA, extracts per-residue area from `freesasa.ResidueArea` |
| **Writes** | `UPDATE dim_residue SET sasa = :value WHERE residue_id = :id` |
| **Source type** | `deterministic` |
| **Consumer** | GNN payload assembler (Step 3), `GET /structures/{id}/atoms` |

### Step 2 — Dehydron Detection

| | |
|---|---|
| **Service** | `services/dehydron_detection.detect_dehydrons(structure)` |
| **Method** | KDTree wrapping count algorithm |
| **Normalizer** | `normalizer/dehydron.normalize_dehydrons()` |
| **Writes** | `fact_dehydron`: dehydron_id, structure_id, donor_chain, donor_residue_index, acceptor_chain, acceptor_residue_index, midpoint_x/y/z, distance, wrapping_count, is_dehydron, is_interchain, **source_type='deterministic'** |
| **Pub/Sub** | `dehydron-ready` |
| **Consumer** | GNN payload (Step 3), void bridge (Tier 2), validation mining (Tier 2), `GET /structures/{id}/dehydrons` |

### Step 3 — Assemble GNN Payload + Publish gnn-ready

| | |
|---|---|
| **Reads from DB** | `dim_residue.sasa`, `dim_residue.sse_code`, `fact_dehydron` (wrapping_count, is_dehydron) |
| **Normalizer** | `normalizer/gnn_payload.assemble_gnn_payload()` |
| **Node features** | `[rho (wrapping_count), tau_flag (is_dehydron), ss_type (sse_code), sasa]` per residue |
| **Edges** | C-alpha distance cutoff → edge_index + edge_attr |
| **Serializes** | `torch_geometric.data.Data` → GCS: `gs://{bucket}/gnn-payloads/{structure_id}/payload.pt` |
| **Pub/Sub** | Publishes `gnn-ready` → `{ structure_id, gcs_uri }` |
| **NULL sentinel** | If `dim_residue.sasa` is NULL: substitute 0.0, record residue_ids in `fact_gnn_inference.sasa_null_warnings` |
| **Consumers** | Vertex AI (push subscription), validation mining (trigger), viewport (event) |

---

## GNN WRITE-BACK — Vertex AI → AlloyDB

*Vertex AI receives `gnn-ready` push, runs GOSPConeMapper inference, POSTs result back.*

**STATUS: BLOCKED** — The Vertex AI endpoint for the GNN model is not yet deployed due to a project quota issue (`CustomModelServingT4GPUsPerProjectPerRegion`). The `gnn-ready-vertex-ai-subscription` push subscription has been created, but the pipeline will not be functional until the quota is increased and the model is deployed.

| | |
|---|---|
| **Normalizer** | `normalizer/gnn.normalize_gnn_inference()` |
| **Writes** | `fact_gnn_inference`: gnn_run_id, structure_id, model_version, projection_dim, num_nodes, num_edges, curvature_value, depth_conditioning_enabled, balance_loss, audit_trail (JSONB), sasa_null_warnings (JSONB), **source_type='probabilistic'** |
| **Writes** | `fact_gnn_node_output`: node_id, gnn_run_id, residue_id — **Input features** (input_rho, input_tau_flag, input_ss_type, input_sasa) + **Outputs** (projections JSONB, cone_depth, cone_width, expert_weights JSONB, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, evidence_mu/nu/alpha/beta), **source_type='probabilistic'** |
| **Pub/Sub** | Publishes `gnn-inference-ready` |
| **Consumer** | `GET /structures/{id}/gnn` → viewport renders Poincaré disc, cone geometry |
| **Note** | Dehydron conformation state is derived on the fly in the viewport from projections/cone_depth/cone_width — **never persisted** |

---

## TIER 2 — Parallel Async Jobs

*All fire after ingest via Cloud Tasks. Independent retry (3×, exponential backoff).*

### Void Detection

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Service** | `services/void_detection.detect_voids(structure)` — DBSCAN clustering |
| **Reads from DB** | `fact_dehydron` — fetches dehydron_id map to resolve bridge FKs |
| **Normalizer** | `normalizer/void.normalize_voids()` |
| **Writes** | `fact_void`: void_id, structure_id, center_x/y/z, volume, point_count, **source_type='deterministic'** |
| **Writes** | `bridge_void_dehydron`: void_id → dehydron_id (for each nearby dehydron) |
| **Pub/Sub** | Publishes `void-ready` |
| **Consumer** | Validation mining (reads from DB), `GET /structures/{id}/voids` → viewport |

### CDD Annotations

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Service** | `services/cdd_annotations.fetch_cdd_annotations(structure)` — NCBI CDD API |
| **Normalizer** | `normalizer/cdd.normalize_cdd_annotations()` |
| **Writes** | `fact_cdd_annotation`: annotation_id, structure_id, chain_label, domain_id, domain_name, start_residue, end_residue, e_value, bit_score, is_synthetic (TRUE if stub/fallback), **source_type='external'** |
| **Pub/Sub** | Publishes `cdd-ready` |
| **Consumer** | Validation mining (reads CDD gates from DB instead of live API) |
| **Graceful degradation** | If NCBI unavailable: skips cleanly, marks job complete with note |

### Immunogenicity Screening

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Service** | `services/immunogenicity_screening.screen_immunogenicity(structure)` — SASA + NetMHCIIpan |
| **Job Handler** | `jobs/tier2.run_immunogenicity_job` |
| **Normalizer** | `normalizer/redzone.normalize_immunogenicity()` |
| **Writes** | `fact_immunogenicity_run`, `fact_immunogenic_epitope`, `fact_job_status` |
| **Pub/Sub** | Publishes `redzone-ready` with `{ source: 'immunogenicity' }` |
| **Consumer** | `GET /structures/{id}/redzone` |

### Metabolic Liability Screening

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Service** | `services/metabolic_liability_screening.screen_metabolic_liability(structure)` — RDKit SMARTS CYP450 |
| **Job Handler** | `jobs/tier2.run_metabolism_job` |
| **Normalizer** | `normalizer/redzone.normalize_metabolism()` |
| **Writes** | `fact_metabolism_run`, `fact_metabolism_site`, `fact_job_status` |
| **Pub/Sub** | Publishes `redzone-ready` with `{ source: 'metabolism' }` |
| **Consumer** | `GET /structures/{id}/redzone` |

### LERP Folding Path

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Service** | `services/physics_kernel.compute_lerp_path(structure)` |
| **Output** | numpy array `(num_frames, num_atoms, 3)` |
| **GCS** | Uploads `float32` bytes → `gs://{bucket}/folding-paths/{structure_id}/lerp_frames.npy` |
| **Normalizer** | `normalizer/folding.normalize_folding_path()` |
| **Writes** | `fact_folding_path`: path_id, structure_id, method='lerp', num_frames, num_atoms, gcs_uri, **source_type='deterministic'** |
| **Pub/Sub** | Publishes `folding-path-ready` with `{ path_id, gcs_uri }` |
| **Consumer** | `GET /structures/{id}/folding` → returns GCS URI (not frame data). Viewport streams frames directly from GCS. |
| **Graceful degradation** | If `compute_lerp_path` not available: skips cleanly |

### Validation Mining *(waits on gnn-ready)*

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued by Tier 1 job handler after `gnn-ready` is published) |
| **Reads from DB** | `fact_dehydron` (precomputed), `fact_void` (precomputed), `fact_cdd_annotation` (precomputed) |
| **Service** | `services/validation_mining_pipeline.ValidationMiningPipeline.run_with_precomputed()` |
| **Note** | Does NOT re-execute dehydron/void/CDD detection — reads from DB |
| **Normalizer** | `normalizer/validation.normalize_validation_run()` |
| **Writes** | `fact_validation_mining_run`: run_id, structure_id, status, duration_sec, started_at |
| **Writes** | `fact_outlier_correlation`: correlation_id, run_id, outlier_id, dehydron_id, distance, correlation_score |
| **Writes** | `fact_glue_site`: glue_site_id, run_id, avg_rho, void_volume_est, score, representative_label |
| **Writes** | `bridge_glue_site_dehydron`: glue_site_id → dehydron_id |
| **Writes** | `fact_wrapper_suggestion`: wrapper_suggestion_id, glue_site_id, wrapper_type, wrapping_gain, predicted_ddg |
| **Pub/Sub** | Publishes `validation_mining-ready` with `{ run_id }` |
| **Consumer** | `GET /structures/{id}/validation` → `v_validation_mining_summary` view → viewport + PROTAC workflow |

### Synthesis Feasibility + Autoprotocol

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Job Handler** | `jobs/tier2.run_synthesis_job` |
| **Normalizer** | `normalizer/synthesis.py` |
| **Writes** | `dim_sequence` (upsert), `fact_synthesis_feasibility`, `fact_autoprotocol` |
| **Pub/Sub** | Publishes `synthesis-ready` and `protocol-ready` |
| **Consumer** | `GET /structures/{id}/synthesis` |

### Energy Calculation

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued at ingest) |
| **Job Handler** | `jobs/tier2.run_energy_job` |
| **Normalizer** | `normalizer/energy.py` |
| **Writes** | `fact_energy_calculation`, `fact_energy_provenance_step` |
| **Consumer** | `GET /structures/{id}/energy` |

### Site Embedding Generation *(waits on gnn-inference-ready)*

| | |
|---|---|
| **Trigger** | Cloud Tasks (enqueued after `gnn-inference-ready` event) |
| **Reads from DB** | `fact_gnn_node_output`, `dim_residue` (for feature vector v1) |
| **Job Handler** | `jobs/embedding_writer.run_embedding_writer_job` |
| **Feature Spec** | `docs/vector-embeddings-feature-spec-v1.md` |
| **Writes** | `fact_site_embedding`: site_embedding_id, structure_id, residue_id, cone_depth, source_type, embedding_model, embedding_dim, embedding (VECTOR), feature_version, ... |
| **Pub/Sub** | Publishes `embedding-ready` |
| **Consumer** | `GET /sites/{site_id}/similar` |
| **Note** | This is an additive, asynchronous batch job. It populates the vector DB for similarity search. |

### HDX Correlation *(on-demand)*

| | |
|---|---|
| **Trigger** | `POST /structures/{id}/hdx` — user uploads DynamX CSV |
| **Service** | `services/hdx_ms_validation.HDXMSValidator` — Spearman correlation |
| **Normalizer** | `normalizer/hdx.normalize_hdx_correlation()` |
| **Writes** | `fact_hdx_correlation`: hdx_run_id, structure_id, r_squared, spearman_rho, p_value, passed, num_residues, **source_type='empirical'** |
| **Writes** | `fact_hdx_residue`: hdx_residue_id, hdx_run_id, residue_index, fractional_uptake, wrapping_count, num_peptides |
| **Pub/Sub** | Publishes `hdx-ready` |
| **Consumer** | `GET /structures/{id}/hdx` |

---

## API Endpoints → DB Tables

| Endpoint | Tables queried | Source type filter |
|---|---|---|
| `GET /structures/{id}` | `dim_structure`, `dim_chain`, `dim_residue` | — |
| `GET /structures/{id}/status` | `fact_job_status` | — |
| `GET /structures/{id}/atoms` | `dim_atom` JOIN `dim_residue` (sasa) | — |
| `GET /structures/{id}/dehydrons` | `fact_dehydron` | ✅ |
| `GET /structures/{id}/voids` | `fact_void`, `bridge_void_dehydron` | ✅ |
| `GET /structures/{id}/energy` | `fact_energy_calculation`, `fact_energy_provenance_step` | ✅ |
| `GET /structures/{id}/redzone` | `fact_immunogenicity_run`, `fact_metabolism_run` | ✅ |
| `GET /structures/{id}/validation` | `v_validation_mining_summary`, `fact_glue_site`, `fact_wrapper_suggestion` | — |
| `GET /structures/{id}/gnn` | `fact_gnn_inference`, `fact_gnn_node_output` | ✅ |
| `GET /structures/{id}/folding` | `fact_folding_path` | — |
| `GET /structures/{id}/synthesis` | `fact_synthesis_feasibility`, `fact_autoprotocol` | ✅ |
| `GET /structures/{id}/audit` | `fact_audit_event` | — |
| `GET /structures/{id}/hdx` | `fact_hdx_correlation`, `fact_hdx_residue` | — |
| `GET /sites/{site_id}/similar` | `fact_site_embedding` | — |
| `WS /ws/{id}` | Pub/Sub event relay to viewport | — |

---

## Pub/Sub Topics

| Topic | Status | Published by | Subscribers |
|---|---|---|---|
| `structure-ready` | Found | `normalizer/structure` | Cloud Tasks dispatcher, WebSocket (`structure-ready-ws-sub`) |
| `gnn-ready` | Found | `normalizer/gnn_payload` | Vertex AI (push), Cloud Tasks (validation_mining trigger), WebSocket |
| `gnn-inference-ready` | Found | `normalizer/gnn` | WebSocket (`gnn-inference-ready-ws-sub`) |
| `void-ready` | Found | `jobs/tier2.run_void_job` | WebSocket (`void-ready-ws-sub`) |
| `cdd-ready` | Found | `jobs/tier2.run_cdd_job` | Cloud Tasks (validation_mining reads CDD from DB), WebSocket (`cdd-ready-ws-sub`) |
| `folding-path-ready` | Found | `jobs/tier2.run_lerp_job` | WebSocket (`folding-path-ready-ws-sub`) |
| `redzone-ready` | Found | `jobs/tier2.run_immunogenicity_job`, `jobs/tier2.run_metabolism_job` | WebSocket (`redzone-ready-ws-sub`) |
| `validation-mining-ready`| Found | `jobs/tier2.run_validation_mining_job` | WebSocket (`validation-mining-ready-ws-sub`) |
| `hdx-ready` | Found | `normalizer/hdx` | WebSocket (`hdx-ready-ws-sub`) |
| `dehydron-ready` | Found | `normalizer/dehydron` | WebSocket (`dehydron-ready-ws-sub`) |
| `energy-ready` | Found | `jobs/tier2.run_energy_job` | WebSocket (`energy-ready-ws-sub`) |
| `embedding-ready` | New | `jobs/embedding_writer` | WebSocket (`embedding-ready-ws-sub`) |

---

## Pub/Sub Subscriptions

| Subscription | Topic | Target | Status |
|---|---|---|---|
| `gnn-ready-vertex-ai-subscription` | `gnn-ready` | Vertex AI Endpoint `tokyo-eye-gnn-endpoint` (push) | **Blocked** |
| `structure-ready-ws-sub` | `structure-ready` | WebSocket (pull) | Active |
| `gnn-inference-ready-ws-sub` | `gnn-inference-ready` | WebSocket (pull) | Active |
| `void-ready-ws-sub` | `void-ready` | WebSocket (pull) | Active |
| `cdd-ready-ws-sub` | `cdd-ready` | WebSocket (pull) | Active |
| `folding-path-ready-ws-sub` | `folding-path-ready` | WebSocket (pull) | Active |
| `redzone-ready-ws-sub` | `redzone-ready` | WebSocket (pull) | Active |
| `validation-mining-ready-ws-sub` | `validation-mining-ready` | WebSocket (pull) | Active |
| `hdx-ready-ws-sub` | `hdx-ready` | WebSocket (pull) | Active |

---

## Known Gaps & Future Dev

| # | Gap | Endpoint affected | Notes |
|---|---|---|---|
| 1 | Tier 1 SASA calculation is a stub | `GET /structures/{id}/atoms` (sasa field) | The `run_tier1` job handler logs a message but does not call the `physics_kernel` or update `dim_residue.sasa`. |
| 2 | Tier 1 GNN Payload assembly is a stub | `gnn-ready` event | The `run_tier1` job handler logs a message but does not call the `gnn_payload` normalizer. The `gnn-ready` event is published with a fake GCS URI. |
| 3 | `run_cdd_job` is not implemented | `GET /structures/{id}/validation` | The Tier 2 job handler for CDD annotations is missing. The `fact_cdd_annotation` table will not be populated. |
| 4 | `run_lerp_job` is not implemented | `GET /structures/{id}/folding` | The Tier 2 job handler for LERP folding path calculation is missing. The `fact_folding_path` table will not be populated. |
| 5 | `run_validation_mining_job` is not implemented | `GET /structures/{id}/validation` | The Tier 2 job handler for validation mining is missing. |
| 6 | GNN inference pipeline is blocked | `GET /structures/{id}/gnn` | The Vertex AI endpoint is not deployed due to a project quota issue. The `fact_gnn_inference` and `fact_gnn_node_output` tables will not be populated. |

---

## Data Provenance Summary

| source_type | Tables | Meaning |
|---|---|---|
| `empirical` | `fact_hdx_correlation`, `fact_hdx_residue` | Raw from instrument / authoritative external source |
| `deterministic` | `fact_dehydron`, `fact_void`, `fact_folding_path`, `fact_energy_calculation`, `fact_autoprotocol`, `dim_residue.sasa` | Physics/algorithm, no stochastic component |
| `probabilistic` | `fact_gnn_inference`, `fact_gnn_node_output` | GNN output, uncertainty-quantified |
| `external` | `fact_cdd_annotation`, `fact_immunogenicity_run`, `fact_synthesis_feasibility` | Third-party API (NCBI, NetMHCIIpan, Twist) |

---

## Database

The project has been migrated from Cloud SQL to AlloyDB for PostgreSQL to leverage its superior performance and vector embedding capabilities. The current AlloyDB cluster is `tokyo-eye-cluster` (version PostgreSQL 17) located in `us-central1`.

**Vector Embeddings:** The vector embedding functionality has **not** yet been enabled for this cluster. This will require setting the `google_ml_integration.enable_embedding` flag on the database instance.
