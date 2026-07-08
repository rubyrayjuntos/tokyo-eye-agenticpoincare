# GNN Topology Input — Remove SASA from Hyperbolic Inference

**Repository:** tokyo-eye-agenticpoincare  
**Version:** 0.1 (draft)  
**Date:** July 2026  
**Status:** Normative for cold-start lineage v6.1+  
**Owner:** Science / training governance  

---

## 1. Problem

Production v6 GNNs consume `data.x = [ρ, τ, ss_type, sasa]`. FreeSASA (Å²) is a **Euclidean solvent-exposure** coordinate. Training `shell_corr_*` losses explicitly maximize `r(cone_depth, SASA)`, which hijacks hyperbolic depth and collapses expert geometry (uniform depth ~1.9, SASA-dominated gates).

SASA was never intended as a GNN inference feature. It belongs in **ingest persistence** and **downstream accessibility** (binding-site scan, residue filters), not in `node_emb`.

---

## 2. Goals

| ID | Requirement |
|----|-------------|
| GTI-01 | GNN `node_emb` input SHALL be `[ρ, τ, ss_type]` only for new lineage (`master_topology_three_vector`). |
| GTI-02 | SASA SHALL remain computed and persisted at ingest (`dim_residue.sasa`, `fact_ingestion_features.sasa`). |
| GTI-03 | SASA SHALL NOT appear in `data.x` for topology lineage; it SHALL be available on `data.sasa` side-channel when a graph is built. |
| GTI-04 | Binding-site scan SHALL use SASA from governed residue rows (or fpocket geometry), not from GNN `node_emb`. |
| GTI-05 | Legacy checkpoints (`node_dim=4`) SHALL remain loadable under `GNN_INPUT_MODE=legacy_four_vector` until retired. |
| GTI-06 | Cold-start training SHALL set `GNN_INPUT_MODE=topology_three_vector`, `node_dim=3`, and zero `shell_corr_*_sasa` weights. |
| GTI-07 | `fact_gnn_node_embedding.input_sasa` SHALL continue to be written for audit (value from `dim_residue`, not from `data.x[:,3]`). |

---

## 3. Non-goals (this spec)

- Retraining or migrating `tokyo_eyes_v6.pt` in place (requires new cold lineage).
- Removing FreeSASA from ingest compute (still required for accessibility).
- Replacing fpocket with SASA-only pocket detection.

---

## 4. Feature modes (two layers)

| Layer | Enum | Purpose |
|-------|------|---------|
| **Ingest / DB** | `FeatureMode.MASTER` | Four-vector persistence + P_FEATURE_01 gate unchanged |
| **GNN input** | `GnnInputMode` | What `node_emb` sees at forward time |

```
ingest:  MASTER four-vector → dim_residue + fact_ingestion_features
              ↓
graph:   GnnInputMode.TOPOLOGY_THREE_VECTOR → data.x [ρ,τ,ss], data.sasa (side)
              ↓
gnn:     node_emb(3) → hyperbolic lift → cone_depth (Poincaré)
              ↓
binding: join dim_residue.sasa at scan time for surface accessibility
```

---

## 5. Environment / contract

| Variable | Values | Default |
|----------|--------|---------|
| `GNN_INPUT_MODE` | `legacy_four_vector` \| `topology_three_vector` | `legacy_four_vector` |
| MLflow `feature_set` | `master_four_vector` \| `master_topology_three_vector` | from `gnn_feature_set_id()` |

Cold-start Makefile targets SHALL export `GNN_INPUT_MODE=topology_three_vector`.

---

## 6. Training loss contract (topology lineage)

| Loss / probe | Topology lineage |
|--------------|------------------|
| `cone_target_mode` | `tau_dehydron_rim` |
| `shell_corr_depth_sasa_weight` | `0` |
| `shell_corr_disc_sasa_weight` | `0` |
| `shell_floor_min_r_depth_sasa` | disabled |
| `probe_r_depth_sasa` | not used in topology lineage gates |
| `P_DEHYDRON_CONE_01` | `r(depth,τ) ≥ 0.15` only (SASA excluded) |

---

## 7. Binding-site accessibility (follow-on)

When ranking `fact_cryptic_site` candidates, optionally filter or score by mean `dim_residue.sasa` of member residues (surface-accessible threshold). fpocket remains primary geometry; SASA is a biochemical exposure check.

---

## 8. Acceptance

1. With `GNN_INPUT_MODE=topology_three_vector`, `data.x.shape[1] == 3` and `data.sasa` is populated.
2. `make gate-p-feature-01` still passes (DB four-vector unchanged).
3. Cold-start checkpoint has `node_emb.weight.shape[1] == 3`.
4. `audit-dehydron-topology` passes without SASA dominating τ on topology lineage.
