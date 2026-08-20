# Chem-MVP re-engage

**Date:** 2026-07-19  
**Status:** PARKED / compare-only (2026-07-19 restore)  
**Active v6.6 trunk:** [`../fix1-s4-restore/README.md`](../fix1-s4-restore/README.md)  
**Geometry:** [`../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)

Chem-MVP and successors collapsed disc fill relative to the Fix-1 + S4 controlled 3d seed2 trunk. Keep checkpoints for historical compare; do not use as the default biology / routing parent. Makefile chem successor trains STOP unless `ALLOW_PARKED=1`.

---

## 0. Stop condition (no successor train yet)

**Do not** start a new chem-MVP architecture / graph / loss train until **both** are true:

1. This document’s **upgrade measurement contract** (§4) is accepted for the arm.  
2. Exactly **one** lever family from §3 is explicitly registered (hypothesis + Pass bars + matched baseline).

**Registered lever (2026-07-19, rev Top-K):** Euclidean reach — [`../v66_chem_MVP/ablation_euclidean_reach.md`](../v66_chem_MVP/ablation_euclidean_reach.md). **Status: CLOSED Fail** — manifold-sync `chem_mvp_euc_reach_manifold_v1` authoritative; cone enrichment Δ≈0. Prior `chem_mvp_euc_reach_v1` INVALIDATED (construction mismatch).

Pass bars **must** include trunk + knockout + post-lift (not packing-edge counts alone, not soft `route_H` alone, not disc-alone).

Forbidden until re-registered: hyp MP, atom-nodes, Path2 warm-start, Jacobian-on-z-norm Pass, HA packing resume, PPI/thermal as unmeasured “next.”

---

## 1. Parked substrate (compare-only)

| Item | Value |
|------|--------|
| Recipe | `role_edge_mp` + `chem_edge_mp`, Stage A-12, `topology_three_vector`, z-norm **off**, feeler P1 |
| Checkpoint | `checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt` |
| Chem-MVP ablation | **CLOSED `partial`, not promoted** — [`../struct-conn-typed-edges/ablation.md`](../struct-conn-typed-edges/ablation.md) |
| Acceptance | **PARKED** — not the default baseline for new biology work. Fix-1 sealed trunk is SSOT. |

### Locked trunks (never overwrite)

| Checkpoint | Role |
|------------|------|
| `chem_mvp_stage_a12_cold_v1` | Primary z-norm-off chem-MVP feeler |
| `chem_mvp_baseline_role_stage_a12_cold_v1` | Role-only matched control |
| `chem_mvp_znorm_stage_a12_cold_v1` | Matched z-norm-on |
| `chem_mvp_tau_abs_dist_stage_a12_cold_v1` | \|ρ−τ\| swap (not Path2) |
| `ha_edges_v1_stage_a12_cold_v1` | Compare-only (**PARKED**) |
| Stage A-12 cache | `graphs_38a6993d7a439aa4.pt` lineage |

---

## 2. Backlog inventory

### (A) Must close / decide before a new architecture train

| Item | Status | Decision |
|------|--------|----------|
| `ha_edges_v1` | Partial + instrument_b no win | **PARKED / STOP** — [`../graph-communication/ablation.md`](../graph-communication/ablation.md); Makefile refuses |
| Measurement contract | HA proved cone_depth-only insufficient | **This §4** — mandatory before next lever |
| Chem-MVP as substrate | CLOSED partial | **PARKED** — Fix-1 sealed trunk is active SSOT |

### (B) Leave parked (do not finish before rearch)

| Item | Why |
|------|-----|
| Path2 diam≤9 | ORPHANED (`PATH2_DIRECTIONALITY.md`) |
| Jacobian causal on z-norm-on | Defect workaround locked |
| Part B soft Pass / Δ_holds OR | Withdrawn; keep Partial |
| Containment diameter unlock | Ablation **FAIL**; not automatic next |
| Barcode P1 promote / Full / typed shared-bar | Mixed; register separately if revived |
| Chem-Full (`hydrog`/`saltbr`/`metalc`) | Separate ingest registration |
| PPI / thermal / unit hierarchy | Deferred/Parked (graph-communication D2) |
| GNNV7 purity/Gram / uncertainty deep-fix | Routing locked (a); reopen only on deliverable bar |
| Geometric angular prior as S1 win | Gates PASS; not biology Pass |
| v65 `dbh_*` | ARCHIVE |

### (C) Already closed (do not reopen)

- Chem-MVP physics gates + chem liveness; chem-pair probe Partial  
- Learned-flow “no directional long-range” triangulation  
- Jacobian-under-z-norm defect  
- Euclidean construction vs hyperbolic inference vocabulary  
- Disc ≠ trunk proxy  

---

## 3. Lever catalog (within chem-MVP lineage)

Knobs: [`science/training/config.py`](../../../science/training/config.py), builders [`science/dtie/v66/`](../../../science/dtie/v66/), MP [`equivariant_conv_multirel.py`](../../../science/dtie/v66/gnn/equivariant_conv_multirel.py).

### Graph / communication

| Lever | chem-MVP now | Knob | Constraint |
|-------|--------------|------|------------|
| Role + chem | ON | `--v66-feeler-lineage` + `--chem-edge-mp` | Lineage exit if off |
| Coupling | OFF | `--v66-feeler-coupling` | May dilute dehydron exclusivity |
| Containment | OFF | `--containment-edge-mp` | Diameter FAIL — needs **new** hypothesis |
| HA packing aux | PARKED | `--ha-edge-mp` | Do not resume as default |
| Spoke / ribbon / dehydron-angular scales | 1.0 | `*_edge_scale`, `dehydron_angular_scale` | Cheap dials |
| Dehydron exclusivity | ON | `--v66-feeler-no-exclusivity` | Rim risk |
| Node/edge barcode | OFF | barcode flags | Feeler deferred |
| Chem-Full / PPI / thermal | Not in lineage | ingest + registration | Deferred |
| `num_layers` / `hidden` | 6 / 128 | `TrainingConfig` | Hop vs oversmoothing |
| Hyp MP / S4 disc k-NN | OFF | exclusive w/ role | **Forbidden v1** |

### Trunk / features / lift

| Lever | Now | Note |
|-------|-----|------|
| Input `[ρ,τ,ss]` | ON | `GNN_INPUT_MODE` cache-keyed |
| Z-norm / \|ρ−τ\| | OFF on primary | Separate locked trunks |
| Gate | topology-only | Don’t drown gate in trunk |
| `geom_theta_prior` | OFF | Disc view; not MP proof |
| Curvature | learnable | Passthrough only |
| Biology grade | post-lift | Tag `hyperbolic_inference` |

### MoE / loss / corpus

| Lever | Now | Reality |
|-------|-----|---------|
| `num_experts=4`, balance 0.05 | Feeler P1 | **Hard loads collapsed** on chem_mvp (E3≥~50%; E1 starved) — soft `route_H` hid this |
| Fix-1 floors / gram / angular diversity | OFF | GNNV7 reopen bar only |
| Feeler P1 loss stack | ON | Coeffs free; crescent risk |
| Corpus | Stage A-12 | Expand 23 / RAF1 available; RAF1 ≠ PPI edges |

### Ranked upgrade families (after §4; not train yet)

1. ~~Measurement-first (§4 columns mandatory)~~ **done**  
2. **Hop / diameter — INVALIDATED (construction mismatch):** [`../v66_chem_MVP/ablation_euclidean_reach.md`](../v66_chem_MVP/ablation_euclidean_reach.md) (`chem_mvp_euc_reach_v1`; not SSE containment). Primary: manifold/loader reconcile before any governor redesign.  
3. Cheap channel dials — coupling + spoke/ribbon scales  
4. MoE hard-load health — only if product needs multi-expert  
5. Corpus diversity — feeler-expand / RAF1  
6. Chem-Full — after ingest registration only  
7. HA / Path2 / hyp MP — **out**

---

## 4. Upgrade measurement contract

Every successor arm vs `chem_mvp_stage_a12_cold_v1` must report **all** of the following (same frozen site list / K rule as HA where applicable). Generalize patterns from [`experiments/diagnostics/ha_edges_v1_instrument_b.py`](../../../experiments/diagnostics/ha_edges_v1_instrument_b.py).

| Column | Salience / method | Tag | Role |
|--------|-------------------|-----|------|
| Post-lift pathway enrichment | `cone_depth` P@K / R@K on frozen S; \(K=\max(10,\lceil 0.15 N\rceil)\) | `hyperbolic_inference` | Biology proxy (not sole if trunk/knockout contradict) |
| Trunk enrichment | ‖`encoder_h`‖₂ P@K / R@K + mean gap on same S | `euclidean_construction` | Did communication change representation? |
| Knockout out-effect | Mean ‖Δencoder_h‖ when zeroing S (and/or top hubs) | causal / communication | Residue↔residue influence proxy |
| MoE **hard** loads | Per-expert share (structure + corpus) | feeler health | Soft `route_H` alone is **not** enough |
| Feeler non-regression | Rim enrichment / eligibility floors as registered | feeler | Floor, not Pass |

### Forbidden as Pass evidence

- Disc-alone P@K  
- Euclidean betweenness headline  
- Packing edge-count deltas alone  
- Soft hold-count OR / soft Pass  
- Path2 / Jacobian-on-z-norm  

### Site list

Reuse frozen KRAS pathway set when grading 4OBE:  
`checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/site_lists/4obe_pathway_residues.json`  
(or a successor list filed **before** train with `frozen=true`).

---

## 5. Related SSOTs

| Doc | Role |
|-----|------|
| [`../struct-conn-typed-edges/ablation.md`](../struct-conn-typed-edges/ablation.md) | Chem-MVP CLOSED partial |
| [`../graph-communication/ablation.md`](../graph-communication/ablation.md) | HA PARKED/STOP |
| [`../hierarchical-containment-edges/ablation.md`](../hierarchical-containment-edges/ablation.md) | Diameter FAIL |
| [`../../audit/GNNV7_SUCCESS_CRITERIA.md`](../../audit/GNNV7_SUCCESS_CRITERIA.md) | MoE reopen bar |
| [`../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md) | Disc ≠ trunk |

---

## 6. Outcomes

| Date | Note |
|------|------|
| 2026-07-19 | Re-engage brief filed; HA parked; stop condition §0 locked |
| 2026-07-19 | Lever registered: Euclidean reach — [`../v66_chem_MVP/ablation_euclidean_reach.md`](../v66_chem_MVP/ablation_euclidean_reach.md); next = Part 0 |
| 2026-07-19 | Euc-reach cold train `chem_mvp_euc_reach_v1` graded under mismatched Part0 vs grade loaders → **INVALIDATED — Pipeline / construction mismatch** (retract CLOSED Fail / empty-lever-ignore; primary = manifold reconcile; see ablation SSOT + `MANIFOLD_RECONCILE.md`) |
