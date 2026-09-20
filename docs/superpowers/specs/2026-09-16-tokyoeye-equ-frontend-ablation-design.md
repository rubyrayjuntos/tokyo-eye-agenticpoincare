# Tokyo Eye EQU — Frontend Ablation (MPtrj vs random vs none)

**Gate ID:** `tokyo_eye_equ_frontend_ablation`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-16  
**Approver:** Bot (Ray sit-back; sequenced ahead of spread_hold)  
**Predecessor context:** geoopt_restore QUALIFIED; theme_restore official FAIL (best-by-loss); MPtrj bank sha `59c6c235…` confirmed = Materials Project crystal DFT (periodic inorganic), not biomolecular.  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

## 1. Intent

Settle whether “restore spine carries dehydron signal” is a claim about **downstream** layers vs **frozen MPtrj** as a useful geometry encoder. External/GOSP language must not imply the frontend is protein-physics-informed until this card says otherwise.

## 2. Honest framing (locked)

- MPtrj = periodic inorganic crystal relaxation (energies/forces/stresses), not peptide H-bonds / wrapping.
- Frozen frontend may still act as a generic SE(3) local featurizer — **untested**.
- Equivariance is architectural; MPtrj did not “teach” dehydrons.

## 3. Stages

### Stage 0 — no-train separability (cheap, first)

On Stage-A structures, with wrap_max=1:
- **Pos:** residues incident to ≥1 R2 dehydron edge
- **Neg:** residues incident to ≥1 R1 wrapped H-bond and **no** R2
- Features: frozen `EquiformerPoolFrontend` outputs `(s, v)` (concat, or s-only) **before** projector/attention/MoE
- Metrics (pre-registered):
  - Fisher / mean-centroid cosine separation
  - ROC-AUC of score = projection onto (μ_pos − μ_neg) in feature space (no fitted classifier beyond that direction)
  - Same metrics on **random-init same-arch** frontend (same seed) and on raw CA coords / trivial baseline
- Interpretation guide (not Fail bars for Stage 0 — diagnostic):
  - AUC ≈ 0.5 on MPtrj → frontend does not linearly separate wrapping classes; downstream does the work
  - AUC ≫ random baseline → frontend features carry some dehydron-relevant geometry even from inorganic pretrain

### Stage 1 — train ablation (after Stage 0 report)

Same theme AUPRC protocol / panels as theme_restore, short budget (pre-register epochs in pins):
1. frozen MPtrj Equiformer-pool (current)
2. frozen random-init same arch
3. no-frontend / identity CA stub (existing Stub path **only** as ablation arm, not sealed spine)

Gate: report theme mean AUPRC ± hygiene; **no** champion alias. Selection uses joint-gate-eligible snapshots (selection fix lands with this runner).

## 4. Non-claims

- Not spread_hold; not affinity; not Pearson; not corpus expand.
- Stage 0 does not QUALIFY biology Pass.
- Does not rewrite theme_restore official FAILED stamp.

## 5. Exit

- Stage 0 report → decide Stage 1 budget / whether MPtrj warm-starts remain default.
- Stage 1 → update operator default init policy in writing.


## 6. Stage 0 arm4 candidate pin (2026-09-16, post Stage0)

**Framing correction:** Stage 0 established *pretrain source* inertness (MPtrj ≈ random), not architecture failure. Equivariant decomposition still beats CA. Pivot to a new frontend is unjustified until a named protein-pretrained arm is probed on the same harness.

### Named candidate (only real loadable protein-structure SSL artifact evaluated)

| Field | Pin |
|-------|-----|
| Artifact | GearNet-Edge Multiview Contrast `mc_gearnet_edge.pth` |
| Source | Zenodo DOI 10.5281/zenodo.7723075 (also 7593637) |
| Arch | TorchDrug `GearNet` residue relational GNN + edge msg (GearNet-Edge) |
| Config | `input_dim=21`, `hidden_dims=[512]*6`, `num_relation=7`, `edge_input_dim=59`, `num_angle_bin=8`, `concat_hidden=True` |
| Native out | `node_feature` ≈ **3072-d** (6×512) SO(3)-**invariant** residue embeddings — not Equiformer `(s,v)` irreps |
| Domain | Protein structure SSL: multiview contrast / optionally distance·angle·dihedral·residue-type siblings on same Zenodo |
| Dehydron / H-bond? | **No.** Not Kabsch–Sander, not wrap/desolvation. “Trained on proteins” ≠ dehydron-aligned. Closer than Materials Project, same honesty bar. |

### Shape / spine implications

- **Stage 0 probe:** fair comparison is Fisher/AUC on **native** per-residue features (no forced map into 128+3). Same 14 structures, same R2 vs R1 labels.
- **Train spine / `RadialAngularProjector`:** **not** drop-in. Would need a new adapter (e.g. Linear 3072→128 + invented/zero `v`, or projector `scalar_dim` change). That is a separate card if Stage 0 arm4 wins by a CA-sized margin.
- **Rejected as arm4:** EquiformerV2 OC20 / any catalyst-materials ckpt (same domain failure as MPtrj). Local affinity/theme run ckpts (not independent pretrained frontends). DiffInt / SKALE (generative or project-specific, not residue encoders for this harness).

### Still open

No public checkpoint is known that is (a) loadable, (b) equivariant residue frontend, and (c) pretrained on H-bond / dehydron / desolvation geometry. GearNet is the strongest *named* “protein geometry SSL” arm available for a cheap Stage 0 test — not a claim that it is dehydron-aligned.
