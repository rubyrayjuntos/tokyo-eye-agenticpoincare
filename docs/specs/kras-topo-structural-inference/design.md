# Tokyo Eye Topo-Structural Inference (v2.0)

**Date locked:** 2026-07-20  
**Status:** Locked for `FIX1_SPARSITY_CHAMPION_CKPT` validation  
**Scope:** KRAS G12D signaling rewiring & resistance prediction (Buffering Atlas)  
**Primary governance:** `experiments.training.v66.healthy_fix1.FIX1_SPARSITY_CHAMPION_CKPT` (epoch 48)  
**Make (model triad):** `make grade-v66-fix1-sparsity-kras-topo-matrix`  
**Make (edge ΔE rematch):** `make grade-v66-kras-topo-edge-four-quadrant`

---

## 1. Structural roster

### 1.1 Model triangulation (betweenness + switch-lock)

| Role | State | PDB | Designation |
|------|--------|-----|-------------|
| WT Inactive | GDP-bound | `4OBE` | Baseline / Anchor |
| Mutant Inactive | GDP-bound (G12D) | `4DSO` | Test Subject |
| Active Reference | GppNHp-bound (G12V) | `5VQ2` | Knockout calibration |

**Betweenness / switch-lock Pass/Fail uses `4OBE` / `4DSO` / `5VQ2`.**

### 1.2 Classical edge-ΔE closeout (four-quadrant rematch)

Inhibitor-free, allele-matched ON/OFF matrix — see [`edge-delta-four-quadrant.md`](edge-delta-four-quadrant.md).

| State | Wild Type | G12D |
|-------|-----------|------|
| Inactive (OFF) GDP | **4LPK** | **5US4** |
| Active (ON) GppNHp | **6GOD** | **6GOF** |

**Canonical residue indices (locked):** Leu**81**, Asp**114**, ILE**156** (terminal integration), ILE**163** (hub migration — completed), position **12** (allele).

---

## 2. Primary matrix (Phase A)

### 2.1 Learned betweenness proximity (model inference)

* **Method:** Forward-knockout `out_effect` per residue (z-norm safe). **Jacobian rankings prohibited.**
* **Alignment:** Shared deposited residue sequence numbers across the triad.
* **Pass:** \(\rho(\text{4DSO},\text{5VQ2}) > \rho(\text{4OBE},\text{5VQ2})\)  
  where \(\rho\) = Spearman of aligned knockout betweenness proxies.
* **Report (non-blocking):** relative ranks of residues 81 / 114 / 156 on each structure.

### 2.2 Localized conductance (completed)

* **Status:** **PASS** — `R_4DSO > R_4OBE` (ΔR ≈ +0.098 into ILE163).  
  Artifact: `checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json`

### 2.3 Classical edge symmetric difference (rewiring closeout)

* **Primary Pass graph:** Cα contact (8 Å) on the **four-quadrant** roster (`4LPK` / `6GOD` / `5US4` / `6GOF`).
* **Pass bars:**
  1. Mimetic inactive: \(|\Delta E(\text{5US4},\text{6GOD})| < |\Delta E(\text{4LPK},\text{6GOD})|\)
  2. Compressed mut span: \(|\Delta E(\text{5US4},\text{6GOF})| < |\Delta E(\text{4LPK},\text{6GOD})|\)
* **Make:** `make grade-v66-kras-topo-edge-four-quadrant`
* **Historical triad** dehydron-wrapper complementarity on `4OBE`/`4DSO`/`5VQ2` is **report-only** (5VQ2 = G12V; wrong allele for WT-ON mimicry). Full note: [`edge-delta-four-quadrant.md`](edge-delta-four-quadrant.md).

### 2.4 Switch-lock tether verification (structure-only)

* **Targets:** Asp12–Tyr32 (Switch I); Asp12–Gln61 (catalytic water displacement).
* **Method:** Cα–Cα distances during graph construction.
* **Cutoffs:** **12–32 ≤ 11.0 Å** (Switch-I breathing tolerance); **12–61 ≤ 10.0 Å**.
* **Pass:** Both tethers form in `4DSO` under those cutoffs and are absent or longer in `4OBE` (distance above cutoff **or** 4DSO strictly shorter by ≥ 1 Å).

---

## 3. Phase B probes (Buffering Atlas — not primary-gate blockers)

### 3.1 Dehydron wrapping probe (epi-structural)

Source-leak soft spots Gly19 / Leu81 / Asp114; wrapping index; inhibitor-bound stress redistribution (e.g. MRTX1133-like). Failure → **Buffering Logic diagnostic dump**, not hard fail of champion.

### 3.2 Hyperbolic signaling rewiring

Poincaré geodesics from On-State (`5VQ2`) to bypass nodes (PI3Kγ, Integrin αvβ3, EGFR feedback). Adjacent = resistance-plausible; distant = durable combo.

### 3.3 Topological coordinate mapping

* X: Frustration \(U_T\) — Source Leak Triad internuclear plasticity  
* Y: Relay betweenness \(C_B\) — G-domain shortest-path coupling

---

## 4. Uncertainty decomposition (observational telemetry)

Non-blocking. Failure does **not** gate training or champion status.

| Metric | Parameters | Role |
|--------|------------|------|
| Epistemic | Residues > 15 Å from functional sites | Core fold familiarity |
| Aleatoric | Residues ≤ 8 Å from position 12 | Pocket dynamics |
| Discrimination ratio | mean_ale(G12D-prox) / mean_ale(core) | SBIR S/N (target > 2.0 monitor) |
| Routing geometry | cone_depth rim vs tip | Observational |

---

## 5. Training governance (reference)

* \(\lambda_{\mathrm{sparse}} = 0.0075\)
* \(0.50 \le \mathrm{mean\_residue\_H} \le 0.90\)
* \(\mathrm{max\_share} < 0.45\)
* Save band is the governor — no dynamic λ release in this phase

---

## 6. Implementation map

| Artifact | Path |
|----------|------|
| Helpers | `science/dtie/common/kras_topo_matrix.py` |
| CLI grade (model triad) | `experiments/diagnostics/kras_topo_structural_matrix.py` |
| Make (model triad) | `grade-v66-fix1-sparsity-kras-topo-matrix` |
| Output (model triad) | `checkpoints/v66/diagnostics/routing_sparsity/kras_topo_matrix_4obe_4dso_5vq2.json` |
| CLI grade (edge ΔE rematch) | `experiments/diagnostics/kras_topo_edge_delta_four_quadrant.py` |
| Make (edge ΔE rematch) | `grade-v66-kras-topo-edge-four-quadrant` |
| Output (edge ΔE rematch) | `checkpoints/v66/diagnostics/routing_sparsity/kras_topo_edge_delta_four_quadrant.json` |
| Four-quadrant note | [`edge-delta-four-quadrant.md`](edge-delta-four-quadrant.md) |
| Section 13 | [`section-13-amendment.md`](section-13-amendment.md) |
