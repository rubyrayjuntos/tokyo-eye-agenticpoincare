# Pre-Registration — Single-Case Cryptic-Interface Test
## Tokyo Eye (DTIE, structural physics) vs PeSTo on unbound elastase (9EST → 1FLE)

| | |
|---|---|
| **Status** | **LOCKED** — confirmatory portion frozen at commit below |
| **Author** | Ray Swan, Eidetix Bio |
| **Repo** | `tokyo-eye-agenticpoincare` |
| **Version** | 1.0 (frozen) |
| **Companion harness** | `experiments/diagnostics/evaluate_9est_prereg.py` |
| **Phase 1 lock** | `data/benchmarks/9est_1fle_interface_lock.json` |

> **Falsification-first framing.** This is a hand-picked *maximally favorable* case: PeSTo published it as a static failure it could only solve with a microsecond of MD, the partner is known, and the interface is known. So the test is asymmetric on purpose. If Tokyo Eye's static physics signal cannot recover the interface *here*, that is strong evidence against the thesis and the cheap result is a kill. If it can, that is **proof-of-concept only** — it licenses the full PPDB5/MaSIF benchmark, not a performance claim. n = 1 proves nothing about general accuracy.

---

## 0. Freeze mechanism and integrity clause

1. Fill every `🔒 LOCK` box below, then commit this file + harness. Record the commit hash and UTC timestamp in §0.1. That commit is the pre-registration.
2. After the first time any **Phase 3** (Tokyo Eye) output is viewed, the confirmatory portion is frozen: no edits to the score definition (§4.2), endpoints (§5), hypotheses (§6), thresholds, or label (§3).
3. Any change after that point is logged in §11 as a dated amendment with a reason and is reported as **exploratory, not confirmatory**.
4. The verdict is computed by `evaluate_9est_prereg.py` from the locked constants — not by eye. The script is the executable lock.

### 0.1 Freeze record

- Frozen commit hash: `57f81d8988ce033cbba81500e521efc01a2f03ee`
- Frozen UTC timestamp: `2026-07-02T13:58:31Z`
- Tokyo Eye / DTIE model commit under test: `lever_a_clean_slate_v1/v6_best_disc.pt`
- PeSTo version / webserver date under test: `PeSTo i_v4_1 local (/tmp/PeSTo, 2026-07-02)`

**Phase 1 completed (output-blind):** 2026-07-02 — chains verified, I_gold locked, τ_epi calibrated. No Tokyo Eye scores on 9EST viewed.

| Phase 1 check | Result |
|---|---|
| 1FLE chain E | ELASTASE (receptor) — verified from header |
| 1FLE chain I | ELAFIN (partner) — verified from header |
| 9EST chain A | Unbound elastase monomer |
| 1FLE↔9EST alignment | 240/240 residues, **100% identity**, zero offset |
| \|I_gold\| on 9EST | **26** interface residues |
| Eval universe \|U\| | **229** residues (present in both structures) |
| Unmapped gold | **0** |

---

## 1. Background and discriminating logic

PeSTo (Krapp et al. 2023, *Nat Commun*, [DOI](https://doi.org/10.1038/s41467-023-37701-8)) predicts interface residues from atomic coordinates with no physics parametrization. On static unbound porcine pancreatic elastase (PDB **9EST**), PeSTo predicts essentially no interface (ROC AUC ~0.56 per paper); recovery requires 1 µs MD revealing a loop conformational switch (cluster-center AUC 0.92). Unbound conformation is ~1.2 Å backbone RMSD from bound — a small global change that hides the interface from geometry-only prediction.

Tokyo Eye / DTIE thesis: a binding interface is a *latent physical property of the unbound static structure* — under-wrapped backbone hydrogen bonds (dehydrons) are pre-formed adhesiveness present before conformational switch. 9EST/1FLE is a clean discriminator.

---

## 2. Materials

- **9EST** — porcine pancreatic elastase monomer (chain A). Evaluation target. *Note: PDB is inhibitor-bound crystal, used as PeSTo static unbound proxy per published benchmark.*
- **1FLE** — elastase·elafin complex. Ground-truth interface source.
- **Chains (verified Phase 1):** `E` = elastase, `I` = elafin.
- **SASA**: Shrake–Rupley (freesasa or MDTraj), heavy atoms only.
- **Numbering**: 1FLE elastase (chain E) ↔ 9EST (chain A) reconciled by global alignment; identity 1.0, no offset (see lock JSON).

---

## 3. Ground-truth interface label (the 5 Å lock) 🔒

- Elastase residue is **true interface** if min **heavy-atom** distance to any elafin atom in **1FLE** is **≤ 5.0 Å**. Hydrogens excluded.
- Transferred to 9EST numbering → gold set **I_gold**. **\|I_gold\| = 26** (locked in `data/benchmarks/9est_1fle_interface_lock.json`).
- Evaluation universe **U** = elastase residues in both structures after alignment. **\|U\| = 229**.

**Locked constant:** `CONTACT_CUTOFF = 5.0 Å`, heavy-atom, min-atom-pair.

---

## 4. Predictors and scores

### 4.1 PeSTo (baseline)
Run PeSTo on **static** 9EST (protein–protein interface head). Record per-residue score ∈ [0,1].

### 4.2 Tokyo Eye / DTIE score 🔒 — **FROZEN**

**Primary score S1 (structural physics only — confirmatory):**
```
S1(res) = w_d · z(wrapping_deficiency(res)) + w_s · z(shell_boundary_proximity(res))
```
- `wrapping_deficiency` = dehydron signal (ρ < τ=13.0; higher = more under-wrapped).
- `shell_boundary_proximity` = disc radial position `disc_r` (outer-shell / boundary proxy from lever_a).
- `z(·)` = z-score across evaluation universe U.
- **Epistemic gate (hard):** residues with epistemic > `τ_epi` excluded from ranking (−∞).

🔒 **`w_d = 0.5`**, **`w_s = 0.5`**, **`τ_epi = 1.5896`**
*(τ_epi = 90th percentile epistemic on lever_a holdout 11QE/4OBE/1IVO/4MNE — no 9EST)*

**Secondary score S2 (exploratory):**
```
S2(res) = S1(res) + w_a · z(aleatoric_uncertainty(res))
```
🔒 **`w_a = 0.5`**

### 4.3 SASA baseline (mandatory control)
Rank by SASA on static 9EST (higher = more exposed). S1 must beat this or CONFIRM downgrades to PARTIAL.

---

## 5. Endpoints (locked) 🔒

- **Primary — recovery@k:** top-k by S1 all in I_gold → 1. **`K_PRIMARY = 3`** (PeSTo parity). Sweep k∈{1..10} secondary.
- **Secondary — ROC AUC:** S1 vs binary I_gold over U.
- **Control — SASA margin:** `ROC_AUC(S1) − ROC_AUC(SASA)`.

---

## 6. Committed hypotheses 🔒

- **H1:** On static 9EST, S1 recovers elafin interface where PeSTo does not — `recovery@3 = 1` **and** `ROC_AUC(S1) ≥ 0.80`, while PeSTo static fails (§7).
- **H0:** S1 no better than PeSTo static — `recovery@3 = 0` or AUC not above PeSTo static.
- **Refutation:** `recovery@3 = 0` **and** `ROC_AUC(S1) < 0.65`.

**Thresholds:** `0.80` = PeSTo bound-state median; `0.65` ≈ midpoint chance–bar; `k=3` = PeSTo parity.

**Control gate:** H1 only if SASA margin > 0.

---

## 7. Precondition gate (run BEFORE Tokyo Eye) 🔒

Run PeSTo on static 9EST. If **PeSTo `recovery@3 = 1`**, case is **VOID** — pick another PeSTo static-failure from the 20-complex MD set.

Paper reports static 9EST: ROC AUC ~0.56, no interface recovery — **precondition expected to pass**, must be verified empirically.

---

## 8. Decision rule 🔒

| Outcome | Condition (precondition passes) | Action |
|---|---|---|
| **CONFIRM** | `recovery@3=1` AND `ROC_AUC(S1)≥0.80` AND SASA margin > 0 | License PPDB5/MaSIF benchmark. n=1 only. |
| **PARTIAL** | `ROC_AUC(S1)∈[0.65,0.80)` OR (`recovery@3=1` but SASA margin ≤ 0) | Exploratory only. |
| **REFUTE** | `recovery@3=0` AND `ROC_AUC(S1)<0.65` | Record negative. |
| **VOID** | PeSTo recovers statically, or label/numbering fails | Pick another case. |

Verdict from `decide()` in harness — not by eye.

---

## 9. Confounds and controls

- **Exposure** — SASA baseline (§4.3, §8).
- **Epistemic blind-spot** — hard gate (§4.2).
- **Provenance** — S1 is physics-only; S2 exploratory with training-split caveat if 9EST in corpus.
- **Single case** — CONFIRM licenses benchmark; REFUTE is more informative.

---

## 10. Procedure with lock points

- **Phase 0 — Freeze.** ✅ Lock boxes filled; commit harness + this file.
- **Phase 1 — Data prep.** ✅ `evaluate_9est_prereg.py --phase1` → lock JSON.
- **Phase 2 — Precondition.** ✅ PeSTo i_v4_1 on static 9EST: `recovery@3=0`, AUC=0.404 — **PASS** (`data/benchmarks/9est_phase2_precondition.json`).
- **Phase 3 — Test.** ✅ lever_a forward pass on static 9EST chain A; channels recorded in Phase 4 artifact. Freeze tripped 2026-07-02.
- **Phase 4 — Evaluate.** ✅ `evaluate_9est_prereg.py --run` → **REFUTE** (`data/benchmarks/9est_phase34_results.json`).
- **Phase 5 — Record.** ✅ §12 filled below (2026-07-02).

---

## 11. Amendment log

| Date (UTC) | Section | Change | Reason |
|---|---|---|---|
| | | | |

---

## 12. Results (Phase 4–5 complete)

**Run:** `make prereg-9est-run` (2026-07-02T14:18:58Z)  
**Checkpoint:** `checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt`  
**Artifact:** `data/benchmarks/9est_phase34_results.json`  
**SASA control:** freesasa, heavy atoms only (chain A)

| Metric | PeSTo (static 9EST) | Tokyo Eye S1 | Tokyo Eye S2 | SASA baseline |
|---|---|---|---|---|
| recovery@3 | 0 | 0 | 0 | 0 |
| ROC AUC | 0.404 | 0.560 | 0.546 | 0.603 |
| \|I_gold\| | 26 | — | — | — |
| SASA margin | — | −0.043 | −0.058 | — |
| **Verdict (§8):** | — | **REFUTE** | — | — |

**`decide()` output:** REFUTE — static dehydron/shell signal does not carry this cryptic interface. Record the negative.

### 12.1 Secondary endpoints

S1 recovery@k sweep (k=1..10): **all 0** — no k recovers the full interface in the top-k set.

### 12.2 Interpretation (post-hoc, exploratory)

- **H1 refuted** on locked criteria: `recovery@3=0` and `ROC_AUC(S1)=0.560 < 0.65`.
- **SASA margin negative** (−0.043): S1 ranks *below* raw solvent exposure — interface signal, if any, is confounded by or weaker than exposure geometry on this static structure.
- **PeSTo precondition held** (static failure at k=3); Tokyo Eye S1 does not outperform PeSTo on AUC (0.560 vs 0.404) but fails the confirmatory bar and does not beat SASA.
- **PPDB5/MaSIF benchmark not licensed.** Per §8, pick another PeSTo static-failure case from the MD set for a future discriminator test, or pursue exposure-controlled scoring amendments via §11 only as exploratory follow-up.
