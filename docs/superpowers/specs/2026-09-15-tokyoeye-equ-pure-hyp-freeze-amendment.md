# Tokyo Eye EQU — Pure Hyperbolic Freeze Amendment

**Status:** APPROVED (operator partnership, 2026-09-15)  
**Amends:** `docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md` (FROZEN 2026-07-22)  
**Display lineage:** Tokyo Eye EQU  
**Why now:** Cold-boot sealed Fail exposed rim pathology; audit also found post-lift tangent Linear / tangent pool shortcuts in live `attention.py` / `affinity_head.py`. Without this amendment, “pure hyp” in operator speech is a lie relative to the code path.

---

## 1. One-line

From the **single Euclidean → hyperbolic lift** until the representation is **stored / used as a hyperbolic graph**, the pipeline **must remain in pure hyperbolic geometry**. **Einstein / Klein barycenters are allowed.** **Tangent-space Linear / pool / mix as a geometry substitute is forbidden.**

---

## 2. Why this belongs in the freeze (not a late patch)

The 2026-07-22 freeze already required:

- one lift → one hyp refinement SSOT  
- aggregation via Einstein midpoint / Klein — not Euclidean matmul on ball points  

It did **not** explicitly forbid `exp₀(W · log₀(z))` as a free Euclidean workspace for Q/K/V, output maps, residual mixes, or graph pools. That omission let vendor-style shortcuts ship while the prose still said “hyperbolic.”

**Operator disposition (2026-09-15):** this invariant is **first-class from the beginning of EQU trunk work**. Leaving it unattended falsifies the geometry claim even if Pearson / MoE look fine.

---

## 3. Contract (LOCKED)

### 3.1 Allowed

| Op | Notes |
|----|--------|
| Single lift | Equiformer scalars/vectors → RadialAngularProjector → `expmap₀` / `project_to_ball` **once** at the Euc→H boundary |
| Manifold distances | `d_ℍ` (Poincaré / equiv) for attention logits |
| Barycenters | Einstein midpoint / Klein (or equivalent **valid** hyp barycenter) for sparse aggregation |
| Gyrovector / Möbius | Ops that stay on the ball without treating log₀ as a free ℝⁿ feature space |
| Diagnostics | `log₀` / radius reads **read-only** for telemetry (PoincaréDiagnosticsEngine) — not trainable geometry substitutes |

### 3.2 Forbidden after the lift (until hyp graph storage / hyp heads that remain on-manifold)

| Pattern | Why veto |
|---------|----------|
| `z ↦ exp₀(W · log₀(z))` as Linear on ball points (Q/K/V, output, FFN) | Tangent Euclidean workspace masquerading as hyp |
| Tangent mean / sum pool then `exp₀` as the **graph** representation | Pool left the manifold; “hyp graph” is a reprint |
| Residual / mix in tangent then `exp₀` as the primary message path | Same |
| Third-party / vendor layers that silently leave the ball | External cheat path |
| Claiming “pure hyp” while any of the above remain in the EQU trunk forward | Language fraud |

### 3.3 Scope boundary

- **Affinity / evidential heads** that currently tangent-pool are **out of geometric Pass** until rewritten under this contract or explicitly parked as non-trunk.
- **Cold-boot / rim cards** do not waive this amendment. A rim Pass with tangent Linear still active is **not** a pure-hyp Pass.
- **No remediation theater:** after cold-boot Fail, do **not** patch the lying tangent spine and continue. Amend the freeze (done) and **start correctly** under a new pre-registered card. This amendment freezes the *law*; live violations mean `pure_hyp_pass=false` until a correct-start Pass.

### 3.4 Gate language (machine + human)

Any EQU train / promote stamp MUST record:

```text
pure_hyp_pass: true | false
pure_hyp_violations: [ ... paths or "none" ]
```

`pure_hyp_pass=false` **blocks** geometry trunk promotion even if MoE / sat / Pearson look good.

---

## 4. Audit snapshot at amendment time (evidence, not Pass)

Live violations known on 2026-09-15 (non-exhaustive):

- `science/tokyo_eye/v8/attention.py` — `_tangent_linear`, output `W_o(log₀(·))`, tangent residual mix  
- `science/tokyo_eye/v8/affinity_head.py` — tangent pool → FFN  

Einstein/Klein aggregation is **claimed** in module docs; Q/K/V path still uses tangent Linear. Amendment makes that a **Fail**, not a footnote.

---

## 5. Relationship to other cards

| Card | Relationship |
|------|----------------|
| Champion disposition (v5 lesson-only) | Unchanged; purity amendment does not revive v5 |
| Cold boot (sealed Fail) | Fail stands; next rim card must cite this amendment |
| Correct start (next) | New pre-registered card under amended freeze — **not** a remediation patch of the tangent spine |
| Rim / volume | Only as part of a correct-start design that already satisfies pure hyp — never a waive |
| Biology / PPI / pathway | Forbidden until geometry + pure_hyp Pass |

---

## 6. Sign-off

**Approver:** Ray Swan + Bot (partnership)  
**Date:** 2026-09-15  
**Freeze amendment ID:** `tokyo_eye_equ_pure_hyp_v1`

By this amendment: the 2026-07-22 freeze’s geometry claim is made **non-optional and auditable**. Pure hyp is a gate, not marketing.
