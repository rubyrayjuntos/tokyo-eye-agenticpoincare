# Spec: TokyoEye-v8 Sprint 10.2 — Multi-Site Ligand Conditioning

**Status:** PRE-REGISTRATION / OUTLINE ONLY — **implementation frozen**  
**Depends on:** sealed affinity closeout `tokyo_eye_v8_affinity_s1011_finetune_all` (Core \(R \approx 0.407\))  
**Does not overwrite:** `HEALTHY_V7_CKPT`, cluster30 manifest, `v8_biophys_s8` graph cache

---

## 1. Boundary & enforcement

* No learning-rate churn or metric optimization loops on CASF-2016 Core.  
* Protein graphs (R0–R5) remain frozen via the `v8_biophys_s8` token-hash contract.  
* Multi-ligand / fragment array inputs map to **distinct, independent R6 channels** (never merged into R0–R5).  
* Gate claims remain affinity-regression only until a separate biology pre-reg is approved.

## 2. Target architectural interventions (outline)

### 2.1 Multi-head asymmetric projections

Split `JointPocketAffinityHead` into \(H=4\) independent cross-attention heads so distinct geometric / chem signals can co-exist before the pocket-gated readout.

### 2.2 Soft distance-decay bias

Replace hard \(d \le 4.5\)Å binary R6 membership *inside the attention logits* with continuous Gaussian decay (contacts may still be prefiltered for sparsity):

\[
\alpha_{ij} = \mathrm{Softmax}_j\left(\frac{Q_i K_j^\top}{\sqrt{d_k}} - \gamma\, d_{ij}^2\right)
\]

\(\gamma\) is a positive scalar (learnable or fixed in pre-reg). Empty-neighbor soft-skip rules from 10.1 remain mandatory.

### 2.3 Transfer testing mappings

Evaluation pipeline for relative \(\Delta\)affinity under in-silico point mutations on held allosteric control sets (KRAS / SHP2) — **report-only** until labels and NO_LEAK mapping are pre-registered.

## 3. Explicit non-goals (this sprint)

* Cryptic apo→holo discovery Pass claims  
* Absolute \(\Delta\Delta G\)  
* Mol* production cutover from v7  
* MoE architecture changes / relation-count expansion

## 4. Approval gate

No code until this outline is expanded to a full design with frozen metrics, mutation pairs, and claim boundaries — then a separate implementation plan.
