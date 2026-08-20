# Graph communication ontology (Euclidean construction)

**Date:** 2026-07-19  
**Status:** **D1–D4 locked**; `ha_edges_v1` ablation **PARKED / STOP** (Partial D4 + instrument_b no win) — stay on chem-MVP communication. Re-engage: [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md).  
**Parent lock:** [`../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)  
**Ablation SSOT:** [`ablation.md`](ablation.md)  
**Scope:** *Who can talk to whom* — nodes, edges, message meaning.  
**Out of scope here:** flow/hub/bridge *grades*, causality tests, hyperbolic MP, multimodal S2F.

---

## 1. Definition (locked language)

**Communication** = capacity for exchange on the protein graph:

- a **node** is a site that can hold state and send/receive messages  
- an **edge** is a permitted communication channel with a typed meaning  
- **message-passing** aggregates neighbor state along those channels into trunk `encoder_h`

Not synonyms: flow, causality, hub, bridge, resistance (nested; not identical).

---

## 2. Design principle (locked)

Dehydrons remain a **core / privileged channel**, not the **ontology of the graph**.

Tokyo Eye throughline still depends on underwrapping / leak physics, but communication must admit chemistry and geometry (and later thermal / PPI) so non–dehydron-local pathways can enter the trunk.

---

## 3. Current baseline (chem-MVP feeler) — matched control

| Layer | Today |
|-------|--------|
| **Node** | Residue |
| **Node state** | `[ρ, τ, ss]` → `node_emb` |
| **Channels** | packing, dehydron, spoke, ribbon + disulf/covale |
| **Geometry on edges** | Cα Δxyz + d; SE(3) SH on directions |
| **Heavy atoms** | Used to *compute* ρ / H-bond rules; not nodes |
| **Units / PPI / thermal edges** | Not first-class |

Any redesign must beat this recipe on **D4**.

---

## 4. Locked decisions (2026-07-19)

### D1 — Node census → **A: residue-only**

- Primary node = residue.  
- Heavy-atom nodes deferred.  
- SSE/domain parent nodes parked (not Path B redo).

### D2 — Channel set (v1)

| Channel | v1 status |
|---------|-----------|
| packing / geometry | **Mandatory** (refine with heavy-atom–aware scoring allowed) |
| dehydron / H-bond | **Mandatory** (privileged leak channel) |
| spoke | **Mandatory** |
| ribbon | **Mandatory** (open note: scaffolding vs true communication — do not drop without ablation) |
| covalent chem (disulf / covale) | **Mandatory** |
| thermal / fluctuation | **Deferred** — only after geometry/chem heavy-atom edge policy is live + D4 instrument exists |
| PPI / interface | **Deferred** — corpus-gated; required for full story later, not v1 mandatory |
| unit hierarchy | **Parked** |

### D3 — Heavy-atom policy → **edges only**

- Heavy atoms **define and score channels** (existence, geometry, chem-specific attrs).  
- Optional edge attrs may include pair-type stats (C–C, N–O, …) when registered in ablation.  
- **Not** atom-nodes in v1.

### D4 — Falsifier (locked sentence)

> On a pre-registered Stage A / KRAS holdout set, **heavy-atom–informed Euclidean communication** (D1–D3) must improve **post-lift** recovery of biologically registered pathway-relevant sites (allosteric and/or PPI-interface residues where annotated) vs matched **chem-MVP** role+chem baseline — measured with `hyperbolic_inference` metrics (locked in ablation: `cone_depth` precision@K / recall@K), **not** disc-alone and **not** Euclidean betweenness as the headline — without collapsing standing dehydron-rim / disc-occupancy feeler gates.

**Pass / fail detail:** [`ablation.md`](ablation.md) §5.2 (pre-registered 2026-07-19).

---

## 5. Theory sketch (still valid under locks)

### Nodes vs edges

| On **nodes** | On **edges** |
|--------------|--------------|
| Local physics state (ρ, τ, ss, …) | Channel type (relation id) |
| Optional local scalars | Euclidean Δxyz, d; heavy-atom–derived strength / pair chemistry |

Rule: **channel identity** on the edge; **site state** on the node.

### Reach

Hop depth × diameter still limits who can talk. Better communication = better channels / attrs (and later PPI), **not** hyperbolic MP as v1.

---

## 6. Sequencing

1. ~~Lock D1–D4~~ **done**  
2. ~~Matched ablation design~~ **done** — [`ablation.md`](ablation.md)  
3. ~~Part 0 construction checks~~ **PASS** (`…/ha_edges_v1/part0/`)  
4. ~~Implement + train + grade `ha_edges_v1`~~ **done** — D4 **Partial** (ΔP=ΔR=0); instrument_b no biology Pass  
5. ~~Close~~ **PARKED / STOP** (2026-07-19) — `make train-v66-ha-edges-v1` refuses; keep chem-MVP  
6. **Next:** chem-MVP re-engage — [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md) (measurement contract before any new lever train)

---

## 7. Open notes (ablation may resolve)

- Ribbon: true communication vs seq inductive bias?  
- Dehydron exclusivity vs packing: keep until ablated?  
- Ensemble: single conformer graph vs occupancy-weighted edges?
