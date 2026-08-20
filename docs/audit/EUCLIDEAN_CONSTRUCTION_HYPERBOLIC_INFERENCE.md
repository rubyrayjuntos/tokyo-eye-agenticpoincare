# Euclidean construction vs hyperbolic inference

**Date locked:** 2026-07-19  
**Status:** **SUPERSEDED for Tokyo Eye v7** (2026-07-21). Remains historical / descriptive for **frozen v6.x** compare-only trunks.  
**v7 SSOT:** [`docs/specs/tokyo-eye-v7/README.md`](../specs/tokyo-eye-v7/README.md) — Hyp MP primary; Euc 3D = reference; production module `science/tokyo_eye/TokyoEye.py`.  
**Parents:** Tokyo Eye premise (dehydron/leak → pathway → pocket); [`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](DISC_PROJECTION_NOT_TRUNK_PROXY.md); chem-MVP / feeler MP map (parked)

---

## One-line lock (v6.x historical)

**Euclidean space builds who can talk (structure, chemistry, typed graph, message-passing trunk). Hyperbolic space defines which sites are pathway / hub / leak–salient under Tokyo Eye inference. Disc projections are human views — never sole evidence for relational biology.**

> **v7 correction:** Communication / transport is hyperbolic (Möbius neighbor MP). Euclidean remains 3D structure/chemistry reference. The “do not move MP into hyperbolic” non-goal below applied to chem-MVP-era sequencing and is **not** the v7 trunk policy.

---

## Split (normative)

### A. Euclidean construction (pre-lift)

| Allowed / required | Examples |
|--------------------|----------|
| Coordinates & chemistry | Heavy atoms, Cα, N, O; bonds; packing contacts |
| Physics node channels | ρ, τ, ss; optional SASA / future chem features on nodes |
| Graph ontology | Residue nodes; typed edges (role, chem, future PPI / unit edges) |
| Message passing | SE(3) / multi-rel MP → trunk **`encoder_h`** |
| Classical graph GT (diagnostics only) | Betweenness on contact graph — **not** primary allosteric claim |

**Meaning:** Euclidean answers *communication capacity* (neighborhood, chemistry, hop reach).

### B. Hyperbolic inference (post-lift)

| Allowed / required | Examples |
|--------------------|----------|
| Lift | `radial_head` / `angular_head`(`encoder_h`) → `expmap0` → **`x_hyp`** |
| Pathway / hub / leak biology claims | Depth / cone, ball geodesics, persistent leak loci, resistance alternatives, pocket-facing sites **scored after lift** |
| MoE / commitment | Routing on ball + physics board |
| Curvature | Learned at GNN; passthrough to hyperbolic jobs |

**Meaning:** Hyperbolic answers *Tokyo Eye salience* once the trunk exists.

### C. Disc / Klein / Lorentz / log-map (views)

| Rule | |
|------|--|
| Role | Audit UX and question-dependent lenses |
| Forbidden | Sole pairwise / relational / pathway evidence ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](DISC_PROJECTION_NOT_TRUNK_PROXY.md)) |
| θ in dashboards | Disc coordinate display — **not** proof that radian angle is an MP input |

---

## Explicit non-goals (locked for now)

1. **Do not** move the full graph + MP trunk into hyperbolic space as the next bet.  
   - Pros (aligned pathway metric, native hierarchy) do not outweigh chemistry fidelity, numerics, and loss of Euclidean audit.  
   - Revisit only as a **later hybrid** if post-lift pathway scoring is clean and long-range still fails *with* an adequate Euclidean graph ontology.
2. **Hyperbolic MP does not preserve functional units by itself.**  
   Units (domains, SSE, PPI) require **node/edge ontology** in construction — not curvature alone.
3. **Do not** treat Cα betweenness or pre-lift trunk ranking as the primary “allosteric pathway residue” claim. Those remain scaffolding / diagnostics unless paired with post-lift criteria.

---

## Labeling rule for diagnostics & claims

Every probe, gate, and paper-facing claim must be tagged:

| Tag | Space | Example |
|-----|--------|---------|
| `euclidean_construction` | Graph / features / MP | Role-edge attach, chem edges, `encoder_h` ER |
| `euclidean_diagnostic` | Classical GT on contact graph | Betweenness correlation (scaffolding) |
| `hyperbolic_inference` | Post-lift biology | Pathway residues, leak persistence, hyp geodesics |
| `disc_view` | Projection only | Viewer θ, Voronoi, Klein chords |

Promotion of Discovery Story biology (leak → pathway → pocket) rides on **`hyperbolic_inference`** (and interventions), not on `disc_view`, and not on Euclidean GT alone.

---

## Relation to existing trunk-vs-disc rule

[`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](DISC_PROJECTION_NOT_TRUNK_PROXY.md) forbids **disc-alone** relational claims and requires a paired non-disc measurement.

This lock **refines** the destination of biology claims:

- **Communication / “did MP change the representation?”** → still measure **`encoder_h`** (Euclidean trunk).  
- **Allosteric pathway / hub / leak salience** → measure **post-lift** (`x_hyp`, depth, geodesics, cone/leak artifacts) — not disc alone, and not Euclidean betweenness as the headline.

---

## Sequencing (next work)

1. Keep Euclidean MP redesign theory-first (nodes/edges/properties; dehydrons as channel, not monopoly). See [`../specs/graph-communication/design.md`](../specs/graph-communication/design.md) (**D1–D4 locked** 2026-07-19).  
2. Migrate pathway/hub **readouts and grades** to post-lift criteria (checklist audits).  
3. Only later: optional hybrid hyp mixing — fresh registration, not default.  
4. **Immediate next after D1–D4:** matched ablation design for heavy-atom–informed edges vs chem-MVP.

---

## Checklist (use before merging a probe or design)

- [ ] Is this claim tagged `euclidean_*` / `hyperbolic_inference` / `disc_view`?  
- [ ] If pathway/hub/leak: is the primary metric post-lift?  
- [ ] If disc numbers appear: is a non-disc pair required and reported?  
- [ ] If changing MP: does the design change Euclidean construction only, without pretending units come “for free” from the ball?
