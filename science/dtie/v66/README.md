# V6.6 GNN lineage — standalone feeler fork

**Status:** Training-only experiment line (fully forked from v6 / v6.5)  
**Production:** Unchanged — do not promote or wire to ingest until a separate contract pass.

## Standalone lineage (CRITICAL)

- **No shims.** `experiments/training/v66/` is a real copy — never import
  `experiments.training.v6`.
- **No look-back.** `science/dtie/v66/` must not import `science.dtie.v6` or
  `science.dtie.v65` (model, loss, visualization, MoE, evidential are forked).
- Shared infra OK: `science.training.*`, `science.dtie.common.*`,
  `science.dtie.v5` primitives (e.g. `MobiusLinear`).

## Why v6.6 exists

v6.5 slim MoE + structural SSOT froze the backbone and bypassed learned disc/depth lift.  
v6.6 is a **clean cold-start** lineage for restoring full MP → geometry → MoE learning with a minimal loss stack and a simple expert-ban rule (share >45% → ban 1 epoch).

| Concern | v6 (frozen) | v6.5 | v6.6 (feeler) |
|---------|-------------|------|----------------|
| Model code | `science/dtie/v6/gnn/` | `science/dtie/v65/gnn/` | `science/dtie/v66/gnn/` |
| Training | `experiments/training/v6/` | (via v6) | `experiments/training/v66/` (standalone) |
| Checkpoints | `checkpoints/v6/runs/` | `checkpoints/v65/runs/` | `checkpoints/v66/runs/` |
| MLflow | `tokyo-eyes-v6` | `tokyo-eyes-v65` | `tokyo-eyes-v66` |
| Ingest runner | `V6GNNRunner` | not wired | **not wired** |

## Lifecycle rules

1. **No weight transfer** — feeler runs use `--no-warm-start` and refuse `--resume`.
2. **Learned path only** — `structural_disc_frozen=False`, `disc_layout_source=gnn_learned`. Never combine with `--slim-moe-structural-ssot`.
3. **Barcode deferred** — do not pass `--use-dehydron-barcode` on the first feeler.
4. **Do not promote** until `gospc_v66` / `v66_champion` exist in the contract catalog (separate change).
5. **Success signal** — inspect `checkpoints/v66/runs/<RUN_ID>/viewers/` after the run.

## Training commands

```bash
# Phase 1 feeler (20-epoch cold start, minimal losses, timeout@45%)
make train-v66-feeler RUN_ID=feeler_p1_v1 DEVICE=cuda

# Phase 2 continue (resume P1 best, unfreeze angular, soft routing)
make train-v66-feeler-p2 RUN_ID=feeler_p2_v1 DEVICE=cuda \
  RESUME=checkpoints/v66/runs/feeler_p1_v1/v66_best.pt

# Corpus expand (resume late P2 → Stage A 23-prot, same light recipe, 50 ep)
# 3CON/4GQB excluded until negative residue_id ingest is fixed. Prerequisites:
make ingest-corpus-master-features CORPUS=v6_corpus_stage_a_feeler_expand_v1.json
make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json
make train-v66-feeler-expand RUN_ID=feeler_expand_23_v1 DEVICE=cuda \
  RESUME=checkpoints/v66/runs/feeler_p2_v4_50ep/v66_phase2_12prot.pt

# Generic v6.6 launcher (standalone package)
make train-v66 RUN_ID=my_run STAGE=1 DEVICE=cuda NO_WARM_START=1
```

## Feeler recipe

### Phase 1 (cold start)
- Epochs: 20; freeze angular; no resume / no warm-start  
- Losses ON: balance, cone (**``rho_rim``** continuous — low ρ→rim / high ρ→center;
  not binary ``τ``), neighborhood, cone-depth anticollapse, disc occupancy, disc
  thickness floor, disc PC repulsion, proj violation  
- Losses OFF: evidential, angular/domain, shell, epistemic stack, routing floor/ceiling, BCE heads  
- **Role edges (training-only):** Replaces isotropic Cα contact MP with four
  relations — packing (ordered↔ordered), dehydron (underwrapped H-bonds),
  spoke (ordered↔disordered), ribbon (seq ±1…±4). Ordered/disordered from
  continuous ρ (≥τ), not binary τ. ``EquivariantConvMultiRel`` with 4 radial
  filters. Not persisted to Normalizer. 

### Phase 2 (continue)
- Resume P1 `v66_best.pt`; unfreeze angular; lr × 0.5  
- Light angular (0.10), softer balance (0.02), stronger occupancy/PC — still no evidential/shell/epistemic  
- Expert timeout remains `share>45% → ban 1 epoch`  

### Expand (corpus diversity, not new losses)
- Resume late P2 (`v66_phase2_12prot.pt`); corpus `v6_corpus_stage_a_feeler_expand_v1.json` (23 enabled; 3CON/4GQB off); 50 epochs  
- Same P2 coeffs + timeout@45% — no random expert ban, no floor/ceiling, no evidential stack  
- Success watch: σ₂/σ₁ / loop thickness, e1≠e3 geometry, routing_H drift without save-ceiling chase, e0 core niche  

## Champion (2026-07-11)

**Production inference / viewer SSOT for feeler expand 23-prot:**

`checkpoints/v66/runs/feeler_expand_23_p3_geom_v1/v66_best_disc.pt` (global ep 57)

| Metric | Value |
|--------|-------|
| σ₂/σ₁ | 0.875 |
| probe_r_proj_depth | **0.567** |
| r(d,ρ) | −0.961 |
| r(d,τ) | 0.749 |

**Do not continue P3 geom occupancy** from this checkpoint — full and half-stack
continues erode global `probe_r_proj_depth` (0.55→0.17) while local σ₂/σ₁ rises.

**Decision tree for next training:**

1. **Use champion as-is** for viewers / inference until a run beats it on **4OBE
   viewer + probe_r_proj_depth ≥ 0.45**.
2. **P4 rim fan-out** from champion (`make train-v66-feeler-rim-fanout`) — rim-only
   angular/PC2 spread, **zero disc occupancy stack**; targets KRAS spike without
   global collapse. Prior P4 from dbh_edges parent failed; P4 from p3_geom not yet
   validated at time of writing.
3. **No barcodes** on feeler until architecture coupling is fixed (edge/global
   barcode runs caused U-shape / teardrop).
4. **Probe guard** — feeler continues stop early if `probe_r_proj_depth` drops
   >0.08 below resume baseline for 2 consecutive epochs.

## Registry

Lineage metadata: `science/training/gnn_lineage.py` (`"v6.6"`).
