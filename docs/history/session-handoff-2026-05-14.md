# Session Handoff — 2026-05-14

## What Was Accomplished This Session

### 1. TP53 Structure Inventory & DTIE Analysis
- Identified 2XWR (WT apo, 1.68Å) vs 4IBS (R273H, 1.78Å) as optimal pair
- Ran DTIE pipeline: λ₂ +5.1% (borderline pathological)
- R248 identified as deepest constitutive dehydron (ρ=5) across all proteins analyzed
- ESMFold surrogate test for R248W/R175H: FAILED (surrogates can't model destabilization)
- Contact mutant replication: R273C +7.1%, R280K +5.1% — Archetype IV validated

### 2. Framework Validation & Hardening
- KEAP1 Kelch artifact exposed: -47% was artifact of NRF2 peptide in graph → corrected to +8.0%
- CDK2 inactive→active: -29.2% (persisted to Aurora)
- SRC inactive→active: -35.4% (persisted to Aurora)
- Framework revised: λ₂ measures allosteric effect size, not pathology
- Constitutive dehydrons (ρ) elevated as primary finding

### 3. CGC Dehydron × COSMIC Cross-Reference
- 9 CGC genes scanned (TP53, KRAS, SPOP, DDX3X, KEAP1, PTEN, EGFR, BRAF, PIK3CA)
- Persisted to Aurora table `fact_dehydron_cosmic` (69 rows)
- Key finding: PIK3CA E545 (ρ=1), H1047 (ρ=5), E542 (ρ=4) — all extreme dehydrons

### 4. GNN v4 Full Hyperbolic Pipeline
- Diagnosed Phase 1 Poincaré embedding bug (raw Cartesian → exp map = broken)
- Implemented Gnnv4.py with angular_diversity_loss + neighborhood_consistency_loss + domain_separation_loss_2d
- Trained through multiple iterations:
  - First run: 1D collapse (PC1=0.99)
  - Angular run: PC1=0.87, Switch-I displacement validated (Δ=0.19, exceeds 95th pct null)
  - Stable run (production): PC1=0.827, disc inside ball, all hardening patches applied
- **Validated finding**: GNN detects KRAS G12D allosteric rewiring from ρ + connectivity alone

### 5. Tier 1 Hub Protein Scan
- Ran 9 new proteins through stable checkpoint (MDM2, NOTCH1, HIF1A, VEGFA, KDR, SMAD3, SMAD4, TGFBR1, MYC)
- 3/9 show 2D structure (NOTCH1, TGFBR1, MYC)
- All disc inside ball — model generalizes

## Production Checkpoints

| Checkpoint | Location | Use |
|-----------|----------|-----|
| v4 Stable (BEST) | `checkpoints_v4_stable/tokyo_eyes_v4.pt` | Production — disc stable, PC1=0.827 |
| v4 Angular Stage 2 | `checkpoints_v4_angular/checkpoint_stage_2.pt` | Validated displacement finding (SHA: 1563c8f3...) |
| v3 (legacy) | `data_science/sub_agents/dtie/robust_experts.pt` | Phase 2/4b structural physics only |

## Key Files Modified/Created

### GNN v4
- `DTIE_GNN_ORCHESTRATION/TokyoEyesv4/Gnnv4.py` — Full v4 architecture with all loss functions
- `DTIE_GNN_ORCHESTRATION/TokyoEyesv4/train_v4.py` — Training script with warm-start, backbone freeze
- `DTIE_GNN_ORCHESTRATION/TokyoEyesv4/phase1_witness_embedding_v4.py` — Hyperbolic witness selection
- `DTIE_GNN_ORCHESTRATION/TokyoEyesv4/phase3_witness_persistence_v4.py` — Hyperbolic TDA

### Findings Documents
- `docs/findings/2026-05-12-tp53-r273h-contact-mutant.md`
- `docs/findings/2026-05-12-tp53-structure-inventory.md`
- `docs/findings/2026-05-12-tp53-alphafold-surrogate-test.md`
- `docs/findings/2026-05-12-framework-validation-hardening.md`
- `docs/findings/2026-05-12-framework-revised-honest-assessment.md`
- `docs/findings/2026-05-13-gnn-poincare-embedding-bug.md`
- `docs/findings/2026-05-13-gnn-v4-kras-hyperbolic-differential.md`

### Frontend Visualization Data
- `frontend/kras_dehydron_data.json` — Full WT+G12D dehydron payload
- `frontend/kras_4obe_poincare_full.json` — Phase 1 Poincaré data (broken, legacy)
- `frontend/kras_4obe_disc_v3_gnn.json` — PCA of GNN projections
- `frontend/kras_wt_vs_g12d_stable.png` — Production visualization
- `frontend/4OBE_displacement_wt_g12d.pdb` — B-factor colored displacement map

## What Needs to Happen Next

### Immediate (next session)
1. **Visualize Tier 1 hub proteins** — run the same WT vs G12D style analysis on MDM2, NOTCH1, MYC
2. **Fix cone_depth constant** — the stable checkpoint still has cone_depth=constant because it warm-started from angular (which had the old BCE loss). Need one more training pass with the MSE cone loss from scratch or fine-tune the stable checkpoint with cone_coeff raised.
3. **Wire v4 into DTIE orchestrator** — replace Phase 1/3 imports, add gnn_runner v4 path

### Short-term
4. **Second seed training run** — confirm Switch-I displacement reproduces with different initialization
5. **Domain separation ramp** — the stable checkpoint has domain_sep_coeff=0.10 but domains aren't fully separated yet. Consider coeff=0.15 with the tighter projection penalty.
6. **Persist Tier 1 results to Aurora** — the hub protein scan wasn't persisted

### Medium-term
7. **SBIR figure generation** — WT vs G12D side-by-side disc with domain coloring
8. **Phase 3 v4 on richer geometry** — once PC1 < 0.80 consistently, H1 features may emerge
9. **Poincaré disc frontend** — wire the stable checkpoint's hyp_projections into the Möbius viewer

## Environment Notes
- Aurora PostgreSQL running at 127.0.0.1:5432 (local)
- All PDB files cached in `/tmp/dtie_pdb_cache/`
- Training outputs in `checkpoints_v4_stable/`, `checkpoints_v4_angular/`, `checkpoints_v4_cone_fix/`
- `DATASET_CONFIG_FILE=tokyoeyes_dataset_config.json` required for imports
- Aurora env vars: `AURORA_HOST=127.0.0.1 AURORA_PORT=5432 AURORA_DATABASE=tokyoeyes AURORA_USER=tokyoeyes AURORA_PASSWORD=localdev AURORA_USE_PASSWORD=true`

## How to Run the GNN v4

```bash
# Internal test (no training data needed)
uv run python DTIE_GNN_ORCHESTRATION/TokyoEyesv4/Gnnv4.py --internaltest

# Train from scratch
uv run python DTIE_GNN_ORCHESTRATION/TokyoEyesv4/train_v4.py \
    --pdb_dir /tmp/dtie_pdb_cache \
    --output_dir ./checkpoints_v4_new

# Train with warm-start from stable checkpoint
uv run python DTIE_GNN_ORCHESTRATION/TokyoEyesv4/train_v4.py \
    --pdb_dir /tmp/dtie_pdb_cache \
    --output_dir ./checkpoints_v4_next \
    --warm_start ./checkpoints_v4_stable/tokyo_eyes_v4.pt \
    --freeze_backbone_epochs 15

# Inference on a protein (example)
import sys, torch
from pathlib import Path
sys.path.insert(0, "DTIE_GNN_ORCHESTRATION/TokyoEyesv4")
from Gnnv4 import GOSPConeMapper
from train_v4 import load_protein_graph

checkpoint = torch.load("checkpoints_v4_stable/tokyo_eyes_v4.pt", weights_only=False)
model = GOSPConeMapper(node_dim=4, hidden=128, num_layers=6,
                       num_experts=4, projection_dim=64, hyp_proj_dim=2)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prot = load_protein_graph("4OBE", "A", Path("/tmp/dtie_pdb_cache"))
with torch.no_grad():
    output = model(prot["data"])
# output["x_routed_hyp"] — 128-dim Poincaré ball positions
# output["hyp_projections"] — 2-dim disc coordinates (inside ball)
# output["cone_depth"] — hierarchical depth (currently constant — needs fix)
# output["uncertainty"] — {"epistemic", "aleatoric", "total"}
```
