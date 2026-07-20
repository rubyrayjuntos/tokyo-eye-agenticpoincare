# Fix-1 corpus expansion (post-restore)

**Status:** Active ladder after sealed Fix-1 SSOT  
**Parent:** [`README.md`](README.md)  
**MLflow experiment:** `tokyo-eyes-v66-fix1-expand`  
**Model package:** `gnn_lineage=v6.6` / `GOSPConeMapperV66` (governance lineage = new experiment)

---

## Docker roles

| Container | Role |
|-----------|------|
| **science** | GPU. Runs `launch_training`, diagnostics, ingest, GNN inference |
| **mlflow** | Tracking server `:5000` only (does not execute GPU trains) |
| **db** | `tokyoeye_dev` facts + separate `mlflow` backend DB |
| **agent** | LLM UX over governed data — read-only for this ladder |

Train via Make → `SCIENCE_RUN` → science container → logs to `http://mlflow:5000`.

---

## MLflow lineage

| Item | Value |
|------|--------|
| Experiment | `tokyo-eyes-v66-fix1-expand` |
| Root stamp | `data/gates/fix1_expand_mlflow_lineage_root.json` |
| Register | `make register-v66-fix1-expand-lineage` |
| Child `parent_run_id` | Resolved from resume `checkpoint_path` tag on root |

Do **not** attach expand children to legacy `tokyo-eyes-v66`.

---

## Phase ladder

| Phase | Corpus | Make | Init |
|-------|--------|------|------|
| P0 | Stage A-12 sealed | (done) + lineage register | sealed ckpt |
| P1 | feeler expand 23 | `train-v66-fix1-expand23-continue` | resume sealed |
| P2 | Stage A hold (23; 3CON/4GQB MASTER-blocked) | `train-v66-fix1-stage-a25-continue` | resume P1 champion |
| P3 | RAF1 mix (8 pathway + 5 Stage A anchors; MASTER-blocked off) | `train-v66-fix1-raf1-continue` | resume P2 champion |
| P4 | Biology grade | `grade-v66-fix1-biology` | eval only |

### Resilience Pass bars (P1–P2 expand)

- ep1 and final `disc_r_mean` ≥ 0.25
- final σ₂/σ₁ ≥ 0.80
- `probe_r_proj_depth` ≥ 0.95, `probe_r_depth_tau` ≥ 0.45
- no expert hard monopoly ≥ 0.50
- Grade: `PYTHONPATH=. python -m experiments.training.v66.grade_fix1_expand_resilience --run-dir …`

### P3 transfer bars (pathway mix)

Domain shift to RAF1 kinases relaxes angular-fill bars while keeping mature disc:

- same disc_r / τ / monopoly bars
- σ₂/σ₁ ≥ 0.75, `probe_r_proj_depth` ≥ 0.85
- Grade: `… --profile transfer`

---

## Standing rules

1. Never overwrite sealed run dir / `v66_healthy_sealed.pt`
2. Hyp-MP Fix-1 exclusive of role/chem multi-rel
3. Chem-MVP / HA / Path2 PARKED
4. Re-run `make gate-p-feature-01 CORPUS=…` before each new corpus
5. UI: http://localhost:5000 → experiment `tokyo-eyes-v66-fix1-expand`

---

## Ladder status (2026-07-20)

| Phase | Run | Result |
|-------|-----|--------|
| P0 | sealed + `tokyo-eyes-v66-fix1-expand` root | Pass |
| P1 | `fix1_s4_expand23_continue_v1` | Pass (expand bars) |
| P2 | `fix1_s4_stage_a25_continue_v2` (hold; 3CON/4GQB off) | Pass |
| P3 | `fix1_s4_raf1_mix_continue_v1` | Pass (`--profile transfer`) |
| P4 | held `epoch_066.pt` vs sealed on 4OBE | Pass (`Δ disc_r ≈ +0.027`) |

Promoted: `data/gates/fix1_expand_promoted_trunk.json`  
Biology: `checkpoints/v66/diagnostics/fix1_expand_biology/biology_pack_report.json`  

Never load bare `phase_12.pt` for biology grade (missing `training_config`).

### Next bet (locked 2026-07-20) — **implemented**

P2/P3 expand champions lost hub scaffolding (4OBE knockout ρ ~0.24–0.29 vs sealed **0.596**). Do **not** continue sparsity work from those runs. Resume from sealed with **mean-residue routing entropy sparsity**:

| Item | Value |
|------|--------|
| Make | `make train-v66-fix1-sparsity-sealed-continue` |
| Default `RUN_ID` | `fix1_s4_sparsity_sealed_continue_v1` |
| Resume | `v66_healthy_sealed.pt` only |
| λ / warmup | `0.0075` / `8` |
| Corpus default | Stage A-12 (`v6_corpus_stage_a_small_v1.json`); override `CORPUS=v6_corpus_stage_a_feeler_expand_v1.json` |
| MLflow | `tokyo-eyes-v66-fix1-expand` |
| Design | [`../routing-entropy-sparsity/design.md`](../routing-entropy-sparsity/design.md) |
| Plan | [`../../superpowers/plans/2026-07-20-routing-entropy-sparsity.md`](../../superpowers/plans/2026-07-20-routing-entropy-sparsity.md) |

Task 6 (GPU): **PARTIAL** (mean-H + hub + stability PASS; legacy H(f̄)≤1.21 FAIL by design).  
Next: `make train-v66-fix1-sparsity-confirm-continue` (resume `epoch_045.pt`, sparsity save gates).
