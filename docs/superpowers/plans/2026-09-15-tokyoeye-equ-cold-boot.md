# Tokyo Eye EQU — Cold boot implementation plan

> **For agentic workers:** execute task-by-task; science container for GPU; MLflow SSOT.

**Goal:** Train a geometry-healthy EQU checkpoint from MPtrj Equiformer bank + fresh hyp spine; stamp hygiene Pass/Fail; **do not** alias `@champion`.

**Spec:** `docs/superpowers/specs/2026-09-15-tokyoeye-equ-cold-boot-design.md` (APPROVED)  
**Disposition:** `docs/superpowers/specs/2026-09-15-tokyoeye-equ-champion-disposition-design.md` (APPROVED)

## Global constraints

- Init frontend: `checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt` → `hf/checkpoint/mptrj_gradient.pt` (HF `mirror-physics/equiformer_v3`). Verify sha256 `59c6c23573a3b05b347662f473209d1bb1ccb5b85f6624a8f87072e7c397addf` before train.
- Init spine: fresh (no load of `507d54…` / v5 / any `eqf_affinity_*` / C1 best).
- Equiformer **frozen** for entire boot.
- Geometry losses only. No affinity head.
- Device: `tokyoeye_science` CUDA. PDB dir: `/tmp/dtie_pdb_cache`.
- Pure hyp invariant: do not “fix” tangent shortcuts on this card.

## Locked PDB tables

### Train (8) — Stage-A small minus shock chains

Shock excluded (same spirit as C1 §4.1): `1BG1` STAT3, `2Z6H` β-catenin, `1IVO` EGFR ECD, `2SHP` SHP2.

| PDB | Chain | Gene / note |
|-----|-------|-------------|
| 1MBN | A | myoglobin |
| 1LYZ | A | lysozyme |
| 1F88 | A | rhodopsin (7TM — size watch on T1000) |
| 1HHP | A | HIV protease |
| 1TEN | A | fibronectin III (Ig-like) |
| 1UBQ | A | ubiquitin |
| 1TIM | A | TIM barrel |
| 4OBE | A | KRAS home (in loss) |

If `1F88` OOMs after one empty-cache retry → skip and log `n_skip`; abort run only if train set drops below 6.

### Probe (6) — one theme each (B0 holdout IDs; not in loss)

| Theme | PDB | Chain |
|-------|-----|-------|
| Ig-like | 1HNG | A |
| Lysozyme-like | 1ALC | A |
| Ubiquitin / β-grasp | 1A5R | A |
| TIM barrel | 1NAL | 1 |
| Globin | 2HHB | B |
| P-loop NTPase | 1GKY | A |

Plus home `4OBE` recorded at probe time (in-train, not a Pass by itself).

## Knobs (boot)

| Knob | Value |
|------|-------|
| Epochs | **40** |
| Probe every | **10** epochs + end |
| τ | `tau_start=0.70`, `tau_end=0.90` |
| wrap_max | **1** (Sprint-8 retune; set once) |
| LR spine | weight-map `lr_hyperbolic` |
| LR backbone | 0 (frozen) |
| MoE | Sprint-9 CV + quota defaults (`cv_coeff`, `moe_quota_floor=0.05`) |
| Seed | 0 |
| Pass bars | per cold-boot spec §6 (sat &lt; 0.50; spread ≥ 0.05; moe_load_min &gt; 0.05 on ≥4/6 probe) |

## Tasks

### Task 1: Manifest + sha gate
- Create `manifests/equ_cold_boot_v1.json` (train/probe roles).
- Script/check: refuse start if frontend sha ≠ pinned MPtrj sha OR if any forbidden ckpt path appears in argv.

### Task 2: Runner
- `experiments/training/v8/run_equ_cold_boot.py` (or thin wrapper on `run_v8_experiment` / C1-style loop).
- Freeze entire frontend; probe protocol; stamp writer `data/gates/tokyo_eye_equ_cold_boot.json`.

### Task 3: Tests
- Manifest disjoint train/probe; shock absent; sha guard unit test.

### Task 4: Execute on science CUDA
- `docker exec tokyoeye_science …`
- MLflow run name `equ_cold_boot_<YYYYMMDD>`; log ckpt artifact; **no alias**.

## Done when

Stamp exists with `hygiene_pass` true/false; MLflow run finished; hub next-step updated. Promote is **out of scope**.
