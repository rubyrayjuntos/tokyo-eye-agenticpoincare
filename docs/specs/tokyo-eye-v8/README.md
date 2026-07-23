# Tokyo Eye v8 — active trunk SSOT

**Status:** ACTIVE TRUNK (2026-07-23)  
**Architecture:** Equiformer/SE(3) frontend + hyperbolic spine + MoE — [`science/tokyo_eye/v8/`](../../../science/tokyo_eye/v8/)  
**Training:** [`experiments/training/v8/`](../../../experiments/training/v8/)  
**Checkpoints:** `checkpoints/v8/`  
**Do not open v7 for new work.** v7 is a different architecture and is **dead / archaeology**.

## Healthy restore pointers

| Alias | Path | Role |
|-------|------|------|
| `HEALTHY_V8_SPINE_CKPT` | `checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt` | Spine / MoE rematch bank |
| `HEALTHY_V8_AFFINITY_CKPT` | `checkpoints/v8/runs/tokyo_eye_v8_affinity_s1011_finetune_all/v8_affinity_best.pt` | CASF Core \(R\approx0.407\) affinity seal |

Python SSOT: [`experiments/training/v8/healthy_v8.py`](../../../experiments/training/v8/healthy_v8.py)

## Sealed affinity gate

[`data/gates/tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json`](../../../data/gates/tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json) — leak-proof affinity regression Pass. Does **not** buy cryptic-pocket / ΔΔG claims.

## Biology roadmap (ported from closed v7 track)

Open work that lived under v7 Hyp-MP probes is **re-homed here as v8 TODOs**. Prior v7 B0/B1 grades do **not** transfer — different architecture; re-prereg on v8 Θ.

→ [`biology-roadmap.md`](biology-roadmap.md) · stamp [`data/gates/tokyo_eye_v8_biology_roadmap.json`](../../../data/gates/tokyo_eye_v8_biology_roadmap.json)

## Investigation metrics

→ [`investigation-allele-epistasis-metrics.md`](investigation-allele-epistasis-metrics.md) (copied; Θ = v8 spine)

## Sprint 10.2 (affinity extension)

PRE-REG only: [`docs/superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md`](../../superpowers/specs/2026-07-23-tokyo-eye-v8-sprint10-2-multisite-ligand-prereg.md)

## Contract

- Production model id: `tokyo_eye_v8` (`TokyoEye-v8`)
- Runner: `science.tokyo_eye.v8.runner.TokyoEyeV8Runner` (ingest parity: see runner TODOs)
- `tokyo_eye_v7` / `TokyoEye.py` Hyp-MP line: **deprecated archaeology**

## Trunk cutover stamp

[`data/gates/tokyo_eye_v8_trunk_ssot.json`](../../../data/gates/tokyo_eye_v8_trunk_ssot.json)
