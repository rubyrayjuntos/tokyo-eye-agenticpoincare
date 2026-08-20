# Tokyo Eye v7 Hyp MP — first forward smoke (pre-registration)

**Status:** LOCKED before smoke run  
**Date:** 2026-07-21  
**Make:** `make grade-v7-forward-smoke`  
**Artifact:** `checkpoints/v7/diagnostics/tokyo_eye_v7_forward_smoke.json`  
**Stamp:** `data/gates/tokyo_eye_v7_forward_smoke_prereg.json`

## Pass

1. `TokyoEye(hyp_mp_primary=True)` constructs and forwards.  
2. `audit_trail.hyp_mp_primary` is true.  
3. `cone_depth` finite; curvature from `model.curvature` (learned `log_c`) — not a hardcoded pin.

## Non-goals

- Biology Pass vs v66 champion (later pre-reg)  
- Editing `science/dtie/`  
- S4 euc-conv-on-hyp-edges
