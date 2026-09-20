# Session handoff — EQU G_fit marginal close (2026-09-20)

**For the next agent:** read this, then [`CLAUDE.md`](../CLAUDE.md). Prefer gates + code + MLflow over dated specs.

**Operator:** Ray Swan  
**Thread closed on:** wrap=1 dehydron LOSO SE(3)-lite G_fit → **CLOSED_MARGINAL**  
**Do not:** open a third lite epoch prereg, retune the 0.80 bar, or re-litigate Arm B from lite alone.

---

## 1. Start here (control plane)

| Priority | Path | Role |
|----------|------|------|
| 1 | `CLAUDE.md` | Live operator brief (overrides dated specs) |
| 2 | `docs/TOKYOEYE_ARCHITECTURE_SSOT.md` | Intended stack; no numeric pins |
| 3 | `docs/superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md` | Numbers / §2.x pins only |
| 4 | `data/gates/tokyo_eye_equ_*.json` | Machine truth for cards |
| 5 | `science/tokyo_eye/v8/assembly_gate.py` + `tests/v8/test_assembly_gate.py` | ENFORCED assembly |

Specs under `docs/superpowers/specs/` are **evidence/history**, not the open roadmap (except the freeze addendum).

---

## 2. What this session changed

### Documentation / operator surface

| File | Change |
|------|--------|
| `CLAUDE.md` | Maxim + architecture SSOT pointers already present; **CLOSED_MARGINAL** row for wrap1 LOSO; ACTIVE item 3 now says stop lite curve work → **pool typed G_fit** |
| `docs/TOKYOEYE_ARCHITECTURE_SSOT.md` | (earlier in thread) front door for intended stack; SE(3)-lite pilot policy |
| `docs/ENFORCEMENT_MATRIX.md` | (earlier) EQU training assembly enforcement |
| `docs/training/MLFLOW_ASSISTANT.md` | Assistant paste updated for **E=400 CUDA** card (then superseded by close — do not re-run E=400) |
| `docs/PROJECT_HUB.md` | (earlier) stitch only; not machine truth |

### Gate stamps (authoritative outcomes)

| Gate | Status / meaning |
|------|------------------|
| `tokyo_eye_equ_wrap1_dehydron_loso_G_fit_result.json` | E=200 → `INCONCLUSIVE_UNDERFIT` (0.756 / 0.736 / 0.752) |
| `tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_prereg.json` | Pre-reg for E=400 (r-trajectory → 0.80 → plateau; three named outcomes) |
| `tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_result.json` | Locked binary: **`DECAY_UNSETTLED`**, `g_fit_pass=false` |
| **`tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_close.json`** | **Scientific close: `CLOSED_MARGINAL`** — stop lite; next = pool G_fit |
| `tokyo_eye_equ_wrap1_dehydron_loso_B_disposition.json` | **`CLOSED_MARGINAL_LITE_G_FIT`**; Arm B not lean-able either way |
| Per-fold E400 | `data/gates/wrap1_dehydron_loso/T_seed0_hold_*_G_fit_E400.json` |

**E=400 finals (seed 0, CUDA):** 1MBN **0.809** · 1LYZ **0.799** (miss by 0.0009) · 1BG1 **0.824**.

### Code / infra (supporting)

| Path | Change |
|------|--------|
| `science/tokyo_eye/v8/assembly_gate.py` | ENFORCED: pool default, pure_hyp, deps, claim-bearing third leg |
| `scripts/wrap1_dehydron_loso_g_fit.py` | G_fit runner; E=400 distinct stamps; E400 classifier; **CUDA hard-fail** (no silent CPU fallback) |
| `docker-compose.yml` | **`mlflow` service GPU reservation** (Assistant runs here; was CPU-only before) |

### Operational lessons already paid

- Assistant (`appuser` uid 999) vs host-owned stamps (uid 1000): disposition writes can `PermissionError` — host may need `chmod o+w` or apply disposition; script soft-fails to `*.UPDATED.json` sibling.
- Recreate mlflow after GPU compose change: `docker compose up -d --force-recreate mlflow`, then assert `torch.cuda.is_available()` inside the container.
- Cursor agent sandbox often **cannot** reach host Docker/GPU; operator runs compose on the host; Assistant runs training inside `tokyoeye_mlflow`.

---

## 3. How to think going forward (discipline)

### Standing rule

> Prose without a failing test always loses to a convenient default.

### What G_fit is for (keep it boring)

One yes/no: **is the harness sound enough to trust a negative from Arm B?**  
It is **not** a venue for geometric decay ceilings, ratio archaeology, or hunting scheduled-event wiggles in the loss curve. If the check needs that much interpretation, the check is underspecified or the result is **marginal** — more estimators do not turn marginal into strong.

### Locked binary vs scientific read

| Layer | E=400 |
|-------|-------|
| **Locked binary** | `DECAY_UNSETTLED` / fail 0.80 (bar **not** moved for 0.799) |
| **Scientific read** | **MARGINAL** — neither clean clear nor clean capacity fail |

Do **not** rewrite 0.80 → 0.799 after seeing the number.  
Do **not** treat 0.799 as “can’t fit” either.

### Escalation amendment (operator-approved)

E=400 prereg said: no pool under `DECAY_UNSETTLED` (stop for new prereg).  
**Amendment in the close stamp:** no third lite prereg; marginal is enough; next = **Equiformer-pool typed G_fit**. That is a stop/escalation read, not a coeff/bar retune.

### Cheap-rig lesson

SE(3)-lite bought real information: leaked dehydron fit is **not trivial** on that frontend. It did **not** buy a crisp binary. Further lite epoch modeling will not fix that.

### Arm B

S1/S2 failed on lite. **Do not** read as `FAIL_NO_SIGNAL` (biology no-signal) and **do not** treat E=200 underfit as decisive. Lite G_fit never cleared the ambiguity cleanly.

---

## 4. Immediate next work

**Next harness card (not yet sealed as a task packet):**  
Equiformer-pool **typed G_fit** — same folds (`1MBN:A`, `1LYZ:A`, `1BG1:A`), same threshold 0.80, seed 0, **`--frontend equiformer_pool`** (governed path; no `--allow-off-path-frontend`).

Design the bar to be **crude**: clear early and wide, or obviously not, within a fixed modest epoch budget decided **before** the run. Avoid another decay-modeling cycle.

**After a clean pool G_fit** (or a decisive pool miss), trunk sealed options remain in `tokyo_eye_equ_next_experiment_sealed.json`:

1. MoE eval-utilization on pool (`tokyo_eye_equ_moe_verify_on_pool.json`), **or**
2. Hyp biology with non-leaking target + loss on `z_hyp` + `--log-biology-grad-sources` + `--claim-bearing-biology`

Defect B (edge_type → dehydron leakage) remains **OPEN** — claim-bearing biology must not use the leaking stack.

---

## 5. Hard stops (do not reopen)

- Promote / alias moves / new model version rename
- Silent hyperparameter or label retune; silent bar nudge
- Third SE(3)-lite epoch prereg or more lite curve modeling as trunk
- Treat lite Arm B S1/S2 as pool biology evidence
- Wrap=19 biology; Stage-2 SDRP / C1 loader; invent C1′
- SE(3)-lite / stub as governed frontend
- v6 / v66 / v7 / Chem-MVP as trunk

---

## 6. Suggested first message for the new session

```text
Read CLAUDE.md and docs/history/session-handoff-2026-09-20-equ-gfit-marginal.md.
Lite wrap1 G_fit is CLOSED_MARGINAL (E400_close). Do not re-open lite curve work.
Next: draft/seal Equiformer-pool typed G_fit (same folds, 0.80, crude bar, fixed epochs) — then run via MLflow Assistant on CUDA.
No promote. No bar nudge. Arm B not lean-able from lite alone.
```

---

## 7. Key MLflow runs (E=400, experiment 11)

| Hold | Run id (prefix) | Final AUPRC |
|------|-----------------|-------------|
| 1MBN | `658413c6…` | 0.809 |
| 1LYZ | `c6e0afbb…` | 0.799 |
| 1BG1 | `7bdd742f…` | 0.824 |

Tracking UI: `http://localhost:5000` (compose service `tokyoeye_mlflow`).

---

*End of handoff. Prefer updating gates + CLAUDE.md over growing this file; if the next card seals, add one row to CLAUDE.md closed table and leave this as history.*
