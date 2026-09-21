# Tokyo Eye — MLflow / Claude operator brief

> **Standing rule:** Prose without a failing test always loses to a convenient default.
>
> A written requirement that CI or the training harness will not fail is a suggestion, not a lock.

You are helping the **Tokyo Eye EQU** trunk from this repo (`/workspace`).
This file overrides dated design specs as your working instructions.

## Control plane (read these first)

0. **Fresh stitch (2026-09-20):** [`docs/history/session-handoff-2026-09-20-equ-gfit-marginal.md`](docs/history/session-handoff-2026-09-20-equ-gfit-marginal.md) — lite dehydron G_fit CLOSED_MARGINAL; next sealed card = **z_hyp SDRP-live G_fit** (`tokyo_eye_equ_wrap1_zhyp_g_fit_prereg.json`)
1. **[`docs/TOKYOEYE_ARCHITECTURE_SSOT.md`](docs/TOKYOEYE_ARCHITECTURE_SSOT.md)** — front door (intended stack; no numeric pins)
2. **[`docs/superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md`](docs/superpowers/specs/2026-09-16-tokyo-eye-v8-freeze-addendum.md)** — sole owner of numbers / signed §2.x
3. `docs/PROJECT_HUB.md` — stitch only; not machine truth
4. Gate JSON under `data/gates/tokyo_eye_equ_*.json` and `data/gates/tokyoeye_mlflow_ssot.json`
5. Code under `science/tokyo_eye/`
6. Live MLflow at `http://localhost:5000` (in-container `http://127.0.0.1:5000`)
7. Harness: `experiments/training/v8/run_v8_experiment.py` + `science/tokyo_eye/v8/assembly_gate.py`
8. EQU enforcement: `docs/ENFORCEMENT_MATRIX.md` § TokyoEye training assembly

**Specs under `docs/superpowers/specs/` are evidence / history only — not the open roadmap** (except the freeze addendum for pins).

## Live identity

- Product: **`TokyoEye`** / operator line **Tokyo Eye EQU**
- `@champion` v5 affinity = lesson-only for EQU geometry — do not init from it
- `@experimental` = staging bytes — not biology seal
- Registry v6 = DRAFT metadata only

## Hard stops

- No promote / alias moves
- No silent hyperparameter or label retune
- No v6/v66/v7/Chem-MVP as trunk
- No wrap=19 biology numbers
- No Stage-2 SDRP / C1 loader change (C1 selection revoked; incidence-like)
- Do not invent C1′ to beat post-hoc G7/G8
- No SE(3)-lite / stub as governed frontend (assembly gate fails closed)

## Closed / parked evidence (do not re-run as “next”)

| Item | Status |
|------|--------|
| Architecture SSOT + assembly ENFORCED | `tokyo_eye_equ_architecture_ssot.json` — pool + pure_hyp + deps |
| Non-claim disposition | `tokyo_eye_equ_nonclaim_disposition.json` — defect A CLOSED (wrap), B OPEN (leakage) |
| pure_hyp full-spine | PASS (eval-only caveat) |
| ε decay-floor full-spine | stamp said `COLLAPSED_POST_FLOOR`; **A0 relabel `NEVER_LIVE`** |
| MoE/ε card (step 1) | **CLOSED:** R=`NEVER_LIVE` → A=`PASS` ablated. MoE-on unhealthy |
| wrap=1 biology rescore | DIAGNOSTIC — weak dehydron; SDRP degenerate |
| SDRP Stage 1 / G7/G8 | PARKED `SDRP_PARKED_DIAGNOSTIC_ONLY` |
| wrap=1 dehydron LOSO on SE(3)-lite | **CLOSED_MARGINAL** — E=200 underfit-shaped; E=400 locked binary `DECAY_UNSETTLED` (1MBN 0.809 / 1LYZ **0.799** / 1BG1 0.824). Scientific read: marginal, not clean. No third lite prereg. Arm B not lean-able either way. Close: `…_G_fit_E400_close.json`. B-era pins (harness `8c7e2be2…` + script `34b75807…`) **dual-orphan / unverifiable class**. |
| wrap=1 z_hyp SDRP-live G_fit (lite) | **INCONCLUSIVE_UNDERFIT** — scoped: capacity miss under **cold-lite / 400 / MoE-ablated** only (not architecture ceiling). G_grad_spine PASS at scale; locked G_fit_train FAIL. **Label ceiling:** maj≈0.85–0.96 → max lift≈1.04–1.18 so bar 1.30 was STRUCTURALLY_UNREACHABLE. **Typed second gap:** `rel_bias`/`gamma`/`beta` near-dark while spine NZ. Stamp+read: `tokyo_eye_equ_wrap1_zhyp_g_fit_result.json`. Open agenda in `tokyo_eye_equ_next_experiment_sealed.json` `post_zhyp_g_fit_2026_09_21`. |

## ACTIVE trunk (after assembly ENFORCED)

Theory isn’t showing because routing + supervision + protocol are wrong — not because “need more epochs on the old path.”

1. **Assembly gate ENFORCED** — default `--frontend equiformer_pool`; stub requires `--allow-off-path-frontend`; live `pure_hyp_pass`; pool deps constructible; claim-bearing third leg (`--claim-bearing-biology` → non-leak + `--log-biology-grad-sources`). Tests: `tests/v8/test_assembly_gate.py`.

2. **Next sealed card only** (see `tokyo_eye_equ_next_experiment_sealed.json`):
   - MoE eval-utilization / §2.6 on **Equiformer pool only** (`tokyo_eye_equ_moe_verify_on_pool.json`), **or**
   - Hyperbolic biology with non-leaking target + loss on `z_hyp` + `--log-biology-grad-sources` + `--claim-bearing-biology`.

3. **SE(3)-lite pilots** — wrap=1 dehydron LOSO G_fit **CLOSED_MARGINAL**. z_hyp SDRP-live G_fit **INCONCLUSIVE_UNDERFIT** (wiring fixed; capacity bar not cleared at 400 steps). Stop lite curve extensions without a new sealed card. Do not lean on Arm B as FAIL_NO_SIGNAL from lite alone.

4. **Do not** open another cold stub diagnostic as trunk science. Defect B (edge_type → dehydron leakage) remains open — wrap AMEND did not fix it; claim-bearing gate refuses leaking stacks.

5. Only after (2): MoE-on vs ablated under identical sealed wiring.

## Working rules

- Prefer gates + MLflow + code over specs.
- Numbers live in the freeze addendum; SSOT links, does not fork.
- Stamps → `data/gates/`. Scripts → `/tmp/` if `scripts/` unwritable.
- Short answers: verdict, evidence, next command.
- Ask before alias/Release/non-DRAFT register.
