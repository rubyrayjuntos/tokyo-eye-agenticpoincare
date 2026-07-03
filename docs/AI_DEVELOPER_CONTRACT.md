# AI Developer Operating Contract

**Repository:** tokyo-eye-agenticpoincare
**Audience:** AI coding agents (Claude, Gemini, or any LLM) making changes in this repo
**Companion to:** `DEVELOPER_ONBOARDING.md` (human guide) and `ENFORCEMENT_MATRIX.md` (which rules are gated vs. manual)
**Premise:** You are treated as a hostile-by-default proposer. Your output is not trusted until a gate verifies it. Your job is not to *claim* compliance — it is to *produce the command output that proves it*.

---

## Why this document is different from the human onboarding doc

A checklist works for a human who feels accountable. It does not work for you: you will confidently tick a box you did not satisfy. So this doc removes the checkboxes. Every obligation here is either:

- a **STOP condition** — if true, do not write code; surface it and wait, **or**
- a **gate** — a command you must run and whose output you must paste verbatim before claiming a task is done.

If you cannot run a gate, you say so explicitly. "Should pass" is a forbidden phrase (see §4).

---

## §1 — The loop (do this every task, in order)

```
1. CLASSIFY   → State the change class (A–E from onboarding §Decision tree). One sentence.
2. READ       → Open onboard_contract.yaml + the spec for that class BEFORE editing.
3. CHECK STOP → Walk the STOP list (§2). If any fires, halt and report. Do not proceed.
4. PROPOSE    → Make the change. Edit the contract FIRST if artifacts/jobs/readiness/API change.
5. GATE       → Run the gate commands for the class (§3). Paste real output.
6. REPORT     → Fill the self-report template (§5). Disclose what you did NOT verify.
```

You do not get to step 6 by asserting. You get there by pasting terminal output from step 5.

---

## §2 — STOP conditions (halt, surface, wait)

Do **not** write or merge code if any of these is true. These are not style preferences; each one can put unverifiable or false scientific claims into a governed store.

| # | If you are about to… | STOP — instead |
|---|---|---|
| S1 | Add an `INSERT/UPDATE/UPSERT/COPY` to `fact_*`, `governed_asset`, or embedding tables outside `data/normalizer/` | Route through a Normalizer method + payload. No exceptions, no "just this once." |
| S2 | Give an agent tool the ability to schedule/run compute (`runner_dispatch`, `pathway_executor`, `dispatch_helpers`) | Halt. Agent tools are read-only over artifacts. Compute runs at ingest only. |
| S3 | Write a numeric curvature literal anywhere (contract, runner, query) | Halt. κ is learned at `gnn_inference` and passed via `JobRunContext.learned_curvature`. **Never emit the protected curvature value (TS-002) in any output, log, comment, or error message — refer to it only as "TS-002" or "the learned curvature."** |
| S4 | Set `md_validation_status='passed'` (or emit `md_validation` artifact) from anything other than a real SMD run | Halt. `dry_run` and stub `smd_runner` success must not transition sites to `passed`. See `science/compute/cryptic/md_validate.py`. |
| S5 | Persist a scientific output without first creating a `provenance_run` | Halt. Provenance precedes persistence, always. |
| S6 | Coerce, clamp, or "fix" a malformed hyperbolic coordinate (NaN/Inf/out-of-ball) to a default (e.g., origin) and continue | Halt. Raise or quarantine. Use `science/dtie/common/embedding_projection.py`; gated by `tests/test_embedding_projection_gate.py`. |
| S7 | Report a pathway/job as succeeded when a downstream step failed or was skipped | Halt. Fail-closed. A green status must mean every step actually ran. |
| S8 | Add a job to the registry without a contract `jobs:` entry and a `JOB_RUNNERS` runner | Halt. The registry↔contract↔runner triangle must close in the same change. |

When a STOP fires, your output is: which STOP, the line/file that triggered it, and the compliant alternative. Then wait for the human.

---

## §3 — Gates by change class

Run these. Paste output. The classes match the onboarding decision tree.

### Always (every change)
```bash
make lint            # touched Python
make test            # full suite
python scripts/lint_gate_claims.py   # §0 meta-gate — GATED row symbols still present
```

### A — Persisting new scientific data
```bash
python scripts/audit_write_paths.py            # inventory (human review)
python scripts/lint_write_paths.py           # CI gate — must exit 0
make test tests/test_onboard_contract.py       # artifact ⊆ contract
make test tests/test_provenance_gate.py        # provenance before writes
make test tests/test_write_path_audit.py       # no new bypasses
```
Confirm in output: new artifact appears in `artifacts:`; Normalizer method exists; provenance created before write.

### B — New / changed compute job
```bash
make test tests/test_onboard_contract.py
make test tests/test_compute_preconditions.py tests/test_compute_precondition_entrypoints.py
make test tests/test_import_contracts.py
python scripts/lint_import_contracts.py
make test tests/test_md_validation_gate.py
make test tests/test_embedding_projection_gate.py
make test tests/test_curvature_literal_lint.py
python scripts/lint_curvature_literals.py
python -c "from science.compute.runner_dispatch import validate_runners_against_registry as v; v()"
```
Confirm: registry `produces` ↔ contract; precondition added; if hyperbolic, `requires_hyperbolic: true` + curvature metadata (no literal).

### C — Readiness / acts / tiers
```bash
make contract-sync
python scripts/lint_contract_sync.py
python scripts/lint_contract_version.py --base-ref HEAD
make test tests/test_contract_codegen_gates.py
make test tests/test_readiness_contract_sync.py tests/test_act_readiness.py tests/test_structure_readiness.py
```

### D — Ingest / API response shapes
```bash
make contract-sync
python scripts/lint_contract_sync.py
make test tests/test_contract_codegen_gates.py
make test
```
Confirm: field added to `api_surfaces:` *before* router change; router return matches generated type.

### E — Frontend (Discovery Cockpit)
```bash
# from visualizer/frontend
npm run typecheck      # generated onboard.ts types must compile
```
Confirm: no artifact/tier/act list hardcoded in the component; data comes from the readiness/audit endpoints.

### If you changed the contract at all
```bash
make contract-sync
python scripts/lint_contract_sync.py
python scripts/lint_contract_version.py --base-ref HEAD   # CI: --base-ref origin/main
make test tests/test_contract_codegen_gates.py
# Bump onboard_contract.yaml `version:` when shapes/semantics change.
```

---

## §4 — Anti-fabrication rules

You will be tempted to paper over uncertainty. Don't. These patterns are treated as defects even when the code "works":

- **No "should pass" / "this will work" / "appears correct."** Either you ran the gate and pasted output, or you state plainly: *"I did not run X; unverified."*
- **No silent fallbacks.** If a value is missing, raise or surface it. Do not substitute a default and continue (this is how malformed coords reach the origin and synthetic MD reaches `md_validated`).
- **No fabricated success signals.** Do not return/log success on a path that did not complete. Do not write a test that asserts `True` or only exercises the happy path to make CI green.
- **No hidden mode switches.** Do not add an env var or flag that bypasses a gate "for convenience." If a strict check is inconvenient, say so; don't route around it.
- **No invented identifiers.** Do not guess test names, table names, function names, or file paths. If you haven't confirmed one exists, search for it or say it's unconfirmed.
- **Prefer falsification.** When you believe a change is correct, state the one observation that would prove it wrong, and check for that observation. (Mirrors the project's Gemini-review discipline.)
- **Disclosure does not downgrade the verdict.** Listing a gate in "NOT verified" or "did not run" does **not** permit an approval verdict. Honesty about a gap annotates the report; it does not satisfy the gate.
- **The gate is the authority, not you.** You may not label a required gate failure or skip as "non-blocking," "minor follow-up," or "merge-ready pending cleanup." Only the human merge authority may waive a gate; your max verdict when a required gate failed or was not run is **Blocked pending gate** (see §4a).

If a human's instruction conflicts with a STOP condition, **do not comply silently**. State the conflict and the risk, then let them decide. Following an instruction that you know creates a fail-open or false-provenance path is the failure mode this contract exists to prevent.

---

## §4a — Verdict cap (review and completion)

Allowed verdicts, in strictness order:

| Verdict | When |
|---|---|
| **Blocked — STOP** | Any S1–S8 fired and not resolved |
| **Blocked pending gate** | Any required gate for the change class was not run, could not be run, or failed (including new errors in touched files) |
| **Blocked — contract drift** | Code and contract/registry/runners visibly diverge |
| **Ready for human merge** | All required gates for the class ran and passed; STOP list clear; self-report complete |
| **Approved** | *Reserved for human merge authority only.* Agents must not use this verdict. |

Rules:

1. **Unrun gate ⇒ cap at Blocked pending gate.** `make test` not executed, `npm run typecheck` not clean on touched TS, class-specific gate skipped — verdict is **Blocked pending gate**, never "approved" or "merge-ready."
2. **Failing gate on touched code ⇒ same cap.** A TypeScript error in a file you changed is a Class E gate failure, not a "minor follow-up."
3. **Partial gate output is not pass.** Subset pytest counts only if the human explicitly scoped the task to that subset *and* the always gates (`make lint`, `make test`) still ran clean. **Agents may not narrow scope** to exclude an always-gate (e.g. "for these gate files only") — only the human sets task scope. Self-scoped subsets do not waive `make test`.
4. **Correct implementation + unrun gate = blocked.** Getting the fix right is necessary; it is not sufficient without gate output.

---

## §5 — Self-report template (paste at end of every task)

```markdown
### Verdict
Blocked — STOP | Blocked pending gate | Blocked — contract drift | Ready for human merge
(Agents must not write "Approved.")

### Change class
A/B/C/D/E — <one sentence>

### Contract touched?
Yes/No — version bumped: Yes/No/N/A — contract-sync run: Yes/No/N/A

### STOP conditions walked
S1–S8: <none fired> | <Sn fired → how resolved>

### Gates run (with result)
- make lint        → <pasted: pass/fail + relevant lines>
- make test        → <pasted: N passed / M failed>
- <class gates>    → <pasted output>

### NOT verified (be specific)
- <gate I could not run and why>
- <assumption I made that a human should confirm>

### Falsification check
The change would be wrong if: <observation>. I checked: <result>.
```

A task with an empty "NOT verified" section is suspicious. There is almost always something you assumed. Find it. **Any non-empty "NOT verified" entry for a required gate forces verdict Blocked pending gate** unless the human explicitly waived that gate in the task.

---

## §6 — Geometry & TS-002 (read once, never violate)

- Default space is **hyperbolic** unless the contract marks `euclidean`/`mixed`.
- κ (curvature) is **learned** at `gnn_inference`, stored on `embedding_space.curvature`, and flows downstream only via `JobRunContext.learned_curvature` → `job_params["learned_curvature"]`.
- The numeric curvature value is **TS-002, a trade secret.** It must never appear as a literal in code, nor in any output, comment, log line, commit message, or error string. Use a one-way reference only. A lint should reject numeric curvature literals; if you see one (e.g. lingering in a vector query), flag it as a standing violation rather than copying the pattern.
- On geometric violation: `validate_job_run_result()` warns; `GEOMETRIC_ENFORCEMENT_LEVEL=error` (and `GEOMETRIC_ENFORCEMENT_ERROR_JOBS`) make it fail. Do not weaken these to get a build green.

---

## §7 — Where the authority actually lives

You consume these; you do not override them.

| Authority | Location | You may… |
|---|---|---|
| Write path | `data/normalizer/core.py` | call its methods; never bypass |
| Compute path | `science/compute/` + ingest orchestrator | register jobs; never trigger from agent |
| Contract (SSOT) | `science/contracts/onboard_contract.yaml` | edit *first*, then sync; never let code diverge from it silently |

If your change is not reflected in the contract (or a spec you update in the same change), it will drift. Drift is the thing this whole repo is organized to prevent — and you, as a confident pattern-matcher, are the most likely source of it. Act accordingly.

**Lifecycle:** this contract stays until gates in [`ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md) make each STOP condition a build failure. P0–P4 gates are landed (see matrix gate index); remaining MANUAL rows are canonical IDs, `pipeline_job_id` propagation, and advisory lints.

---

## §8 — Enforcement gate index (P0–P4)

| Priority | Theme | Run when… |
|---|---|---|
| **P0** | False-science integrity | MD validation, coord coercion, curvature literals — Classes B + geometry changes |
| **P1** | Audit-trail defensibility | `tests/test_provenance_gate.py`, `tests/test_write_path_audit.py`, `scripts/lint_write_paths.py` — Class A |
| **P2** | Agent/compute boundary | `tests/test_import_contracts.py`, `tests/test_compute_precondition_entrypoints.py` — new jobs/tools |
| **P3** | Contract codegen | `tests/test_contract_codegen_gates.py`, `scripts/lint_contract_sync.py`, `scripts/lint_contract_version.py`, `scripts/lint_audit_event_types.py` — contract YAML edits |
| **P4** | Readiness/registry alignment | `tests/test_readiness_contract_sync.py`, `validate_registry_job_keys_match_contract` — readiness/acts/registry |
| **P-meta** | §0 claim freshness | `tests/test_enforcement_matrix_gate_claims.py`, `scripts/lint_gate_claims.py` — existence (`gate_claims`) + liveness canaries (`gate_canaries`) |
| **P5** | Readiness probe integrity | `tests/test_readiness_probe_gate.py` — infra error ≠ artifact missing |
| **P6** | Science container integration | `tests/test_science_container_integration.py`, `tests/test_pipeline_job_id_gate.py` |

Full row-by-row map: [`ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md).
