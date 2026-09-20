# MLflow Assistant — Codex + Claude Code

The Assistant runs inside `tokyoeye_mlflow`. MLflow 3.14 can use **OpenAI Codex**
or **Claude Code** as the provider. Repository root is `/workspace` (read/write).
Review all changes in git before accepting them.

## Build and start

```bash
docker compose build mlflow
docker compose up -d mlflow
docker compose exec mlflow codex --version
docker compose exec mlflow claude --version
```

### Docker localhost gate (required for the UI)

MLflow's Assistant API only accepts loopback clients. With Compose
`127.0.0.1:5000:5000`, the browser is local but the container sees the Docker
bridge IP, which surfaces in the setup wizard as
**"You do not have permission to access this resource."**
That is **not** a Claude login failure.

This repo sets `MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT=1` and patches the gate
in `Dockerfile.mlflow` so private Docker clients are allowed when the port is
host-bound. Rebuild/recreate `mlflow` after pulling that change.

---

## Provider A — OpenAI Codex (volume auth)

### One-time ChatGPT login

```bash
docker compose exec mlflow codex login
```

Complete the browser/device flow. Credentials persist in the
`mlflow_codex_auth` Docker volume.

---

## Provider B — Claude Code (host auth mount)

Use the Claude login **already on the host** (`~/.claude`). Compose bind-mounts:

`${HOME}/.claude` → `/home/appuser/.claude`

The container runs as `appuser` (uid **999**). Host credential files are usually
mode `600` owned by your uid, so grant ACL once:

```bash
bash scripts/mlflow_claude_host_auth_acl.sh
docker compose up -d mlflow
```

Verify the mount and auth from outside:

```bash
docker compose exec mlflow sh -lc \
  'test -r /home/appuser/.claude/.credentials.json && claude -p hi --max-turns 1 --output-format json'
```

If `claude` is missing, rebuild the image (`Dockerfile.mlflow` installs
`@anthropic-ai/claude-code`). If credentials are unreadable, re-run the ACL script.

---

## Configure Assistant

```bash
docker compose exec mlflow mlflow assistant --configure
```

Select **Claude Code** (or **OpenAI Codex**), set the repository to `/workspace`,
and enable file editing/read docs as needed. Config persists in
`mlflow_assistant_cfg`.

## Verify

```bash
python scripts/mlflow_assistant_preflight.py
```

Open `http://localhost:5000`, select an experiment, and open **Assistant**.
First ask a read-only question. Then test a write only under `/workspace/tmp/`.
Confirm `TokyoEye@champion` and `TokyoEye@experimental` aliases are unchanged.

### Keep the Assistant on the live trunk

Claude Code loads [`CLAUDE.md`](../../CLAUDE.md) at the repo root. That brief is the
control plane for Assistant sessions: **gates + code + MLflow**, not dated
`docs/superpowers/specs/*` (except the freeze addendum for numeric pins). Front
door for intended stack: [`TOKYOEYE_ARCHITECTURE_SSOT.md`](../TOKYOEYE_ARCHITECTURE_SSOT.md).

Paste this as the first message when starting a **new** Assistant thread
(or after an Assistant UI reset — the UI chat is not durable; gates + this paste are):

```text
REHYDRATE — do not re-run closed cards. Prefer gates + code + MLflow over dated specs.

Read /workspace/CLAUDE.md and follow it.
Also read (in order):
  docs/history/session-handoff-2026-09-20-equ-gfit-marginal.md
  data/gates/tokyo_eye_equ_g_fit_per_structure_recheck.json
  data/gates/tokyo_eye_equ_grad_attribution_check.json
  data/gates/tokyo_eye_equ_edge_type_reachability_check.json
  data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_close.json
  data/gates/tokyo_eye_equ_wrap1_dehydron_loso_B_disposition.json

Repo master @ 7c7529b (EQU campaign landed). CUDA required for any model run.

=== CLOSED — do not reopen ===
- wrap1 lite G_fit E200: INCONCLUSIVE_UNDERFIT
- wrap1 lite G_fit E400: locked DECAY_UNSETTLED; scientific close CLOSED_MARGINAL
  (1MBN 0.809 / 1LYZ 0.799 / 1BG1 0.824). Bar NOT moved for 0.799.
- No third lite epoch prereg. No decay-ceiling curve modeling.
- MoE/ε CLOSED; SDRP wrap1 PARKED; defect A CLOSED; defect B OPEN (leakage).
- No promote / alias / silent retune / wrap=19 / C1 / Stage-2.
- No pool typed G_fit at 0.80 yet — premise wrong (pool drops edge_type; lite type-silent).

=== DIAGNOSTICS ALREADY STAMPED (trust; do not redo unless asked) ===
1) edge_type reachability — lite init ~type-invariant; pool del edge_index, edge_type.
2) grad attribution — dehydron BCE+margin never reach spine (euclidean_skip_dehydron).
3) INSTR CUDA E400 + per-structure recheck — pooled ~0.80 masks memorisation
   (heldout 0.476 / 0.566 / 0.526). Epoch-budget chase moot as capacity question.
Artifacts: data/gates/wrap1_dehydron_loso/INSTR_seed0_hold_*_E400.json

=== OPEN — continue here ===
Orphan B harness pin:
  B pin run_v8_experiment_py_sha256 = 8c7e2be2e73bc8ca5d11f2eaa5becc3e2c1b6e5984f62db4f90a9a7efd676ae1
  committed/working hash               = c1df1048bc817dac30afd1658a3b0c5c14eab3478fc8e6ea2dcd823d3b15f2c6
  MISMATCH — do not treat INSTR replay as exact B loop.
B disposition next_command_note mentioning pool typed G_fit is STALE.
DO NOT edit B disposition until (a) recover 8c7e2be2… bytes or (b) operator
approves explicit "pin unrecoverable" narrow rewrite.

=== NEXT TASK (only) ===
1) CUDA preflight: python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
2) Search host/retired clones/old images for run_v8_experiment.py sha256=8c7e2be2…; report found/not found.
3) If not found: propose (do not apply) minimal B-disposition wording — S1/S2+G_fit say nothing about hyp geometry / typed signal / pool biology; loop identity vs B unverified.
4) No pool G_fit / MoE-on / biology claim / permanent tests/v8 instrumentation without a new sealed card.

Claude session IDs (transcripts under ~/.claude/projects, not UI chat):
  044f0e40-2c37-403e-806c-582e3cb72e7a  (main G_fit diagnostics)
  6379e625-9be7-4204-b9e3-ef1aa2a980cf  (post-login B-pin thread)

Standing rule: prose without a failing test loses to a convenient default.
```

Lite G_fit E200/E400 and the wiring/memorisation diagnostics are closed.
Do **not** re-run `G_fit_E400_prereg` or the old CPU `G_fit_task.json`.
Hold pool typed G_fit until a redesigned card exists. Optional CLI resume of a
Claude transcript (outside the UI):  
`docker compose exec mlflow sh -lc 'cd /workspace && claude --resume <session-id>'`.

The container runs as `appuser` (uid 999). Host-owned repo directories may deny
create unless prepared. For the write smoke:

```bash
docker compose exec -u root mlflow \
  sh -lc 'mkdir -p /workspace/tmp && chown appuser:appuser /workspace/tmp'
```

## Recovery

- Assistant UI chat reset: paste the starter block above; optional
  `claude --resume <session-id>` inside the container (transcripts live under
  host `~/.claude/projects/`, bind-mounted). Gates remain machine truth.
- Missing CLI: rebuild `mlflow`.
- Codex logged out: rerun `docker compose exec mlflow codex login`.
- Claude auth broken: re-auth on the host, then `bash scripts/mlflow_claude_host_auth_acl.sh`.
- Wrong provider/project: rerun `docker compose exec mlflow mlflow assistant --configure`.
- Reset config only: remove `mlflow_assistant_cfg` after explicit operator approval.
- Reset Codex login only: remove `mlflow_codex_auth` after explicit operator approval.
- Assistant cannot write under `/workspace` (uid 999 vs host 1000): ask it to write under
  `data/gates/` (world-writable stamps) or have the host agent land scripts into `scripts/`.
  Do not widen whole-repo `chmod` without operator approval.
