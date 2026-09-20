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

Paste this as the first message when starting a **new** Assistant thread:

```text
Read /workspace/CLAUDE.md and follow it. Do not survey dated specs.
Prefer gates under data/gates/tokyo_eye_equ_*.json + assembly_gate.py over prose.

Hard stops (do not re-open):
- No SE(3)-lite as governed frontend; no silent retune; no wrap=19; no C1/Stage-2.
- Defect A CLOSED ≠ defect B fixed. MoE/ε CLOSED; SDRP wrap1 PARKED.
- No new model version / experiment rename.
- Do not treat B S1/S2 as no-signal yet.

Next task ONLY — sealed packet:
data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_prereg.json

Preflight (STOP if either fails):
  python3 -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))"
  # E=200 G_fit result must remain INCONCLUSIVE_UNDERFIT; do not overwrite it.

1) Optional smoke: python3 scripts/wrap1_dehydron_loso_g_fit.py --smoke --device cuda --no-mlflow
2) Exact run:   python3 scripts/wrap1_dehydron_loso_g_fit.py --device cuda --seed 0 --epochs 400
3) Confirm stamps:
   data/gates/tokyo_eye_equ_wrap1_dehydron_loso_G_fit_E400_result.json
   data/gates/wrap1_dehydron_loso/T_seed0_hold_*_G_fit_E400.json
   disposition updated.
4) Read order (mandatory, before interpreting): r-trajectory → AUPRC≥0.80 → plateau.
   Named outcomes ONLY: CLEARS_0_80 | PLATEAU_BELOW | DECAY_UNSETTLED.
   No pool escalate under DECAY_UNSETTLED. Do not promote.
If CUDA unavailable: STOP — do not fall back to cpu for this card.
```

E=200 G_fit is closed (`INCONCLUSIVE_UNDERFIT`). Do not re-run the old
`G_fit_task.json` cpu packet. After E=400 resolves, read
`tokyo_eye_equ_next_experiment_sealed.json` only if the E400 outcome allows
escalation (pool under `PLATEAU_BELOW` only).

The container runs as `appuser` (uid 999). Host-owned repo directories may deny
create unless prepared. For the write smoke:

```bash
docker compose exec -u root mlflow \
  sh -lc 'mkdir -p /workspace/tmp && chown appuser:appuser /workspace/tmp'
```

## Recovery

- Missing CLI: rebuild `mlflow`.
- Codex logged out: rerun `docker compose exec mlflow codex login`.
- Claude auth broken: re-auth on the host, then `bash scripts/mlflow_claude_host_auth_acl.sh`.
- Wrong provider/project: rerun `docker compose exec mlflow mlflow assistant --configure`.
- Reset config only: remove `mlflow_assistant_cfg` after explicit operator approval.
- Reset Codex login only: remove `mlflow_codex_auth` after explicit operator approval.
- Assistant cannot write under `/workspace` (uid 999 vs host 1000): ask it to write under
  `data/gates/` (world-writable stamps) or have the host agent land scripts into `scripts/`.
  Do not widen whole-repo `chmod` without operator approval.
