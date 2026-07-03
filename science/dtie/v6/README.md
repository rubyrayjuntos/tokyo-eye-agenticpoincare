# V6 GNN — hyperbolic prototype MoE

Production inference uses **one Python source tree** under `science/dtie/v6/gnn/` paired with weights at `checkpoints/v6/tokyo_eyes_v6.pt`. Training (`experiments/training/v6/launch_training.py`) imports the same `GOSPConeMapperV6` — there is no separate training copy.

| Piece | Location |
|-------|----------|
| Architecture | `model.py`, `hyperbolic_moe.py`, `runner.py` |
| Weights | `checkpoints/v6/tokyo_eyes_v6.pt` (gitignored) |
| Contract | `science/contracts/onboard_contract.yaml` → `gnn_models.gospc_v6` |

Docker bind-mounts `./science` → `/app/science` and `./checkpoints` → `/app/checkpoints`. **Run all dev commands below inside the science container** so paths and dependencies match ingest/compute.

### Interactive 3D viewer (restored from v2)

After each successful `gnn_inference` (when `GNN_INTERACTIVE_HTML=true`, default), the science job writes:

- `{GNN_VIEWER_OUTPUT_DIR}/{structure_id}/{structure_id}_gosp_native.pdb` — B-factor = epistemic, occupancy = cone depth
- `{structure_id}_interactive.html` — NGL cartoon + uncertainty surface (open in browser)

Agent serves: `GET /api/structures/{structure_id}/gnn-viewer` (also linked from readiness as `gnn_viewer_url` when present).

## Dev checklist (science container)

Start DB + science if not already up:

```bash
make up-science
```

### 1. After editing `model.py`, `hyperbolic_moe.py`, or `runner.py`

Restart the long-running service (bind-mount picks up code; reload avoids stale imports):

```bash
docker compose restart science
```

### 2. Promote trained weights to production (optional)

Container paths only (`/app/checkpoints/...`):

```bash
make promote-production-v6 \
  SOURCE=checkpoints/v6/runs/full_hyp_moe_test/v6_best.pt
```

Or explicitly:

```bash
docker compose run --rm --user $(id -u):$(id -g) science \
  cp /app/checkpoints/v6/runs/full_hyp_moe_test/v6_best.pt \
     /app/checkpoints/v6/tokyo_eyes_v6.pt
```

### 3. Verify architecture + checkpoint pairing

```bash
make verify-v6-gnn
```

Equivalent one-liner:

```bash
docker compose run --rm --user $(id -u):$(id -g) science python -c "
from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate
from science.contracts.model_registry import checkpoint_status, get_production_checkpoint_path
from science.dtie.v6.gnn.model import verify_v6_checkpoint
print('hyperbolic_moe OK')
print(checkpoint_status(get_production_checkpoint_path()))
print(verify_v6_checkpoint())
"
```

Expect: `hyperbolic_moe OK`, checkpoint `exists: True`, `load_ok: True`, `deep_hyperbolic_gate: True`, `gate_disc_scale: 2.5`.

### 4. Integration tests (science container)

```bash
make test-v6-gnn-integration
```

### 5. Health endpoint (optional)

With stack up:

```bash
curl -s http://localhost:8001/health | jq '.gnn_production'
```

Look for `architecture_compat.ok: true`.

## What breaks if only one piece is present

| Missing | Symptom |
|---------|---------|
| Old `model.py` + new `.pt` | Missing keys, random-init `mobius3`, wrong routing |
| New `model.py` + old `.pt` | Loads but no deep gate / hypmix behavior |
| New code but no `hyperbolic_moe.py` | `ImportError` on inference |

## Pipeline wiring

Ingest `gnn_inference` → `science/compute/jobs/gnn_inference.py` → `V6GNNRunner` → Normalizer. Production checkpoint and runner class come from `science/contracts/model_registry.py` (contract YAML).
