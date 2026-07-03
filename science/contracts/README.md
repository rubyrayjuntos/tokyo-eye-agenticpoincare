# Onboard Contract & Geometric Semantics

The **Master Onboard Contract** (`onboard_contract.yaml`) is the single source of truth for:

- Artifact catalog, readiness tiers, and Discovery Story acts
- Job metadata (normalizer destinations, geometric requirements)
- API surface shapes consumed by readiness endpoints and the frontend

## Hyperbolic geometry rule

**All geometric computations run in hyperbolic (Poincaré) space by default.** The contract declares which artifacts and jobs are explicitly `hyperbolic`, `euclidean`, or `mixed`.

### Learned curvature (not hardcoded)

Curvature is **learned by the GNN at inference time** (`gnn_inference`). The value is persisted on `embedding_space.curvature` and exposed in `JobRunResult.outputs.curvature`.

Downstream hyperbolic jobs must receive this learned value via `JobRunContext.learned_curvature` and `job_params["learned_curvature"]`. Never hardcode a numeric curvature in the contract or runners.

### Enforcement

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEOMETRIC_ENFORCEMENT_LEVEL` | `warning` | `warning` logs contract violations; `error` fails the job run |
| `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` | (empty) | Comma-separated job IDs forced to `error` even when global level is `warning` |

### Regenerating derived artifacts

```bash
make contract-sync   # job_schema.json + frontend TypeScript types
```

### V6 GNN (production inference)

Architecture lives in `science/dtie/v6/gnn/`; weights at `checkpoints/v6/tokyo_eyes_v6.pt`. **Dev commands run in the science container** — see [`science/dtie/v6/README.md`](../dtie/v6/README.md) (`make verify-v6-gnn`, `make promote-production-v6`).

## Pipeline runtime audit

Geometric enforcement, curvature passthrough, preconditions, and pathway lifecycle are recorded in `audit_pipeline_events` for historical query.

**Canonical documentation:** [`docs/audit/PIPELINE_AUDIT.md`](../../docs/audit/PIPELINE_AUDIT.md)  
**Developer compliance:** [`docs/DEVELOPER_ONBOARDING.md`](../../docs/DEVELOPER_ONBOARDING.md)
