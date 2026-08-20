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

### TokyoEye (production inference)

Production restore is `models:/TokyoEye@champion`; staging restore is
`models:/TokyoEye@experimental`. Architecture code lives under
`science/tokyo_eye/` (the current implementation package still includes an
internal `v8` directory). Local `.pt` files are cache/evidence only, not
contract restore pointers. Spec:
[`docs/superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md`](../../docs/superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md).
Hyp-MP package lines are **deprecated archaeology**.

### Legacy V6 GNN (compare-only)

Architecture lives in `science/dtie/v6/gnn/`; contract id `gospc_v6` (status `legacy`).
See [`science/dtie/v6/README.md`](../dtie/v6/README.md).

## Pipeline runtime audit

Geometric enforcement, curvature passthrough, preconditions, and pathway lifecycle are recorded in `audit_pipeline_events` for historical query.

**Canonical documentation:** [`docs/audit/PIPELINE_AUDIT.md`](../../docs/audit/PIPELINE_AUDIT.md)  
**Developer compliance:** [`docs/DEVELOPER_ONBOARDING.md`](../../docs/DEVELOPER_ONBOARDING.md)
