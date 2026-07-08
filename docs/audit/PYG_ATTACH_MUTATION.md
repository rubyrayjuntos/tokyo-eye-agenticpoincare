# PyG structural attach — in-place mutation audit

`attach_structural_disc_to_pyg` and `attach_structural_disc_for_forward` **mutate**
the passed `Data` object in place (set `structural_z_disc`, hyperbolic edges, etc.).
Reusing the same `prot["data"]` or the same `Data` instance across sequential
checkpoint loads **without `.clone()`** produces false-identical tensors: the second
attach overwrites the first, so cold vs route comparisons silently read one artifact.

Same failure class as stale `_proj` scope bugs: a tensor survives where a fresh one
was assumed.

## Required pattern

```python
data = attach_v6_features(prot["data"].clone().to(device))
data = attach_structural_disc_for_forward(data, prot, curvature_c)
```

Or use `prepare_training_batch(..., structural_disc_frozen=True)` which clones internally.

## Call-site inventory (v6 structural / multi-checkpoint paths)

| Location | Status | Notes |
|----------|--------|-------|
| `experiments/training/v6/train_loop.py` `prepare_training_batch` | **Fixed** | `.clone()` before attach |
| `experiments/diagnostics/embedding_occupancy_audit.py` `_forward_audit` | **Fixed** | `.clone()` before attach |
| `scripts/expert_specialization_audit.py` | **Fixed** | frozen path uses `prepare_training_batch`; else branch clones |
| `experiments/training/v6/export_corpus_viewers.py` | **Fixed** | frozen path OK; else branch clones |
| `tests/test_structural_disc_compose.py` | Safe | single attach per test |
| `science/compute/jobs/gnn_inference.py` | Safe | one attach per ingest forward |
| `experiments/training/v6/assess_checkpoint.py` | Low risk | no structural attach; single model per call |
| `experiments/training/v6/encoder_pathology_audit.py` | Low risk | no structural attach unless added |
| Diagnostics without structural attach | Safe | `attach_v6_features` only adds derived x columns |

Re-grep after new audit scripts:

```bash
rg 'attach_structural_disc|attach_v6_features\(prot\["data"\]' --glob '*.py'
```

## Regression tests

- `tests/test_pyg_structural_attach_mutation.py`
  - `test_attach_independent_across_checkpoints` — different c → different `structural_z_disc` when cloned
  - `test_repeated_attach_on_same_data_without_clone_is_last_writer_wins` — documents hazard

## Verified finding (1MBN, c=0.769 vs 0.686)

| Path | Max \|z_cold − z_route\| |
|------|-------------------------|
| `compose_from_training_prot` (fresh) | 0.060 |
| `prepare_training_batch` (clone fix) | 0.060 |
| Second `attach` on same `Data` without clone | **0.000** (false identical) |

Prior claim “cold vs route identical disc_2d — expected SSOT frozen” was **invalid**;
it was PyG mutation artifact, not proof that curvature is pinned.
