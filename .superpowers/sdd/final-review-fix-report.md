# Final Whole-Branch Review Fix Report

Date: 2026-07-09
Branch: `feature/dehydron-barcode-input-channel`
Base HEAD reviewed: `acb9a29`

## Changes

- Fixed barcode corpus cache invalidation by adding a sidecar-state digest to the barcode cache suffix. The digest covers matching `*_dehydron_barcode_v1.pt` file count, names, sizes, and `mtime_ns`; missing, absent, or empty sidecar directories are tagged as `dbh_sidecars_none`.
- Hardened dehydron barcode sidecar loading by preferring `torch.load(..., weights_only=True)` and falling back to legacy unsafe loading only for `TypeError` / `pickle.UnpicklingError`, with a one-time warning to re-precompute legacy sidecars.
- Updated `design.md` and the implementation plan to document P1's intentional structure-global barcode summary broadcast semantics, with only `n_dehydrons_touching` remaining residue-local.
- Fixed the `ablation.md` here-doc example and removed trailing whitespace from the related-design line.

## Tests

- Added regression coverage for cache suffix changes when a sidecar is added or replaced.
- Added regression coverage proving `attach_dehydron_barcode_features` attempts the `weights_only=True` load path.

Full focused test command run before commit:

```bash
pytest tests/test_corpus_barcode_cache.py tests/test_dehydron_barcode_features.py tests/test_launch_training_barcode.py -q
```
