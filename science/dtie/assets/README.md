# DTIE assets

Small, versioned inputs bundled with the science stack (not governed DB facts).

## `benchmark_pdbs/`

21 structurally diverse PDBs for GPU smoke tests and v2-teacher distillation benchmarks.
Referenced by `manifests/v6_corpus_benchmark.json`.

Used as `--pdb-dir` for `make train-v6-benchmark` and `make assess-v6-benchmark`.
