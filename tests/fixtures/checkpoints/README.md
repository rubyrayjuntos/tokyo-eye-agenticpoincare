# P_CURV_01 fixture checkpoint

`lever_a_v6_best_disc.pt` is a copy of the production lever_a disc checkpoint
(`checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt`). It exists so
GitHub Actions can run `test_p_curv_01_probe_sources_match` without relying on
gitignored `checkpoints/`.

**Tradeoff (MVP):** ~3.7MB binary in git history. Acceptable for one fixture;
if `make sync-p-curv-fixture` runs more than a couple of times, move to Git LFS
or CI pull from S3 (MLflow Layer 2 artifact store).

Refresh after a deliberate production checkpoint promotion:

```bash
make sync-p-curv-fixture
```
