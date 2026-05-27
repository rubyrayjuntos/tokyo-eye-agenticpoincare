# Source Inventory & Audit Log

This document tracks the state of the three source locations during consolidation.

## Locations

1. **Agent Repo** (`SRC_AGENT`)
2. **Demensional Investigator** (`SRC_DEM`) — Strong v3 DTIE
3. **Visualizer** (`SRC_VIZ`) — Current v4 GNN + frontend

## Status

- [ ] Full recursive file listing captured
- [ ] Line count per major module
- [ ] Dependency analysis (imports between files)
- [ ] Identification of training-only vs inference code
- [ ] License / IP review

See `MIGRATION_MAP.md` for the current best mapping of what should move where.

---

**This is a living audit document.** Update frequently during Phase 0 and Phase 1.