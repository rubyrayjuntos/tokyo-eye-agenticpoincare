# LEGACY — kept for provenance and backward compatibility only.
"""V4 DTIE — GOSPConeMapper-v4 (Hyperbolic, single pathway).

⚠️  DEPRECATED: Do not use for new analysis. Use the v5 pipeline instead.

This code exists solely to:
- Reproduce historical v4 results for validation
- Provide provenance for data produced by v4 runs
- Document the architectural evolution from v4 → v5

The v4 GNN had a training conflict (domain_sep vs cone_loss competing
for the same parameters). V5 resolves this with decoupled radial-angular heads.
"""
