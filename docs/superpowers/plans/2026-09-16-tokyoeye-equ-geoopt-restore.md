# Plan — tokyo_eye_equ_geoopt_restore

## Goal
Restore the alive architecture: EquiformerV3 → (cut before energy pool) → geoopt.PoincareBall lift → R0–R5 hyp attention → MoE → evidential. Kill SE(3)-lite + hand-rolled lift as the sealed spine path.

## Tasks
1. **geoopt projector** — `RadialAngularProjector` uses `geoopt.manifolds.PoincareBall.expmap0` + `proj`; return coords safe for spine; optional ManifoldTensor at API boundary.
2. **Equiformer-to-pool frontend** — vendor `experimental/models/equiformer_v3`; wrap forward; expose node `(s,v)` after `_forward_blocks`, **before** `energy_block`. Load MPtrj `mptrj_gradient.pt` into that module. Freeze frontend for geometry/biology spine train.
3. **Protein atom batch adapter** — map PDB residues/atoms → Equiformer `data` (pos, atomic_numbers, batch, natoms, cell/pbc off for biomolecules).
4. **Disable lite** — under this gate, `live_backbone` SE(3)-lite is a Fail preflight if selected.
5. **Keep** on-manifold gyro/Einstein attention, soft MoE, EvidentialHead, R0–R5, PoincareDiagnostics.
6. **Preflight** then sealed cold train (fresh spine; not theme Fail θ; not lite θ).

## Pins
- frontend sha MPtrj `59c6c235…`
- wrap_max=1
- vendor path `vendor/equiformer_v3`
