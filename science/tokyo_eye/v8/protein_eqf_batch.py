"""Map protein CA / atom coords → fairchem-style Equiformer ``data`` batch.

Under ``tokyo_eye_equ_geoopt_restore``, SE(3)-lite is forbidden. EquiformerV3_OC
needs ``pos``, ``atomic_numbers``, ``batch``, ``natoms``, ``cell``, ``pbc``.
Residue-level spine graphs use CA nodes; default adapter treats each CA as
carbon (Z=6) so node count matches R0–R5.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

# Carbon for CA-level residue nodes (spine alignment). Full-atom Z map is later.
_CA_ATOMIC_NUMBER = 6


def ca_coords_to_eqf_data(
    pos: torch.Tensor,
    *,
    atomic_number: int = _CA_ATOMIC_NUMBER,
    cell_scale: float = 1000.0,
) -> SimpleNamespace:
    """Build a single-graph Equiformer batch from CA positions ``[N, 3]``."""
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"expected pos [N,3], got {tuple(pos.shape)}")
    device = pos.device
    dtype = pos.dtype
    n = int(pos.shape[0])
    atomic_numbers = torch.full(
        (n,), int(atomic_number), dtype=torch.long, device=device
    )
    batch = torch.zeros(n, dtype=torch.long, device=device)
    natoms = torch.tensor([n], dtype=torch.long, device=device)
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * float(cell_scale)
    pbc = torch.zeros(1, 3, dtype=torch.bool, device=device)
    return SimpleNamespace(
        pos=pos.contiguous(),
        atomic_numbers=atomic_numbers,
        batch=batch,
        natoms=natoms,
        cell=cell,
        pbc=pbc,
    )


def attach_eqf_data(batch: dict[str, Any], *, atomic_number: int = _CA_ATOMIC_NUMBER) -> dict[str, Any]:
    """Attach ``eqf_data`` built from batch[\"x\"] (CA coords)."""
    batch = dict(batch)
    batch["eqf_data"] = ca_coords_to_eqf_data(batch["x"], atomic_number=atomic_number)
    return batch


__all__ = ["attach_eqf_data", "ca_coords_to_eqf_data"]
