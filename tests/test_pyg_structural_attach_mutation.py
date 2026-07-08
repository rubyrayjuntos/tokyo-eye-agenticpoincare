"""PyG in-place structural attach + expmap σ₂/σ₁ property tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from experiments.training.v6.train_loop import attach_v6_features, prepare_training_batch
from science.dtie.common.hyperbolic_lorentz_ops import expmap0_tangent_at_origin
from science.dtie.common.structural_disc_compose import (
    StructuralResidueInput,
    attach_structural_disc_for_forward,
    compose_structural_disc,
    compose_from_training_prot,
)
from science.training.disc_occupancy import disc_occupancy_from_numpy

C_LOW = 0.6864332556724548
C_HIGH = 0.7687298655509949


def _synthetic_prot(n: int = 24) -> dict:
    rng = np.random.default_rng(42)
    x = torch.tensor(
        np.column_stack(
            [
                rng.uniform(8.0, 18.0, n),
                rng.integers(0, 2, n).astype(np.float64),
                rng.uniform(0.0, 1.0, n),
                rng.uniform(0.05, 0.9, n),
            ]
        ),
        dtype=torch.float32,
    )
    ca = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    return {
        "pdb_id": "SYN",
        "chain": "A",
        "data": Data(x=x, edge_index=edge_index),
        "ca_coords": ca,
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
    }


def _synthetic_residue_inputs(n: int = 40) -> list[StructuralResidueInput]:
    rng = np.random.default_rng(7)
    out: list[StructuralResidueInput] = []
    for i in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.15, 1.8)
        out.append(
            StructuralResidueInput(
                residue_id=f"A:{i + 1}:",
                residue_index=i + 1,
                chain_label="A",
                rho=float(rng.uniform(8.0, 18.0)),
                tau_flag=float(rng.integers(0, 2)),
                ss_type=float(rng.uniform(0, 1)),
                ca_xyz=np.array([rad * np.cos(ang), rad * np.sin(ang), 0.0]),
            )
        )
    return out


def test_repeated_attach_on_same_data_without_clone_is_last_writer_wins() -> None:
    """Same Data: without a snapshot, both handles read the last attach (false-identical)."""
    prot = _synthetic_prot()
    data = attach_v6_features(prot["data"])
    d_high = attach_structural_disc_for_forward(data, prot, C_HIGH)
    z_high_snapshot = d_high.structural_z_disc.detach().clone()
    d_low = attach_structural_disc_for_forward(data, prot, C_LOW)
    assert d_high is d_low is data
    # After the second attach, both handles read the low-c tensor on `data`.
    assert torch.allclose(d_high.structural_z_disc, d_low.structural_z_disc)
    assert torch.max(torch.abs(d_high.structural_z_disc - z_high_snapshot)) > 1e-4
    fresh_low = compose_from_training_prot(prot, C_LOW).z_disc_matrix
    fresh_high = compose_from_training_prot(prot, C_HIGH).z_disc_matrix
    assert np.max(np.abs(d_low.structural_z_disc.numpy() - fresh_low)) < 1e-5
    assert np.max(np.abs(fresh_high - fresh_low)) > 1e-4


def test_attach_independent_across_checkpoints() -> None:
    """Different curvature on cloned Data must produce different structural_z_disc."""
    prot = _synthetic_prot()
    d1 = attach_structural_disc_for_forward(
        attach_v6_features(prot["data"].clone()), prot, C_HIGH
    )
    d2 = attach_structural_disc_for_forward(
        attach_v6_features(prot["data"].clone()), prot, C_LOW
    )
    diff = np.max(np.abs(d1.structural_z_disc.numpy() - d2.structural_z_disc.numpy()))
    assert diff > 1e-4


def test_prepare_training_batch_does_not_mutate_cached_prot_data() -> None:
    """Cached prot['data'] must not gain structural fields after prepare_training_batch."""
    prot = _synthetic_prot()
    before_keys = set(prot["data"].keys())
    model = type("M", (), {"curvature": torch.tensor(C_HIGH)})()
    _ = prepare_training_batch(model, prot, "cpu", structural_disc_frozen=True)  # type: ignore[arg-type]
    after_keys = set(prot["data"].keys())
    assert before_keys == after_keys
    assert not hasattr(prot["data"], "structural_z_disc")


def test_expmap0_sigma_ratio_near_invariant_across_curvature() -> None:
    """expmap0 is not uniform scaling; verify σ₂/σ₁ stability empirically."""
    rng = np.random.default_rng(11)
    tangents: list[np.ndarray] = []
    for _ in range(8):
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        radii = rng.uniform(0.2, 2.5, size=angles.shape)
        for ang, rad in zip(angles, radii, strict=True):
            tangents.append(np.array([rad * np.cos(ang), rad * np.sin(ang)]))
    v = np.stack(tangents, axis=0)

    def mapped(c: float) -> np.ndarray:
        return np.stack([expmap0_tangent_at_origin(vi, c) for vi in v], axis=0)

    r1 = disc_occupancy_from_numpy(mapped(C_LOW))["disc_sigma2_sigma1"]
    r2 = disc_occupancy_from_numpy(mapped(C_HIGH))["disc_sigma2_sigma1"]
    assert abs(r1 - r2) < 0.05


def test_compose_structural_disc_sigma_ratio_near_invariant_across_curvature() -> None:
    """Full ρ/τ compose at two c values — σ₂/σ₁ should be close but coords differ."""
    residues = _synthetic_residue_inputs()
    z_low = compose_structural_disc(residues, C_LOW, structure_id="SYN").z_disc_matrix
    z_high = compose_structural_disc(residues, C_HIGH, structure_id="SYN").z_disc_matrix
    assert np.max(np.abs(z_low - z_high)) > 1e-4
    r_low = disc_occupancy_from_numpy(z_low)["disc_sigma2_sigma1"]
    r_high = disc_occupancy_from_numpy(z_high)["disc_sigma2_sigma1"]
    assert abs(r_low - r_high) < 0.08


@pytest.mark.skipif(
    not __import__("os").environ.get("TRAINING_LOAD_FROM_PDB"),
    reason="Set TRAINING_LOAD_FROM_PDB=1 with cached 1MBN for corpus check",
)
def test_1mbn_compose_sigma_ratio_empirical() -> None:
    """Corpus spot-check: 1MBN σ₂/σ₁ similar across cold/route c; coords differ."""
    from pathlib import Path

    from experiments.training.v6._data import load_protein_graph

    prot = load_protein_graph("1MBN", "A", Path("/tmp/dtie_pdb_cache"))
    assert prot is not None
    z_c = compose_from_training_prot(prot, C_HIGH).z_disc_matrix
    z_r = compose_from_training_prot(prot, C_LOW).z_disc_matrix
    assert np.max(np.abs(z_c - z_r)) > 0.01
    s_c = disc_occupancy_from_numpy(z_c)["disc_sigma2_sigma1"]
    s_r = disc_occupancy_from_numpy(z_r)["disc_sigma2_sigma1"]
    assert abs(s_c - s_r) < 0.02
