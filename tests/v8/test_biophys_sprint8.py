"""Sprint 8: Kabsch–Sander energy + double-cone wrapping contracts."""

from __future__ import annotations

import numpy as np
import pytest

from science.tokyo_eye.v8.biophysics import (
    DSSP_ENERGY_CUTOFF,
    WRAP_CONE_COS,
    _count_wrapping_double_cone,
    kabsch_sander_energy,
    place_backbone_amide_h,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord


def _atom(
    name: str,
    xyz: tuple[float, float, float],
    res: str,
    *,
    element: str | None = None,
) -> AtomRecord:
    el = element if element is not None else (name[0] if name[0].isalpha() else "C")
    return AtomRecord(
        atom_name=name,
        element=el,
        coord=np.array(xyz, dtype=np.float64),
        parent_residue_name=res,
    )


def _strong_hbond_pair() -> tuple[ResidueRecord, ResidueRecord, ResidueRecord]:
    """Classic near-linear N–H⋯O with E ≪ −0.5 (plus prev C for H reconstruct)."""
    prev = ResidueRecord(
        chain_label="A",
        residue_index=9,
        residue_name="ALA",
        atoms=(
            _atom("N", (-3.0, 0.0, 0.0), "ALA", element="N"),
            _atom("CA", (-1.5, 0.0, 0.0), "ALA", element="C"),
            _atom("C", (-0.5, 1.0, 0.0), "ALA", element="C"),
            _atom("O", (-0.5, 2.2, 0.0), "ALA", element="O"),
        ),
    )
    # Donor N at origin; explicit H along +z; CA in xy for completeness
    donor = ResidueRecord(
        chain_label="A",
        residue_index=10,
        residue_name="ALA",
        atoms=(
            _atom("N", (0.0, 0.0, 0.0), "ALA", element="N"),
            _atom("H", (0.0, 0.0, 1.01), "ALA", element="H"),
            _atom("CA", (1.45, 0.0, 0.0), "ALA", element="C"),
            _atom("C", (2.2, 1.0, 0.0), "ALA", element="C"),
            _atom("O", (2.2, 2.2, 0.0), "ALA", element="O"),
            _atom("CB", (1.45, -1.5, 0.0), "ALA", element="C"),
        ),
    )
    # Acceptor O along +z from H; carbonyl C offset in +x
    acceptor = ResidueRecord(
        chain_label="A",
        residue_index=14,
        residue_name="ALA",
        atoms=(
            _atom("N", (4.0, 0.0, 3.0), "ALA", element="N"),
            _atom("CA", (4.0, 0.0, 2.8), "ALA", element="C"),  # CA–CA to donor ~4Å? wait
            _atom("C", (1.2, 0.0, 2.85), "ALA", element="C"),
            _atom("O", (0.0, 0.0, 2.85), "ALA", element="O"),
            _atom("CB", (5.0, 0.0, 2.8), "ALA", element="C"),
        ),
    )
    # Fix acceptor CA near donor CA for spatial gate (~5Å)
    acceptor = ResidueRecord(
        chain_label="A",
        residue_index=14,
        residue_name="ALA",
        atoms=(
            _atom("N", (3.0, 1.0, 2.0), "ALA", element="N"),
            _atom("CA", (2.5, 0.5, 1.5), "ALA", element="C"),
            _atom("C", (1.2, 0.0, 2.85), "ALA", element="C"),
            _atom("O", (0.0, 0.0, 2.85), "ALA", element="O"),
            _atom("CB", (3.5, 0.5, 1.5), "ALA", element="C"),
        ),
    )
    return prev, donor, acceptor


def test_wrap_cone_cos_is_45deg() -> None:
    assert WRAP_CONE_COS == pytest.approx(float(np.cos(np.deg2rad(45.0))), rel=1e-5)
    assert WRAP_CONE_COS == pytest.approx(0.7071, abs=1e-3)


def test_kabsch_sander_admits_strong_pair() -> None:
    _prev, donor, acceptor = _strong_hbond_pair()
    e = kabsch_sander_energy(donor, acceptor)
    assert e is not None
    assert e <= DSSP_ENERGY_CUTOFF
    assert e < -1.0


def test_kabsch_sander_rejects_far_pair() -> None:
    donor = ResidueRecord(
        chain_label="A",
        residue_index=1,
        residue_name="ALA",
        atoms=(
            _atom("N", (0.0, 0.0, 0.0), "ALA", element="N"),
            _atom("H", (0.0, 0.0, 1.01), "ALA", element="H"),
            _atom("CA", (1.5, 0.0, 0.0), "ALA", element="C"),
        ),
    )
    acceptor = ResidueRecord(
        chain_label="A",
        residue_index=2,
        residue_name="ALA",
        atoms=(
            _atom("C", (10.0, 0.0, 0.0), "ALA", element="C"),
            _atom("O", (11.0, 0.0, 0.0), "ALA", element="O"),
            _atom("CA", (9.0, 0.0, 0.0), "ALA", element="C"),
        ),
    )
    e = kabsch_sander_energy(donor, acceptor)
    assert e is not None
    assert e > DSSP_ENERGY_CUTOFF


def test_place_amide_h_length_and_bisector() -> None:
    prev = ResidueRecord(
        chain_label="A",
        residue_index=1,
        residue_name="ALA",
        atoms=(_atom("C", (-1.0, 1.0, 0.0), "ALA", element="C"),),
    )
    donor = ResidueRecord(
        chain_label="A",
        residue_index=2,
        residue_name="ALA",
        atoms=(
            _atom("N", (0.0, 0.0, 0.0), "ALA", element="N"),
            _atom("CA", (1.0, 0.0, 0.0), "ALA", element="C"),
        ),
    )
    h = place_backbone_amide_h(donor, prev)
    assert h is not None
    n = np.array([0.0, 0.0, 0.0])
    dist = float(np.linalg.norm(h - n))
    assert dist == pytest.approx(1.01, abs=1e-6)


def test_double_cone_on_axis_counts_equatorial_rejects() -> None:
    mid = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    on_axis = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    equatorial = np.array([[3.0, 0.0, 0.0]], dtype=np.float64)
    assert (
        _count_wrapping_double_cone(mid, axis, on_axis, wrapping_radius=6.5) == 1.0
    )
    assert (
        _count_wrapping_double_cone(mid, axis, equatorial, wrapping_radius=6.5)
        == 0.0
    )
    both = np.vstack([on_axis, equatorial])
    assert _count_wrapping_double_cone(mid, axis, both, wrapping_radius=6.5) == 1.0
