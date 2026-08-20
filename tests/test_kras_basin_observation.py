"""Unit tests for KRAS basin observation classify helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from science.dtie.common.kras_basin_observation import (
    allele_delta_site_specificity,
    basin_observation_verdict,
    parse_g12_allele,
    parse_nucleotide_class,
    r_star_for_structure,
)


def test_parse_guards_from_cache() -> None:
    cache = Path("/tmp/dtie_pdb_cache")
    if not (cache / "4LPK.pdb").is_file():
        return  # skip if cache absent in CI sandbox
    assert parse_nucleotide_class(cache / "4LPK.pdb") == "GDP"
    assert parse_nucleotide_class(cache / "5US4.pdb") == "GDP"
    assert parse_nucleotide_class(cache / "6GOD.pdb") == "GppNHp"
    assert parse_nucleotide_class(cache / "6GOF.pdb") == "GppNHp"
    assert parse_g12_allele(cache / "4LPK.pdb") == "G12"
    assert parse_g12_allele(cache / "5US4.pdb") == "G12D"
    assert parse_g12_allele(cache / "6GOD.pdb") == "G12"
    assert parse_g12_allele(cache / "6GOF.pdb") == "G12D"


def test_r_star_and_verdict() -> None:
    r = r_star_for_structure({10, 12, 30, 60, 100}, [12, 32])
    assert 12 in r and 30 in r and 60 in r
    assert 100 not in r  # not switch/N12
    assert 10 not in r

    ok = basin_observation_verdict(
        guards_ok=True,
        same_nuc=0.9,
        cross_nuc=0.7,
        allele_off_pass=True,
        allele_on_pass=True,
    )
    assert ok["pass"] is True
    fail = basin_observation_verdict(
        guards_ok=True,
        same_nuc=0.7,
        cross_nuc=0.9,
        allele_off_pass=True,
        allele_on_pass=True,
    )
    assert fail["pass"] is False


def test_allele_delta_prefers_n12() -> None:
    # Synthetic: large delta on N12; small on other R★ sites (scramble pool)
    n = 100
    depth_wt = np.zeros(n)
    depth_mut = np.zeros(n)
    map_ids = {i: i - 1 for i in range(1, n + 1)}
    n12 = [9, 10, 11, 12, 13, 14]
    for r in n12:
        depth_mut[r - 1] = 1.0
    for r in list(range(25, 41)) + list(range(57, 76)):
        depth_mut[r - 1] = 0.05
    mut_prot = {"residue_ids": [f"A:{i}:" for i in range(1, n + 1)]}
    rstar = sorted(set(n12) | set(range(25, 41)) | set(range(57, 76)))
    out = allele_delta_site_specificity(
        depth_wt,
        map_ids,
        depth_mut,
        map_ids,
        rstar=rstar,
        n12=n12,
        mut_prot=mut_prot,
    )
    assert out["pass_arm"] is True
    assert out["mean_abs_delta_n12"] > out["mean_abs_delta_scramble"]
