"""Verification tests for FeatureMode.MASTER (DSSP-class SSE + FreeSASA Å²).

Run before DB repopulation — confirms the shared module produces all four
master features on a single structure without relying on any ingest path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from science.dtie.common import residue_features as rf

pytest.importorskip("freesasa")

FIXTURE_PDB = Path("science/dtie/assets/benchmark_pdbs/1CRN.pdb")
FIXTURE_CHAIN = "A"


def _master_features() -> list[rf.ResidueNodeFeatures]:
    feats = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    assert len(feats) >= 10
    return feats


def test_master_features_all_four_populated() -> None:
    """MASTER mode produces rho, tau, sse_code, and FreeSASA sasa on 1CRN:A."""
    feats = _master_features()

    for feat in feats:
        assert feat.rho >= 0.0
        assert int(round(feat.rho)) == feat.rho, "rho must be integer-valued"
        assert feat.tau_flag in (0.0, 1.0)
        expected_tau = 1.0 if feat.rho < rf.TAU else 0.0
        assert feat.tau_flag == expected_tau
        assert feat.sse_code in ("H", "E", "C")
        assert feat.ss_type == rf.encode_ss_type_from_sse_code(feat.sse_code)
        assert feat.sasa >= 0.0
        assert feat.sasa <= 300.0, f"sasa {feat.sasa} outside sane per-residue Å² range"

    sasa_vals = [f.sasa for f in feats]
    assert max(sasa_vals) > 50.0, "expected some solvent-exposed residues with substantial SASA"
    assert sum(1 for v in sasa_vals if v > 0.0) > len(feats) * 0.5

    sse_counts = rf.sse_code_counts([f.sse_code for f in feats])
    assert sse_counts["H"] >= 5, "1CRN should have helix content (not all-coil)"
    assert sse_counts["H"] + sse_counts["E"] >= 8, "1CRN SS must not be constant coil"


def test_master_features_idempotent_dual_call() -> None:
    """Two calls to the same shared functions must agree exactly."""
    a = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    b = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    rf.assert_feature_parity(a, b)


def test_master_sasa_exposed_gt_core_median() -> None:
    """FreeSASA: highest-sasa residues should exceed buried core (sanity, not biology)."""
    feats = _master_features()
    sasa = np.array([f.sasa for f in feats])
    top_quartile = float(np.quantile(sasa, 0.75))
    bottom_quartile = float(np.quantile(sasa, 0.25))
    assert top_quartile > bottom_quartile
    assert top_quartile >= 20.0, "exposed quartile should be materially > 0 Å²"


def test_master_dssp_differs_from_geometric_ss() -> None:
    """MASTER SS must not silently fall through to geometric encoding."""
    current = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.TRAINING_CURRENT)
    master = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    cur_map = {rf.residue_key(f.chain_label, f.residue_index): f.ss_type for f in current}
    mas_map = {rf.residue_key(f.chain_label, f.residue_index): f.ss_type for f in master}
    diffs = [k for k in cur_map if not np.isclose(cur_map[k], mas_map[k], atol=0.0)]
    assert diffs, "MASTER ss_type identical to geometric — DSSP path may be broken"


def test_master_sasa_differs_from_proxy() -> None:
    """MASTER FreeSASA (Å²) must not match unitless proxy scale."""
    current = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.TRAINING_CURRENT)
    master = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    cur_sasa = np.array([f.sasa for f in current])
    mas_sasa = np.array([f.sasa for f in master])
    assert mas_sasa.max() > 1.0, "FreeSASA should exceed unitless proxy upper bound"
    assert not np.allclose(cur_sasa, mas_sasa, rtol=0.05, atol=0.05)


TIM_BARREL_PDB = Path("pdb_cache/1TIM.pdb")


@pytest.mark.skipif(not TIM_BARREL_PDB.exists(), reason="1TIM PDB not in pdb_cache")
def test_master_sse_tim_barrel_has_helix_and_sheet() -> None:
    """1TIM α/β barrel — falsifiable SS correctness gate (not all-coil)."""
    feats = rf.build_from_pdb_chain(TIM_BARREL_PDB, "A", mode=rf.FeatureMode.MASTER)
    assert len(feats) >= 200
    codes = [f.sse_code for f in feats]
    counts = rf.sse_code_counts(codes)
    assert counts["H"] >= 50, f"1TIM expected substantial helix, got {counts}"
    assert counts["E"] >= 20, f"1TIM expected β-strands, got {counts}"
    structured_frac = (counts["H"] + counts["E"]) / len(codes)
    assert structured_frac >= 0.40, f"1TIM structured fraction too low: {structured_frac:.2f}"


def test_master_freesasa_uses_multiple_heavy_atoms_per_residue() -> None:
    """FreeSASA must aggregate real heavy atoms, not a degenerate Cα-only fallback."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sasa", str(FIXTURE_PDB))
    records: list[rf.ResidueRecord] = []
    for res in structure.get_residues():
        if res.get_id()[0] != " " or res.parent.id != FIXTURE_CHAIN:
            continue
        atoms = tuple(
            rf.AtomRecord(
                atom.name,
                (atom.element or "").upper(),
                np.asarray(atom.coord, dtype=np.float64),
                res.get_resname().strip().upper(),
            )
            for atom in res.get_atoms()
        )
        records.append(
            rf.ResidueRecord(
                chain_label=FIXTURE_CHAIN,
                residue_index=int(res.get_id()[1]),
                residue_name=res.get_resname().strip().upper(),
                atoms=atoms,
            )
        )
    heavy_per_res = [
        sum(
            1
            for a in r.atoms
            if (a.element or "").upper() != "H" and not a.atom_name.upper().startswith("H")
        )
        for r in records
    ]
    assert max(heavy_per_res) >= 4, "expected backbone+sidechain heavy atoms per residue"
    assert sum(1 for n in heavy_per_res if n >= 4) > len(records) * 0.8
    sasa = rf.compute_sasa_freesasa(records)
    assert float(sasa.max()) > 50.0
