"""P_FEATURE_01 — per-residue MASTER feature parity (two-sided).

Scope (unit / module parity — runs in CI without DB):
  legacy PDB MASTER load (_data.load_protein_graph_from_pdb_legacy)
  == GraphBuilder.build_node_features_from_pdb(MASTER)
  == residue_features.build_from_pdb_chain(MASTER)

Equality semantics (FeatureMode.MASTER):
  rho:      exact integer dehydron count (±0)
  tau_flag: exact binary (rho < 13.0)
  ss_type:  exact DSSP-class categorical (0.0 / 0.5 / 1.0 via biotite P-SEA)
  sasa:     same-algorithm tolerance (FreeSASA Å², heavy atoms)

Not covered here (required before master-feature training — see science/training/p_feature_01_gate.py):
  make gate-p-feature-01  →  fresh SSOT recompute == fact_ingestion_features == load_graph_from_db
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.training.v6 import _data
from science.dtie.common import residue_features as rf
from science.dtie.common.graph_builder import GraphBuilder

FIXTURE_PDB = Path("science/dtie/assets/benchmark_pdbs/1CRN.pdb")
FIXTURE_CHAIN = "A"


def _training_features_from_load(pdb_id: str, chain: str, pdb_dir: Path) -> list[rf.ResidueNodeFeatures]:
    """Training-side features via legacy PDB MASTER load (module parity tests)."""
    loaded = _data.load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    assert loaded is not None, f"load_protein_graph returned None for {pdb_id}:{chain}"
    x = loaded["data"].x.detach().cpu().numpy()
    feats: list[rf.ResidueNodeFeatures] = []
    for i, rid in enumerate(loaded["residue_ids"]):
        idx = int(rid.split(":")[1])
        feats.append(
            rf.ResidueNodeFeatures(
                chain_label=chain,
                residue_index=idx,
                rho=float(x[i, 0]),
                tau_flag=float(x[i, 1]),
                ss_type=float(x[i, 2]),
                sasa=float(x[i, 3]),
            )
        )
    return feats


def _ca_coords_for_features(pdb_path: Path, chain: str, features: list[rf.ResidueNodeFeatures]) -> np.ndarray:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("parity", str(pdb_path))
    res_by_idx = {
        int(r.get_id()[1]): r
        for r in structure.get_residues()
        if r.get_id()[0] == " " and r.parent.id == chain
    }
    coords = []
    for feat in features:
        res = res_by_idx[feat.residue_index]
        coords.append(res["CA"].get_coord())
    return np.array(coords, dtype=np.float64)


@pytest.fixture
def pdb_dir(tmp_path: Path) -> Path:
    """Use repo fixture PDB copied into tmp dir (mirrors training download layout)."""
    dest = tmp_path / "pdb_cache"
    dest.mkdir()
    dest.joinpath("1CRN.pdb").write_bytes(FIXTURE_PDB.read_bytes())
    return dest


def test_p_feature_01_positive_training_vs_graphbuilder_pdb(pdb_dir: Path) -> None:
    """POSITIVE: legacy PDB MASTER load == GraphBuilder SSOT on same PDB/chain."""
    training = _training_features_from_load("1CRN", FIXTURE_CHAIN, pdb_dir)
    inference = GraphBuilder.build_node_features_from_pdb(
        str(FIXTURE_PDB), FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER
    )
    rf.assert_feature_parity(training, inference)


def test_p_feature_01_positive_training_vs_residue_features_ssot(pdb_dir: Path) -> None:
    """POSITIVE: legacy PDB MASTER load == direct residue_features SSOT."""
    training = _training_features_from_load("1CRN", FIXTURE_CHAIN, pdb_dir)
    ssot = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN, mode=rf.FeatureMode.MASTER)
    rf.assert_feature_parity(training, ssot)


def test_p_feature_01_negative_burial_rho_skew_detected(pdb_dir: Path) -> None:
    """NEGATIVE: legacy burial-count rho skew must fail and name rho."""
    reference = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN)
    ca_coords = _ca_coords_for_features(FIXTURE_PDB, FIXTURE_CHAIN, reference)
    skewed = rf.inject_burial_rho_skew(reference, ca_coords)

    mismatches = rf.compare_feature_parity(reference, skewed)
    assert mismatches, "expected burial rho skew to break parity"
    rho_mismatches = [m for m in mismatches if m.feature == "rho"]
    assert rho_mismatches, f"expected rho divergence, got: {mismatches[:5]}"
    assert all(m.rule == "exact-int" for m in rho_mismatches)


def test_p_feature_01_negative_tau_flag_skew_detected() -> None:
    """NEGATIVE: flipped tau_flag must fail with exact-bool rule."""
    reference = rf.build_from_pdb_chain(FIXTURE_PDB, FIXTURE_CHAIN)
    flipped = [
        rf.ResidueNodeFeatures(
            chain_label=f.chain_label,
            residue_index=f.residue_index,
            rho=f.rho,
            tau_flag=0.0 if f.tau_flag == 1.0 else 1.0,
            ss_type=f.ss_type,
            sasa=f.sasa,
        )
        for f in reference
    ]
    mismatches = rf.compare_feature_parity(reference, flipped)
    assert any(m.feature == "tau_flag" and m.rule == "exact-bool" for m in mismatches)


def test_p_feature_01_equality_criteria_documented() -> None:
    """Sanity: compare_feature_parity attaches per-feature rules."""
    a = rf.ResidueNodeFeatures("A", 1, rho=5.0, tau_flag=1.0, ss_type=0.0, sasa=0.5)
    b = rf.ResidueNodeFeatures("A", 1, rho=8.0, tau_flag=0.0, ss_type=0.5, sasa=0.51)
    mismatches = rf.compare_feature_parity([a], [b])
    rules = {m.feature: m.rule for m in mismatches}
    assert rules["rho"] == "exact-int"
    assert rules["tau_flag"] == "exact-bool"
    assert rules["ss_type"] == "exact-categorical"
    assert rules["sasa"] == "same-algorithm-tolerance"
