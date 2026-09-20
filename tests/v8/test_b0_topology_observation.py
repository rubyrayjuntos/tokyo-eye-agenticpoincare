"""B0 topology-observation panel + rhyme + stamp (no GPU, no training)."""

from __future__ import annotations

import math
from pathlib import Path

from experiments.training.v8.b0_topology_observation import (
    HELD_OUT_PDBS,
    THEMES,
    build_stamp,
    load_panel,
    rho_by_ss,
    ss_class_for_residue,
    theme_rhyme_narratives,
)
from science.tokyo_eye.sse_hierarchy import SSERange

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "v8_b0_topology_observation_v1.json"


def test_load_panel_has_six_themes_three_each() -> None:
    panel = load_panel(MANIFEST)
    assert len(panel) == 18
    assert {p["theme"] for p in panel} == set(THEMES)
    pairs = [(p["pdb_id"], p["chain"]) for p in panel]
    assert len(pairs) == len(set(pairs))
    for theme in THEMES:
        members = [p for p in panel if p["theme"] == theme]
        assert len(members) == 3, theme
        assert all(p["enabled"] for p in members)
    held = {p["pdb_id"] for p in panel}
    assert held.isdisjoint(HELD_OUT_PDBS)
    pg = next(p for p in panel if p["pdb_id"] == "1PGB")
    assert pg["flags"]["grasp_cousin"] is True
    nmr = next(p for p in panel if p["pdb_id"] == "1A5R")
    assert nmr["flags"]["nmr_model1"] is True
    ghl = next(p for p in panel if p["pdb_id"] == "1GHL")
    assert ghl["flags"]["first_ca_polymer"] is True
    nal = next(p for p in panel if p["pdb_id"] == "1NAL")
    assert nal["flags"]["first_ca_polymer"] is True
    hng = next(p for p in panel if p["pdb_id"] == "1HNG")
    assert hng["flags"]["fallback_pdb_id"] == "1CD2"


def test_ss_class_helix_sheet_coil() -> None:
    ranges = [
        SSERange(sse_type="H", chain="A", start_resseq=10, end_resseq=20),
        SSERange(sse_type="E", chain="A", start_resseq=30, end_resseq=35),
        SSERange(sse_type="E", chain="A", start_resseq=12, end_resseq=14),
    ]
    assert ss_class_for_residue(12, "A", ranges) == "sheet"
    assert ss_class_for_residue(18, "A", ranges) == "helix"
    assert ss_class_for_residue(32, "A", ranges) == "sheet"
    assert ss_class_for_residue(1, "A", ranges) == "coil"
    assert ss_class_for_residue(12, "B", ranges) == "coil"


def test_rho_by_ss_empty_class_is_nan() -> None:
    stats = rho_by_ss(
        [0.1, 0.9, 0.2],
        ["helix", "helix", "coil"],
        [2.0, 3.0, 1.0],
    )
    assert stats["n_helix"] == 2
    assert stats["n_sheet"] == 0
    assert math.isnan(stats["mean_rho_sheet"])
    assert abs(stats["mean_rho_helix"] - 0.5) < 1e-6
    assert abs(stats["mean_tau_helix"] - 2.5) < 1e-6


def _row(
    *,
    pdb_id: str,
    theme: str,
    mean_rho_sheet: float,
    mean_rho_helix: float,
    mean_rho_coil: float,
    loaded: bool = True,
) -> dict:
    return {
        "pdb_id": pdb_id,
        "theme": theme,
        "loaded": loaded,
        "mean_rho_sheet": mean_rho_sheet,
        "mean_rho_helix": mean_rho_helix,
        "mean_rho_coil": mean_rho_coil,
        "z_hyp_finite": True,
        "curvature_finite": True,
    }


def test_theme_rhyme_rhyme_singleton_collapse() -> None:
    rhyme_rows = [
        _row(
            pdb_id="1TEN",
            theme="ig_like",
            mean_rho_sheet=0.40,
            mean_rho_helix=0.10,
            mean_rho_coil=0.12,
        ),
        _row(
            pdb_id="1FNA",
            theme="ig_like",
            mean_rho_sheet=0.42,
            mean_rho_helix=0.11,
            mean_rho_coil=0.13,
        ),
        _row(
            pdb_id="1HNG",
            theme="ig_like",
            mean_rho_sheet=0.41,
            mean_rho_helix=0.09,
            mean_rho_coil=0.14,
        ),
        _row(
            pdb_id="1MBN",
            theme="globin",
            mean_rho_sheet=0.08,
            mean_rho_helix=0.08,
            mean_rho_coil=0.07,
        ),
        _row(
            pdb_id="2HHB",
            theme="globin",
            mean_rho_sheet=0.09,
            mean_rho_helix=0.08,
            mean_rho_coil=0.08,
        ),
        _row(
            pdb_id="1ASH",
            theme="globin",
            mean_rho_sheet=0.07,
            mean_rho_helix=0.07,
            mean_rho_coil=0.08,
        ),
    ]
    narratives = theme_rhyme_narratives(rhyme_rows)
    assert narratives["ig_like"]["label"] == "rhyme"
    assert narratives["globin"]["label"] == "collapse"

    singleton_rows = [
        _row(
            pdb_id="1TEN",
            theme="ig_like",
            mean_rho_sheet=0.55,
            mean_rho_helix=0.10,
            mean_rho_coil=0.12,
        ),
        _row(
            pdb_id="1FNA",
            theme="ig_like",
            mean_rho_sheet=0.12,
            mean_rho_helix=0.11,
            mean_rho_coil=0.10,
        ),
        _row(
            pdb_id="1HNG",
            theme="ig_like",
            mean_rho_sheet=0.11,
            mean_rho_helix=0.09,
            mean_rho_coil=0.10,
        ),
    ]
    singleton = theme_rhyme_narratives(singleton_rows)
    assert singleton["ig_like"]["label"] == "singleton"


def test_build_stamp_hygiene_and_no_biology_pass() -> None:
    rows = [
        {
            "pdb_id": "1TEN",
            "chain": "A",
            "theme": "ig_like",
            "theta": "champion",
            "loaded": True,
            "n_res": 90,
            "z_hyp_finite": True,
            "curvature_finite": True,
            "curvature": 1.2,
            "mean_rho_sheet": 0.4,
            "mean_rho_helix": 0.1,
            "mean_rho_coil": 0.12,
            "dehydron_auprc": "auprc_na",
        },
        {
            "pdb_id": "1FNA",
            "chain": "A",
            "theme": "ig_like",
            "theta": "champion",
            "loaded": False,
            "skip_reason": "no_ca_complete",
            "z_hyp_finite": False,
            "curvature_finite": False,
        },
        {
            "pdb_id": "1HNG",
            "chain": "A",
            "theme": "ig_like",
            "theta": "champion",
            "loaded": True,
            "n_res": 176,
            "z_hyp_finite": False,
            "curvature_finite": True,
            "curvature": 1.2,
            "mean_rho_sheet": 0.41,
            "mean_rho_helix": 0.1,
            "mean_rho_coil": 0.12,
            "dehydron_auprc": 0.3,
        },
    ]
    stamp = build_stamp(
        rows,
        champion_sha256="507d54bd" + "0" * 56,
        champion_version=5,
        mode_c_s9_status="mode_c_s9_absent",
    )
    assert stamp["biology_pass"] is False
    assert stamp["biology_gate"] == "not_applicable"
    assert stamp["hygiene_finite"] is False
    assert stamp["n_loaded"] == 2
    assert stamp["n_skip"] == 1
    assert "theme_narratives" in stamp
    assert stamp["theta"]["alias"] == "champion"
    assert stamp["mode_c_s9"]["status"] == "mode_c_s9_absent"
    assert "pass" not in stamp["theme_narratives"]["ig_like"]
