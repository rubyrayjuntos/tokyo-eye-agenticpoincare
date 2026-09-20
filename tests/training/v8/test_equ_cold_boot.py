"""CPU unit tests for Tokyo Eye EQU cold boot guards + hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training.v8.equ_cold_boot import (
    PINNED_FRONTEND_SHA256,
    ColdBootGuardError,
    assert_frontend_bank,
    assert_not_forbidden_ckpt,
    boot_hygiene_pass,
    build_boot_stamp,
    load_boot_split,
)


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "manifests" / "equ_cold_boot_v1.json"


def test_manifest_split():
    split = load_boot_split(MANIFEST)
    assert len(split["train"]) == 8
    assert len(split["probe"]) == 6
    assert {p["pdb_id"] for p in split["train"]} == {
        "1MBN", "1LYZ", "1F88", "1HHP", "1TEN", "1UBQ", "1TIM", "4OBE"
    }
    assert "1BG1" not in {p["pdb_id"] for p in split["train"] + split["probe"]}


def test_sha_guard_ok(tmp_path: Path):
    bank = tmp_path / "bank.pt"
    bank.write_bytes(b"not-real")
    # wrong content -> mismatch
    with pytest.raises(ColdBootGuardError):
        assert_frontend_bank(bank)
    # pin match via rewriting expected on a fake with matching hash of empty? use real pin file if present
    real = ROOT / "checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt"
    if real.is_file():
        assert assert_frontend_bank(real) == PINNED_FRONTEND_SHA256


def test_forbidden_ckpt_paths():
    with pytest.raises(ColdBootGuardError):
        assert_not_forbidden_ckpt("/tmp/TokyoEye_champion_v5/from_vault/v8_affinity_best.pt")
    with pytest.raises(ColdBootGuardError):
        assert_not_forbidden_ckpt("checkpoints/tokyoeye/runs/eqf_c1_topology_curriculum_20260825/tokyoeye_best.pt")
    assert_not_forbidden_ckpt("checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt")


def _probe_row(**kwargs):
    base = {
        "loaded": True,
        "z_hyp_finite": True,
        "curvature_finite": True,
        "boundary_saturation": 0.2,
        "radius_spread": 0.12,
        "moe_load_min": 0.08,
    }
    base.update(kwargs)
    return base


def test_hygiene_pass_and_fail():
    rows = [_probe_row() for _ in range(6)]
    home = _probe_row(pdb_id="4OBE")
    ok = boot_hygiene_pass(rows, home)
    assert ok["hygiene_pass"] is True

    rim = [_probe_row(boundary_saturation=0.99, radius_spread=0.001) for _ in range(6)]
    bad = boot_hygiene_pass(rim, home)
    assert bad["hygiene_pass"] is False

    dead = [_probe_row(moe_load_min=0.0) for _ in range(6)]
    bad2 = boot_hygiene_pass(dead, home)
    assert bad2["hygiene_pass"] is False
    assert bad2["n_probe_moe_spread"] == 0


def test_stamp_schema():
    rows = [_probe_row(pdb_id=f"P{i}") for i in range(6)]
    stamp = build_boot_stamp(
        rows,
        _probe_row(pdb_id="4OBE"),
        frontend_sha256=PINNED_FRONTEND_SHA256,
        wrap_max=1,
        boundary_radius=0.80,
        tau_probe=0.90,
    )
    assert stamp["gate"] == "tokyo_eye_equ_cold_boot"
    assert stamp["display_lineage"] == "Tokyo Eye EQU"
    assert stamp["hygiene_pass"] is True
    assert "init_from_v5_or_affinity_or_c1" in stamp["forbidden"]
