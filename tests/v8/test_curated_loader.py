"""Sprint 6 tests: curated PDB loader (4OBE + manifest filter)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from science.tokyo_eye.v8.loader import (
    TokyoEyeCuratedDataset,
    dehydron_labels_from_edges,
    load_structure_batch,
    parse_enabled_manifest,
)
from science.tokyo_eye.v8.r0_r5_graph import R2_DEHYDRON


ROOT = Path(__file__).resolve().parents[2]
PDB_DIR = ROOT / "pdb_cache"
MANIFEST = ROOT / "manifests" / "v8_stage_a_small_v1.json"


@pytest.mark.skipif(not (PDB_DIR / "4OBE.pdb").is_file(), reason="4OBE.pdb missing")
def test_4obe_batch_has_nontrivial_dehydron_labels() -> None:
    import numpy as np

    from science.tokyo_eye.v8.r0_r5_graph import (
        get_dehydron_wrap_max,
        set_dehydron_wrap_max,
    )

    set_dehydron_wrap_max(19)
    batch = load_structure_batch(
        "4OBE", "A", pdb_dir=PDB_DIR, device="cpu", use_graph_cache=False
    )
    assert batch["num_nodes"] > 50
    assert batch["edge_index"].ndim == 2
    meta = batch["graph_meta"]
    assert "hbond_wrap_counts" in meta
    assert meta.get("dssp_energy_cutoff") == -0.5
    # Cone compresses wraps; τ=19 marks nearly all DSSP H-bonds as R2.
    wraps = meta["hbond_wrap_counts"]
    assert wraps, "expected DSSP-admitted H-bonds"
    assert max(wraps) < 19, "cone wrap should compress below legacy τ=19"

    # Epoch-0 style retune: median then descend until frac < 0.60
    frac = float(batch["dehydron_labels"].mean())
    if frac >= 0.60:
        tau = int(np.median(np.asarray(wraps, dtype=float)))
        while tau >= 0:
            set_dehydron_wrap_max(tau)
            batch = load_structure_batch(
                "4OBE", "A", pdb_dir=PDB_DIR, device="cpu", use_graph_cache=False
            )
            frac = float(batch["dehydron_labels"].mean())
            if frac < 0.60 or tau == 0:
                break
            tau -= 1
    assert 0.05 < frac < 0.60, f"expected calibrated dehydron frac, got {frac}"
    assert int(batch["sdrp_target"].max()) < 5
    assert batch["pdb_id"] == "4OBE"
    set_dehydron_wrap_max(19)
    assert get_dehydron_wrap_max() == 19


@pytest.mark.skipif(not (PDB_DIR / "4OBE.pdb").is_file(), reason="4OBE.pdb missing")
def test_graph_cache_hit_second_load(tmp_path: Path) -> None:
    from science.tokyo_eye.v8.loader import graph_cache_path

    cache_dir = tmp_path / "v8_graph_cache"
    b1 = load_structure_batch(
        "4OBE", "A", pdb_dir=PDB_DIR, use_graph_cache=True, graph_cache_dir=cache_dir
    )
    assert b1.get("from_graph_cache") is False
    path = graph_cache_path("4OBE", "A", cache_dir=cache_dir)
    assert path.is_file()
    b2 = load_structure_batch(
        "4OBE", "A", pdb_dir=PDB_DIR, use_graph_cache=True, graph_cache_dir=cache_dir
    )
    assert b2.get("from_graph_cache") is True
    assert b2["num_nodes"] == b1["num_nodes"]
    assert float(b2["dehydron_frac"]) == pytest.approx(float(b1["dehydron_frac"]))


def test_parse_enabled_manifest_stage_a() -> None:
    entries = parse_enabled_manifest(MANIFEST)
    assert len(entries) == 12
    assert all("pdb_id" in e and "chain" in e for e in entries)
    assert any(e["pdb_id"] == "4OBE" for e in entries)


def test_manifest_disabled_filtered(tmp_path: Path) -> None:
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "proteins": [
                    {"pdb_id": "AAAA", "chain": "A", "enabled": False},
                    {"pdb_id": "BBBB", "chain": "B", "enabled": True},
                ]
            }
        )
    )
    entries = parse_enabled_manifest(path)
    assert entries == [{"pdb_id": "BBBB", "chain": "B"}]


def test_dehydron_incidence_helper() -> None:
    ei = __import__("numpy").array([[0, 1, 2], [1, 0, 3]], dtype="int64")
    et = __import__("numpy").array([R2_DEHYDRON, R2_DEHYDRON, 0], dtype="int64")
    lab = dehydron_labels_from_edges(4, ei, et)
    assert lab.tolist() == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.skipif(not (PDB_DIR / "4OBE.pdb").is_file(), reason="4OBE.pdb missing")
def test_dataset_mode_a_default() -> None:
    ds = TokyoEyeCuratedDataset(pdb_dir=PDB_DIR)
    assert len(ds) == 1
    assert ds.mode == "single"
    b = ds[0]
    assert b["num_nodes"] > 0
