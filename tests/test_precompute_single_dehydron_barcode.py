from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.training.v6 import precompute_dehydron_barcodes as precompute
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    SCALAR_NAMES,
    barcode_sidecar_filename,
)


def _locked_stats(path: Path, *, version: str = BARCODE_FEATURE_VERSION) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "scalar_names": list(SCALAR_NAMES),
                "n_valid_rows": 24,
                "mean": [1.0, 2.0, 3.0],
                "std": [2.0, 4.0, 5.0],
                "eps": 1e-6,
                "scope": "master_training_graph_rows",
            }
        )
    )
    return path


def test_single_pdb_precompute_reuses_locked_stats(tmp_path, monkeypatch) -> None:
    pdb_path = tmp_path / "1AGP.pdb"
    pdb_path.write_text("placeholder")
    stats_path = _locked_stats(tmp_path / "locked_stats.json")
    out_dir = tmp_path / "sidecars"

    payload = {
        "scalars": np.asarray([[3.0, 6.0, 8.0], [9.0, 9.0, 9.0]], dtype=np.float32),
        "missing": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "residue_indices": np.asarray([12, 13], dtype=np.int32),
        "edge_pairs": np.empty((0, 2), dtype=np.int32),
        "edge_scalars": np.empty((0, 5), dtype=np.float32),
        "metadata": {"n_midpoints": 8, "n_bars": 2},
    }
    monkeypatch.setattr(precompute, "_resolve_pdb_path", lambda *_: pdb_path)
    monkeypatch.setattr(
        precompute,
        "featurize_chain_dehydron_barcode",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        precompute,
        "build_from_pdb_chain",
        lambda *_args, **_kwargs: [
            SimpleNamespace(residue_index=12),
            SimpleNamespace(residue_index=13),
        ],
    )

    sidecar = precompute.precompute_single_pdb(
        pdb_id="1AGP",
        chain="A",
        pdb_dir=tmp_path,
        out_dir=out_dir,
        zscore_stats_path=stats_path,
        use_binned=False,
    )

    assert sidecar == out_dir / barcode_sidecar_filename("1AGP", "A")
    saved = torch.load(sidecar, map_location="cpu", weights_only=True)
    assert torch.allclose(saved["scalars"][0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(saved["scalars"][1], torch.zeros(3))
    assert saved["metadata"]["corpus_zscore"]["path"] == str(stats_path)
    assert saved["metadata"]["corpus_zscore"]["n_valid_rows"] == 24
    assert not (out_dir / "corpus_zscore_stats.json").exists()


def test_single_pdb_precompute_rejects_wrong_stats_version(tmp_path) -> None:
    stats_path = _locked_stats(tmp_path / "wrong_stats.json", version="obsolete")

    with pytest.raises(ValueError, match="version"):
        precompute.precompute_single_pdb(
            pdb_id="1AGP",
            chain="A",
            pdb_dir=tmp_path,
            out_dir=tmp_path / "sidecars",
            zscore_stats_path=stats_path,
            use_binned=False,
        )
