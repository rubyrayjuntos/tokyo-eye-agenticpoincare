"""Tests for GNN lifecycle control plane helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from science.training.lifecycle import (
    add_structure_to_corpus,
    enqueue_train_job,
    gate_stamp_status,
    list_lineages,
    resolve_checkpoint_for_preview,
)


def test_list_lineages_includes_v6_and_v65() -> None:
    lineages = list_lineages()
    ids = {x["lineage_id"] for x in lineages}
    assert "v6" in ids
    assert "v6.5" in ids
    assert "v6.6" in ids


def test_enqueue_train_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr("science.training.lifecycle._JOBS_PATH", jobs_path)
    job = enqueue_train_job({"lineage_id": "v6.5", "run_id": "t1"})
    assert job["status"] == "queued"
    assert job["payload"]["run_id"] == "t1"
    saved = json.loads(jobs_path.read_text())
    assert saved["active"]["job_id"] == job["job_id"]


def test_add_structure_to_corpus(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"proteins": []}))
    result = add_structure_to_corpus(
        structure_id="4obe",
        manifest_path=manifest,
        pdb_id="4OBE",
    )
    assert result["action"] == "added"
    data = json.loads(manifest.read_text())
    assert data["proteins"][0]["pdb_id"] == "4OBE"
    assert data["proteins"][0]["enabled"] is True

    again = add_structure_to_corpus(
        structure_id="4obe",
        manifest_path=manifest,
        pdb_id="4OBE",
    )
    assert again["action"] == "updated"
    assert len(json.loads(manifest.read_text())["proteins"]) == 1


def test_resolve_checkpoint_for_preview_missing_path() -> None:
    out = resolve_checkpoint_for_preview(checkpoint_path="/nonexistent/model.pt")
    assert out["exists"] is False
    assert out["source"] == "path"


def test_gate_stamp_status_shape() -> None:
    status = gate_stamp_status()
    assert "p_feature_01_passed" in status
