"""Tests for pipeline-derived residue supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.training.v6.pipeline_labels import (
    PipelineFacts,
    _cryptic_targets_for_chain,
    _leak_targets_for_chain,
    apply_pipeline_labels,
    load_pipeline_facts,
    training_key_from_governed_residue_id,
)


def test_training_key_mapping() -> None:
    assert training_key_from_governed_residue_id("4obe:A:31") == "A:31:"
    assert training_key_from_governed_residue_id("bad") is None


def test_leak_targets_top_quartile_on_chain() -> None:
    residue_ids = ["A:25:", "A:27:", "A:30:", "A:31:"]
    scores = {
        "4obe:A:25": 10.0,
        "4obe:A:27": 12.0,
        "4obe:A:30": 11.0,
        "4obe:A:31": 13.0,
    }
    target, mask = _leak_targets_for_chain(residue_ids, "A", scores)
    assert mask.all()
    positives = int(target.sum())
    assert positives >= 1
    assert target[3] == 1.0


def test_cryptic_targets_union_on_chain() -> None:
    residue_ids = ["A:10:", "A:11:", "A:12:"]
    target, mask, matched = _cryptic_targets_for_chain(
        residue_ids,
        "A",
        ["4obe:A:11", "4obe:B:99"],
    )
    assert matched == {"A:11:"}
    assert mask.all()
    assert target.tolist() == [0.0, 1.0, 0.0]


def test_apply_pipeline_labels_sets_leak_tensors() -> None:
    prot = {
        "pdb_id": "4OBE",
        "chain": "A",
        "residue_ids": ["A:25:", "A:27:", "A:30:", "A:31:"],
        "label_meta": {},
    }
    facts = PipelineFacts(
        structure_id="4obe",
        leak_scores={
            "4obe:A:25": 10.0,
            "4obe:A:27": 12.0,
            "4obe:A:30": 11.0,
            "4obe:A:31": 13.0,
        },
        source="test",
    )
    apply_pipeline_labels(prot, facts)
    assert "target_leak" in prot
    assert "leak_label_mask" in prot
    assert prot["leak_label_mask"].all()
    assert prot["label_meta"]["leak_source"] == "pipeline_source_leak"


def test_load_pipeline_facts_from_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "4obe.json"
    sidecar.write_text(
        json.dumps(
            {
                "structure_id": "4obe",
                "cryptic_residue_ids": ["4obe:A:31"],
                "source_leaks": [{"residue_id": "4obe:A:31", "leak_score": 12.0}],
            }
        )
    )
    facts = load_pipeline_facts("4OBE", sidecar_dir=tmp_path)
    assert facts is not None
    assert facts.has_leaks
    assert "4obe:A:31" in facts.leak_scores
