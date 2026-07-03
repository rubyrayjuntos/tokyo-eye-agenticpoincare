"""Tests for hydrate bundle availability metadata."""

from __future__ import annotations

from science.contracts.hydrate_availability import (
    build_artifact_availability,
    build_hydrate_meta,
    enrich_hydrate_bundle,
)
from science.contracts.onboard_contract import get_tier_artifact_keys


def test_artifact_availability_maps_tier1_present():
    payload = {
        "structure_snapshot": {
            "structure": {"structure_id": "rcsb_4obe"},
            "scope": {"primary_chain_ids": ["chain_a"]},
            "residues": [{"residue_id": "4obe:A:1", "x": 0.1, "y": 0.2}],
            "curvature": 0.85,
        },
        "embeddings": {
            "structure_id": "rcsb_4obe",
            "curvature": 0.85,
            "residues": [{"residue_id": "4obe:A:1", "x": 0.1, "y": 0.2}],
        },
        "graph_metrics": {"metrics": [{"residue_id": "4obe:A:1"}]},
        "source_leaks": {"source_leaks": [{"residue_id": "4obe:A:1"}]},
        "binding_scan": {
            "structure_id": "rcsb_4obe",
            "status": "complete",
            "sites": [{"site_id": "site_1", "site_type": "surface_pocket", "druggability_score": 0.5}],
            "count": 1,
        },
    }
    availability = build_artifact_availability(payload)
    assert availability["dims"]["present"] is True
    assert availability["scope"]["present"] is True
    assert availability["gnn_hyp"]["present"] is True
    assert availability["graph"]["present"] is True
    assert availability["source_leaks"]["present"] is True
    assert availability["binding_scan"]["present"] is True
    assert availability["motifs"]["present"] is False
    assert availability["motifs"]["reason"] == "not_in_hydrate_bundle"


def test_hydrate_meta_not_degraded_when_tier1_binding_scan_present():
    payload = {
        "structure_snapshot": {
            "structure": {"structure_id": "11qe"},
            "scope": {"primary_chain_ids": ["A"]},
            "residues": [{"residue_id": "11qe:A:1"}],
        },
        "embeddings": {"residues": [{"residue_id": "11qe:A:1"}]},
        "graph_metrics": {"metrics": [{"residue_id": "11qe:A:1"}]},
        "source_leaks": {"source_leaks": [{"residue_id": "11qe:A:1"}]},
        "allosteric_sites": {"sites": [{"site_id": "s1", "residue_ids": []}]},
        "binding_scan": {
            "structure_id": "11qe",
            "status": "complete",
            "sites": [{"site_id": "p1", "site_type": "cryptic_wedge", "druggability_score": 0.7}],
            "count": 1,
        },
    }
    availability = build_artifact_availability(payload)
    meta = build_hydrate_meta(payload, availability)
    assert availability["binding_scan"]["present"] is True
    assert "binding_scan" not in meta["missing_keys"]


def test_hydrate_meta_flags_degraded_when_tier1_missing():
    payload = {
        "structure_snapshot": {"structure": {"structure_id": "rcsb_4obe"}, "scope": {}},
        "embeddings": None,
        "graph_metrics": None,
        "source_leaks": None,
    }
    availability = build_artifact_availability(payload)
    meta = build_hydrate_meta(payload, availability)
    assert meta["degraded"] is True
    assert meta["missing_tier1_count"] >= 1
    assert "gnn_hyp" in meta["missing_keys"]
    assert meta["contract_version"] == "1.6"


def test_enrich_hydrate_bundle_attaches_metadata():
    payload = {"structure_id": "rcsb_4obe", "structure_snapshot": None}
    enriched = enrich_hydrate_bundle(payload)
    assert "artifact_availability" in enriched
    assert "hydrate_meta" in enriched
    for tier_key in get_tier_artifact_keys(1):
        assert tier_key in enriched["artifact_availability"]
