"""Tests for Discovery Story act-scoped readiness."""

from __future__ import annotations

from data.act_readiness import (
    ACT_ORDER,
    derive_act_readiness,
    derive_current_act,
)


def _full_artifacts(**overrides: bool) -> dict[str, bool]:
    base = {
        "dims": True,
        "scope": True,
        "gnn_hyp": True,
        "alignment": True,
        "graph": True,
        "dtie_core": True,
        "source_leaks": True,
        "discovery_extended": True,
        "dtie_phases": True,
        "binding_scan": True,
        "motifs": True,
        "md_validation": True,
        "strain_vulnerability": True,
        "witness_embedding": True,
        "pharmacophores": True,
        "drug_candidates": True,
        "resistance_pathway": True,
        "buffering_atlas": True,
        "allosteric_sites": True,
        "allele_selectivity": True,
    }
    base.update(overrides)
    return base


class TestActReadiness:
    def test_signal_complete_when_graph_present(self):
        artifacts = _full_artifacts(graph=True, witness_embedding=False, strain_vulnerability=False)
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": False}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert acts["signal"].status == "degraded"
        assert acts["signal"].required_artifacts["graph"] is True
        assert acts["signal"].required_artifacts["gnn_hyp"] is True

    def test_signal_complete_with_optionals(self):
        artifacts = _full_artifacts()
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": True}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert acts["signal"].status == "complete"

    def test_persistent_leak_pending_without_gnn(self):
        artifacts = _full_artifacts(gnn_hyp=False, graph=False, source_leaks=False)
        artifacts["dims"] = True
        artifacts["scope"] = True
        foundation = {"dims": True, "scope": True, "gnn_hyp": False, "alignment": False}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert acts["signal"].status == "pending"
        assert acts["persistent_leak"].status == "pending"

    def test_cryptic_pocket_running_during_job(self):
        artifacts = _full_artifacts(binding_scan=False, md_validation=False)
        artifacts["graph"] = True
        artifacts["source_leaks"] = True
        artifacts["dtie_core"] = True
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": False}
        acts = derive_act_readiness(
            artifacts,
            foundation=foundation,
            pipeline_job={"status": "running", "job_id": "j1"},
        )
        assert acts["signal"].status == "complete"
        assert acts["persistent_leak"].status == "complete"
        assert acts["cryptic_pocket"].status == "running"

    def test_fragment_pending_before_pharmacophore_outputs(self):
        artifacts = {
            "binding_scan": True,
            "pharmacophores": False,
            "drug_candidates": False,
        }
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": False}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert acts["fragment"].status == "pending"

    def test_current_act_is_first_incomplete(self):
        artifacts = _full_artifacts(binding_scan=False, md_validation=False)
        artifacts["source_leaks"] = True
        artifacts["dtie_core"] = True
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": False}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert derive_current_act(acts) == 3

    def test_all_acts_complete(self):
        artifacts = _full_artifacts()
        foundation = {"dims": True, "scope": True, "gnn_hyp": True, "alignment": True}
        acts = derive_act_readiness(artifacts, foundation=foundation, pipeline_job=None)
        assert all(acts[act_id].status == "complete" for act_id in ACT_ORDER)
        assert derive_current_act(acts) == 5
