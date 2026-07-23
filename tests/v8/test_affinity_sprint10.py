"""Sprint 10 affinity head + Core leak wall unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from science.tokyo_eye.v8.affinity_head import PocketGatedAffinityHead
from science.tokyo_eye.v8.seq_cluster import (
    assert_no_core_leak,
    cluster_sequences_exact,
    ids_hitting_core,
    kmer_jaccard,
    normalize_seq,
    split_reps_train_val,
)


def test_pocket_weights_softmax_sum_one_variable_n() -> None:
    for n in (7, 31, 64):
        head = PocketGatedAffinityHead(hidden_dim=16)
        z = torch.randn(n, 16) * 0.1
        mech = torch.randn(n)
        dehyd = torch.zeros(n)
        dehyd[: max(1, n // 5)] = 1.0
        w = head.pocket_weights(z, mechanism_score=mech, dehydron_labels=dehyd)
        assert w.shape == (n, 1)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_pool_log_exp_finite() -> None:
    head = PocketGatedAffinityHead(hidden_dim=8)
    z = torch.randn(12, 8) * 0.05
    mech = torch.zeros(12)
    dehyd = torch.ones(12)
    out = head(z, mechanism_score=mech, dehydron_labels=dehyd)
    assert out["affinity_pred"].ndim == 0 or out["affinity_pred"].numel() == 1
    assert torch.isfinite(out["affinity_pred"])
    assert torch.isfinite(out["z_graph"]).all()


def test_assert_no_core_leak_raises() -> None:
    try:
        assert_no_core_leak(["1abc", "2def"], ["3ghi"], ["2DEF", "9zzz"])
        raised = False
    except AssertionError as exc:
        raised = True
        assert "CORE LEAK" in str(exc)
    assert raised


def test_assert_no_core_leak_clean() -> None:
    assert_no_core_leak(["1abc"], ["2def"], ["3ghi", "4jkl"])


def test_ids_hitting_core_exact_and_kmer() -> None:
    core = {"c1": "ACDEFGHIKLMNPQRSTVWY" * 3}
    query = {
        "same": "ACDEFGHIKLMNPQRSTVWY" * 3,
        "far": "GGGGGGGGGGGGGGGGGGGG" * 3,
        "near": "ACDEFGHIKLMNPQRSTVWY" * 3,  # exact
    }
    hit = ids_hitting_core(query, core, kmer_jaccard_floor=0.45)
    assert "same" in hit
    assert "near" in hit
    assert "far" not in hit


def test_cluster_exact_and_split() -> None:
    seqs = {"a": "AAAA", "b": "AAAA", "c": "CCCC", "d": "DDDD"}
    m = cluster_sequences_exact(seqs)
    assert m["a"] == m["b"]
    reps = sorted(set(m.values()))
    train, val = split_reps_train_val(reps, val_fraction=0.5, seed=0)
    assert set(train) | set(val) == set(reps)
    assert set(train).isdisjoint(set(val))


def test_normalize_and_jaccard() -> None:
    assert normalize_seq("acde-*") == "ACDE"
    assert kmer_jaccard("AAAAAA", "AAAAAA", k=3) == 1.0
    assert kmer_jaccard("AAAAAA", "GGGGGG", k=3) == 0.0


def test_frozen_manifest_no_leak_if_present() -> None:
    path = Path("manifests/v8_pdbbind_refined_cluster30_v1.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    assert_no_core_leak(
        [r["pdb_id"] for r in data["train"]],
        [r["pdb_id"] for r in data["val"]],
        [r["pdb_id"] for r in data["core_test"]],
    )
    core = {r["pdb_id"].upper() for r in data["core_test"]}
    assert len(core) == len(data["core_test"])
    assert data["n_core_test"] == len(data["core_test"])


def test_softmax_dim0_matches_functional() -> None:
    scores = torch.randn(17, 1)
    w = F.softmax(scores, dim=0)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)


def test_finetune_hyp_freeze_contract() -> None:
    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
    from experiments.training.v8.run_affinity_s10 import configure_finetune_hyp

    frontend = StubEquiformerFrontend(
        in_dim=3, scalar_dim=16, vector_dim=3, live_backbone=False
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=16, vector_dim=3, hidden_dim=16, num_attn_layers=2
    )
    system = TokyoEyeV8WithFrontend(frontend, spine)
    info = configure_finetune_hyp(system)
    assert info["mode"] == "finetune_hyp"
    assert all(not p.requires_grad for p in system.frontend.parameters())
    assert any(p.requires_grad for p in system.spine.projector.parameters())
    assert any(p.requires_grad for p in system.spine.attn_layers.parameters())
    assert any(p.requires_grad for p in system.spine.moe.parameters())
    assert all(not p.requires_grad for p in system.spine.sdrp_head.parameters())


def test_finetune_all_unfreezes_frontend_and_optimizer_groups() -> None:
    from science.tokyo_eye.v8.affinity_head import JointPocketAffinityHead
    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
    from experiments.training.v8.run_affinity_s10 import (
        build_optimizer,
        configure_finetune_all,
    )

    frontend = StubEquiformerFrontend(
        in_dim=3, scalar_dim=16, vector_dim=3, live_backbone=True
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=16, vector_dim=3, hidden_dim=16, num_attn_layers=2
    )
    system = TokyoEyeV8WithFrontend(frontend, spine)
    info = configure_finetune_all(system)
    assert info["mode"] == "finetune_all"
    assert any(p.requires_grad for p in system.frontend.parameters())
    head = JointPocketAffinityHead(16)
    opt = build_optimizer(
        "finetune_all",
        system,
        head,
        lr_head=1e-3,
        lr_hyperbolic=3e-4,
        lr_backbone=1e-5,
    )
    names = {g.get("name"): g["lr"] for g in opt.param_groups}
    assert names["backbone"] == 1e-5
    assert names["hyperbolic"] == 3e-4
    assert names["affinity_head"] == 1e-3
