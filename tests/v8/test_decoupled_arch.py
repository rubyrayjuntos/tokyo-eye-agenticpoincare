"""Decoupled-R2 architecture: input graph, label typing, heads, value-path attention."""

from __future__ import annotations

import numpy as np
import torch

from science.tokyo_eye.v8.attention import HyperbolicGraphAttention
from science.tokyo_eye.v8.loader import dehydron_labels_from_edges
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.r0_r5_graph import (
    R1_HBOND,
    R2_DEHYDRON,
    build_r0_r5_graph,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord


def _a(name, xyz, el):
    return AtomRecord(
        atom_name=name, element=el, coord=np.array(xyz, dtype=np.float64),
        parent_residue_name="ALA",
    )


def _hbond_records() -> list[ResidueRecord]:
    """Same geometry as test_dual_layer_r0_and_r1_coexist_for_adjacent_hbond."""
    prev = ResidueRecord("A", 9, "ALA", (
        _a("N", (-3.0, 0.0, 0.0), "N"), _a("CA", (-1.5, 0.0, 0.0), "C"),
        _a("C", (-0.5, 1.0, 0.0), "C"), _a("O", (-0.5, 2.2, 0.0), "O"),
    ))
    r10 = ResidueRecord("A", 10, "ALA", (
        _a("N", (0.0, 0.0, 0.0), "N"), _a("H", (0.0, 0.0, 1.01), "H"),
        _a("CA", (1.45, 0.0, 0.0), "C"), _a("C", (2.2, 1.0, 0.0), "C"),
        _a("O", (2.2, 2.2, 0.0), "O"), _a("CB", (1.45, -1.5, 0.0), "C"),
    ))
    r11 = ResidueRecord("A", 11, "ALA", (
        _a("N", (3.0, 1.0, 2.0), "N"), _a("CA", (2.5, 0.5, 1.5), "C"),
        _a("C", (1.2, 0.0, 2.85), "C"), _a("O", (0.0, 0.0, 2.85), "O"),
        _a("CB", (3.5, 0.5, 1.5), "C"),
    ))
    return [prev, r10, r11]


def test_decoupled_input_has_no_r2_and_no_wrap_but_labels_match_legacy() -> None:
    recs = _hbond_records()
    legacy = build_r0_r5_graph(recs)
    dec = build_r0_r5_graph(recs, decouple_r2_input=True)

    assert not np.any(dec.edge_type == R2_DEHYDRON)
    assert np.all(dec.edge_attr[:, 0] == 0.0)  # raw wrap scrubbed
    assert np.all((dec.edge_attr[:, 2].astype(np.int64) >> R2_DEHYDRON) & 1 == 0)

    n = legacy.num_nodes
    lab_legacy = dehydron_labels_from_edges(n, legacy.edge_index, legacy.edge_type)
    lab_dec = dehydron_labels_from_edges(n, dec.label_edge_index, dec.label_edge_type)
    np.testing.assert_array_equal(lab_legacy, lab_dec)
    # Every legacy H-bond (R1 or R2) is R1 in the decoupled input graph.
    hb_legacy = (legacy.edge_type == R1_HBOND) | (legacy.edge_type == R2_DEHYDRON)
    assert int((dec.edge_type == R1_HBOND).sum()) >= int(hb_legacy.sum())
    assert dec.meta["n_r2"] == legacy.meta["n_r2"]  # counts stay label-side


def test_legacy_default_unchanged() -> None:
    r = build_r0_r5_graph(_hbond_records())
    assert r.label_edge_index is None and r.label_edge_type is None


def _spine(**kw) -> TokyoEyesHyperbolicV8:
    torch.manual_seed(0)
    return TokyoEyesHyperbolicV8(
        scalar_dim=16, vector_dim=3, hidden_dim=16, num_attn_layers=2,
        num_sdrp_classes=5, gate_hidden=8, **kw,
    )


def _batch(n: int = 10):
    torch.manual_seed(1)
    s, v = torch.randn(n, 16), torch.randn(n, 3) * 0.1
    src = torch.arange(n)
    ei = torch.stack([torch.cat([src, (src + 1) % n]), torch.cat([(src + 1) % n, src])])
    et = torch.randint(0, 6, (ei.shape[1],))
    return s, v, ei, et


def test_decoupled_spine_heads_and_gradients() -> None:
    m = _spine(decoupled_arch=True)
    m.set_moe_mode("ablated")
    m.train()
    s, v, ei, et = _batch()
    out = m(s, v, ei, et, tau_ceiling=0.7)
    assert out["evidence"] is None
    assert m.model_summary()["evidential_head"] == "STRUCTURALLY_EXCLUDED"
    assert not m._log_c.requires_grad
    y = (torch.rand(10) > 0.5).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["mechanism_score"], y)
    loss = loss + torch.nn.functional.cross_entropy(out["sdrp_logits"], torch.randint(0, 5, (10,)))
    loss.backward()
    g = lambda sub: sum(  # noqa: E731
        float(p.grad.pow(2).sum()) for n_, p in m.named_parameters()
        if sub in n_ and p.grad is not None
    ) ** 0.5
    assert g("mechanism_head") > 0
    assert g("R_v_rel") > 0 and g("v_scale") > 0
    # mechanism BCE now reaches the hyperbolic spine, not just euc_skip.
    m.zero_grad()
    out = m(s, v, ei, et, tau_ceiling=0.7)
    torch.nn.functional.binary_cross_entropy_with_logits(out["mechanism_score"], y).backward()
    assert g("attn_layers") > 0 or g("projector") > 0


def test_legacy_spine_layout_unchanged() -> None:
    m = _spine()
    assert m.evidential_head is not None and not m.decoupled_arch
    assert m.attn_layers[0].beta is not None
    assert m._log_c.requires_grad


def test_beta_shift_cancels_but_relation_value_path_does_not() -> None:
    """Single-relation neighborhoods: beta is a constant shift inside the softmax."""
    torch.manual_seed(0)
    n, d = 6, 8
    z = torch.randn(n, d) * 0.2
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    et = torch.full((6,), 5)

    legacy = HyperbolicGraphAttention(d)
    base = legacy(z, ei, et)
    with torch.no_grad():
        legacy.beta.weight[5] += 3.0
    assert torch.allclose(base, legacy(z, ei, et), atol=1e-5)

    vp = HyperbolicGraphAttention(d, relation_value_path=True)
    out_r5 = vp(z, ei, et)
    out_r3 = vp(z, ei, torch.full((6,), 3))
    assert not torch.allclose(out_r5, out_r3, atol=1e-6)


def test_salt_first_only_in_decoupled_input_typing() -> None:
    from science.tokyo_eye.v8.r0_r5_graph import R4_SALT_BRIDGE, resolve_layer_b_primary

    kw = dict(has_hbond=True, is_dehydron=False, has_pi_or_hydrophobic=False,
              has_salt=True, has_r5_neighborhood=False)
    assert resolve_layer_b_primary(**kw)[0] == R1_HBOND  # frozen label typing
    assert resolve_layer_b_primary(**kw, salt_first=True)[0] == R4_SALT_BRIDGE
    # secondary flags identical either way
    assert resolve_layer_b_primary(**kw)[1] == resolve_layer_b_primary(**kw, salt_first=True)[1]


def test_label_side_equals_legacy_on_real_loso_structures() -> None:
    """Decoupled input typing (salt-first, 6.5 A R3) must never move labels/SDRP targets."""
    import sys
    from pathlib import Path

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        import wrap1_zhyp_g_fit as G
        from science.tokyo_eye.v8.loader import (
            ca_coords,
            chemistry_gate_features,
            ensure_pdb_cached,
            parse_residue_records_from_pdb_chain,
            sdrp_heuristic_from_edges,
        )

        tags = G._all_structure_tags()[:3]
        loaded = []
        for t in tags:
            pdb, ch = G._parse_tag(t)
            recs = [
                r for r in parse_residue_records_from_pdb_chain(
                    ensure_pdb_cached(pdb, G.PDB_DIR), ch)
                if r.get_atom("CA") is not None
            ]
            loaded.append(recs)
    except Exception as exc:  # noqa: BLE001 - PDBs unavailable offline
        pytest.skip(f"real structures unavailable: {exc}")

    for recs in loaded:
        legacy = build_r0_r5_graph(recs)
        dec = build_r0_r5_graph(recs, decouple_r2_input=True)
        n = legacy.num_nodes
        np.testing.assert_array_equal(legacy.edge_index, dec.label_edge_index)
        np.testing.assert_array_equal(legacy.edge_type, dec.label_edge_type)
        np.testing.assert_array_equal(
            sdrp_heuristic_from_edges(n, legacy.edge_index, legacy.edge_type),
            sdrp_heuristic_from_edges(n, dec.label_edge_index, dec.label_edge_type),
        )
        assert not np.any(dec.edge_type == R2_DEHYDRON)
        chem = chemistry_gate_features(n, dec.edge_index, dec.edge_type, ca_coords(recs))
        assert float(np.abs(chem[:, 1]).max()) == 0.0  # tau column must not carry the label
        assert int((dec.edge_type == 3).sum()) >= int((legacy.edge_type == 3).sum())
