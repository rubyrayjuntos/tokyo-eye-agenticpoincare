"""Gate/prototype init must be stable across node_emb width under the same seed."""
from __future__ import annotations

import torch

from science.dtie.common.isolated_init import derived_seed, isolated_torch_seed
from science.dtie.v66.gnn.model import GOSPConeMapperV66


STACK_KWARGS = {
    "hidden": 128,
    "num_experts": 4,
    "num_layers": 3,
    "hyperbolic_gate": True,
    "topology_only_gate": True,
    "gate_include_sasa": True,
    "gate_disc_input": True,
}


def _build(node_dim: int, seed: int, *, init_seed: int | None) -> GOSPConeMapperV66:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return GOSPConeMapperV66(node_dim=node_dim, init_seed=init_seed, **STACK_KWARGS)


def test_derived_seed_stable_and_domain_separated() -> None:
    a = derived_seed("hyperbolic_prototype_gate", 1)
    b = derived_seed("hyperbolic_prototype_gate", 1)
    c = derived_seed("hyperbolic_prototype_bank", 1)
    d = derived_seed("hyperbolic_prototype_gate", 2)
    assert a == b
    assert a != c
    assert a != d


def test_isolated_torch_seed_restores_global_stream() -> None:
    torch.manual_seed(0)
    before = torch.randn(4).clone()
    torch.manual_seed(0)
    with isolated_torch_seed(999):
        _ = torch.randn(8)
    after = torch.randn(4)
    assert torch.equal(before, after)


def test_prototype_bank_identical_across_node_dim_with_init_seed() -> None:
    m3 = _build(3, seed=1, init_seed=1)
    m4 = _build(4, seed=1, init_seed=1)
    p3 = m3.gate.prototype_bank.prototype_tangent
    p4 = m4.gate.prototype_bank.prototype_tangent
    assert torch.equal(p3, p4), (
        f"prototype_tangent diverged under node_dim 3 vs 4 "
        f"(max |Δ|={(p3 - p4).abs().max().item()})"
    )
    # Gate Linear inits should also match (isolated stream for whole gate).
    assert torch.equal(
        m3.gate.topo_encoder[0].weight, m4.gate.topo_encoder[0].weight
    )


def test_without_init_seed_prototypes_still_shift_with_node_dim() -> None:
    """Regression guard: the confound remains if init_seed is omitted."""
    m3 = _build(3, seed=1, init_seed=None)
    m4 = _build(4, seed=1, init_seed=None)
    p3 = m3.gate.prototype_bank.prototype_tangent
    p4 = m4.gate.prototype_bank.prototype_tangent
    assert not torch.equal(p3, p4)


def test_different_init_seeds_yield_different_prototypes() -> None:
    m_a = _build(3, seed=0, init_seed=1)
    m_b = _build(3, seed=0, init_seed=2)
    assert not torch.equal(
        m_a.gate.prototype_bank.prototype_tangent,
        m_b.gate.prototype_bank.prototype_tangent,
    )
