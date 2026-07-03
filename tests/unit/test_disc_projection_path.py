"""v5.5 forward-pass contract: disc_projection_path + pre/post audit tensors."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from science.dtie.v6.gnn.model import (
    GOSPConeMapperV6,
    resolve_disc_projection_path,
    resolve_disc_radial_source,
)


def _minimal_graph(n: int = 8) -> Data:
    x = torch.randn(n, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 4)
    edge_attr[:, :3] = edge_attr[:, :3] / (edge_attr[:, :3].norm(dim=-1, keepdim=True) + 1e-8)
    edge_attr[:, 3:4] = edge_attr[:, :3].norm(dim=-1, keepdim=True).clamp(min=1e-3)
    clustering = torch.rand(n)
    degree = torch.rand(n) * 5 + 1
    rho = torch.rand(n)
    ss_onehot = torch.zeros(n, 3)
    ss_onehot[:, 0] = 1.0
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        clustering=clustering,
        degree=degree,
        rho=rho,
        ss_onehot=ss_onehot,
    )


def test_resolve_disc_projection_path() -> None:
    assert resolve_disc_projection_path(legacy_disc_projection=True) == "post_routing"
    assert resolve_disc_projection_path(legacy_disc_projection=False) == "pre_routing"
    assert resolve_disc_projection_path(disc_projection_path="pre_routing") == "pre_routing"


def test_forward_emits_pre_and_post_disc_tensors() -> None:
    model = GOSPConeMapperV6(legacy_disc_projection=False)
    with torch.inference_mode():
        out = model(_minimal_graph())
    assert out["disc_path_used"] == "pre_routing"
    assert out["hyp_projections_2d_pre"].shape == out["hyp_projections_2d_post"].shape
    assert torch.equal(out["hyp_projections_2d"], out["hyp_projections_2d_pre"])


def test_post_routing_active_path_uses_legacy_teacher() -> None:
    model = GOSPConeMapperV6(disc_projection_path="post_routing")
    with torch.inference_mode():
        out = model(_minimal_graph())
    assert out["disc_path_used"] == "post_routing"
    assert torch.equal(out["hyp_projections_2d"], out["hyp_projections_2d_legacy_teacher"])
    assert out["gate_mode"] == "hyperbolic"


def test_resolve_disc_radial_source_defaults_and_validates() -> None:
    assert resolve_disc_radial_source(None) == "mobius"
    assert resolve_disc_radial_source("dist0_x_hyp") == "dist0_x_hyp"
    try:
        resolve_disc_radial_source("invalid")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_disc_radial_override_emits_audit_and_expands_radius() -> None:
    mobius = GOSPConeMapperV6(legacy_disc_projection=False, disc_radial_source="mobius")
    dist0 = GOSPConeMapperV6(legacy_disc_projection=False, disc_radial_source="dist0_x_hyp")
    dist0.load_state_dict(mobius.state_dict(), strict=False)
    data = _minimal_graph(n=64)
    with torch.inference_mode():
        out_m = mobius(data)
        out_d = dist0(data)
    assert out_d["audit_trail"]["disc_radial_source"] == "dist0_x_hyp"
    r_m = out_m["hyp_projections_2d_pre"].norm(dim=-1).median()
    r_d = out_d["hyp_projections_2d_pre"].norm(dim=-1).median()
    assert float(r_d) > float(r_m)
    assert not torch.isnan(out_d["hyp_projections_2d_pre"]).any()


def test_post_routing_disc_radial_override_uses_depth_routed() -> None:
    """Lever A₂: post override scales MobiusLinear direction by depth_routed."""
    from science.dtie.v6.gnn.hyperbolic_moe import project_disc_2d
    from science.training.disc_occupancy import disc_line_thickness_from_tensor

    model = GOSPConeMapperV6(legacy_disc_projection=False, disc_radial_source="radial_depth")
    with torch.inference_mode():
        n = 48
        c = model.curvature
        k = -c
        x_routed_hyp = torch.randn(n, 128) * 0.05
        depth_routed = torch.linspace(0.15, 0.55, n).unsqueeze(-1)
        raw_mobius = model.hyp_proj_head_2d(x_routed_hyp, c=c)
        raw_override = model._post_routing_disc_raw(
            x_routed_hyp,
            depth_routed=depth_routed,
            c=c,
            k=k,
        )
        soft_m, _ = project_disc_2d(raw_mobius, k=k, softness=0.95)
        soft_o, _ = project_disc_2d(raw_override, k=k, softness=0.95)
        thick_m = float(disc_line_thickness_from_tensor(soft_m.float()).item())
        thick_o = float(disc_line_thickness_from_tensor(soft_o.float()).item())
    assert thick_o > thick_m
    assert float(soft_o.norm(dim=-1).median()) > float(soft_m.norm(dim=-1).median())
