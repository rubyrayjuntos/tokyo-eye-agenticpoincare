"""Check whether node_emb width shifts prototype-bank (and other) RNG draws.

Model params are drawn sequentially during ``__init__``. ``node_emb`` is built
*before* the gate/prototype bank. ``Linear(4→H)`` consumes more RNG draws than
``Linear(3→H)``, so the same ``torch.manual_seed(seed)`` can produce unrelated
prototype inits under 3-D vs 4-D — invalidating seed-matched comparisons.

This is a constructor-only check: no training, no data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from science.dtie.v66.gnn.model import GOSPConeMapperV66

# Stack-typical kwargs matching Fix-1 + S4 feeler lineage (topology_only + SASA board).
STACK_KWARGS: dict[str, Any] = {
    "hidden": 128,
    "num_experts": 4,
    "num_layers": 3,
    "hyperbolic_gate": True,
    "topology_only_gate": True,
    "gate_include_sasa": True,
    "gate_disc_input": True,
}


def _build(node_dim: int, seed: int) -> GOSPConeMapperV66:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # init_seed isolates gate/prototype from node_emb width (the fix under test).
    return GOSPConeMapperV66(node_dim=node_dim, init_seed=seed, **STACK_KWARGS)


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = (a - b).abs()
    return {
        "shape": list(a.shape),
        "identical": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        ),
    }


def compare_inits(seed: int = 1) -> dict[str, Any]:
    m3 = _build(3, seed)
    m4 = _build(4, seed)

    keys = [
        ("node_emb.weight", m3.node_emb.weight, m4.node_emb.weight[:, :3]),  # truncated view
        ("prototype_tangent", m3.gate.prototype_bank.prototype_tangent, m4.gate.prototype_bank.prototype_tangent),
        ("expert_bias", m3.gate.expert_bias, m4.gate.expert_bias),
        ("logit_scale", m3.gate.logit_scale, m4.gate.logit_scale),
        ("topo_encoder.0.weight", m3.gate.topo_encoder[0].weight, m4.gate.topo_encoder[0].weight),
        ("log_c", m3.log_c, m4.log_c),
        ("experts.0.0.weight", m3.experts[0][0].weight, m4.experts[0][0].weight),
        ("projection_head.weight", m3.projection_head.weight, m4.projection_head.weight),
    ]

    # First conv layer if present (heavy RNG consumer between node_emb and gate).
    conv_cmp: dict[str, Any] | None = None
    if hasattr(m3.convs[0], "linear") or len(list(m3.convs[0].parameters())) > 0:
        p3 = list(m3.convs[0].parameters())
        p4 = list(m4.convs[0].parameters())
        if p3 and p4 and p3[0].shape == p4[0].shape:
            conv_cmp = _tensor_stats(p3[0], p4[0])

    comparisons: dict[str, Any] = {}
    for name, t3, t4 in keys:
        if t3.shape != t4.shape:
            comparisons[name] = {
                "shape_3d": list(t3.shape),
                "shape_4d": list(t4.shape),
                "identical": False,
                "note": "shape mismatch (expected for node_emb.weight full tensors)",
            }
            continue
        comparisons[name] = _tensor_stats(t3, t4)

    # Full node_emb shapes (not truncated) for documentation.
    comparisons["node_emb.weight_full_shapes"] = {
        "node_dim_3": list(m3.node_emb.weight.shape),
        "node_dim_4": list(m4.node_emb.weight.shape),
        "extra_draws_in_4d_weight": int(
            m4.node_emb.weight.numel() - m3.node_emb.weight.numel()
        ),
        "extra_draws_in_4d_bias": int(m4.node_emb.bias.numel() - m3.node_emb.bias.numel()),
    }

    proto = comparisons["prototype_tangent"]
    topo = comparisons["topo_encoder.0.weight"]
    confound = not proto.get("identical", False)
    return {
        "seed": seed,
        "stack_kwargs": {**STACK_KWARGS, "init_seed": seed},
        "construction_order_note": (
            "GOSPConeMapperV66: node_emb (Linear) → EquivariantConv×N → log_c → "
            "radial/angular → HyperbolicPrototypeGate (init_seed-isolated) → "
            "experts → heads"
        ),
        "first_conv_param0": conv_cmp,
        "comparisons": comparisons,
        "verdict": (
            "RNG_STREAM_SHIFT_CONFIRMED"
            if confound
            else "PROTOTYPES_IDENTICAL_ACROSS_NODE_DIM"
        ),
        "gate_topo_encoder_identical": bool(topo.get("identical")),
        "implication": (
            "Same seed + different node_emb width ⇒ unrelated prototype (and "
            "downstream) inits. Prior 3-D vs 4-D seed-matched comparisons are "
            "uncontrolled for init."
            if confound
            else "Prototype bank (and gate Linear) match under init_seed isolation. "
            "Trunk/expert inits may still differ with node_emb width (expected); "
            "gate/prototype comparisons across widths are now controlled."
        ),
    }


def main() -> None:
    out = Path(
        "/app/checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/"
        "node_emb_width_rng_shift_check.json"
    )
    reports = {f"seed_{s}": compare_inits(s) for s in (1, 2)}
    # Sanity: same node_dim twice must match.
    m_a = _build(3, 1)
    m_b = _build(3, 1)
    same_dim_ok = torch.equal(
        m_a.gate.prototype_bank.prototype_tangent,
        m_b.gate.prototype_bank.prototype_tangent,
    )
    report = {
        "tag": "NODE_EMB_WIDTH_RNG_SHIFT_CHECK",
        "same_node_dim_repro_identical": same_dim_ok,
        "by_seed": reports,
        "headline": {
            "seed1": reports["seed_1"]["verdict"],
            "seed2": reports["seed_2"]["verdict"],
            "any_confound": any(
                r["verdict"] == "RNG_STREAM_SHIFT_CONFIRMED" for r in reports.values()
            ),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", out)
    print(json.dumps(report["headline"], indent=2))
    for s, r in reports.items():
        p = r["comparisons"]["prototype_tangent"]
        print(
            s,
            r["verdict"],
            "proto identical=",
            p.get("identical"),
            "max_abs_diff=",
            p.get("max_abs_diff"),
            "cosine=",
            p.get("cosine"),
            "extra_weight_draws=",
            r["comparisons"]["node_emb.weight_full_shapes"]["extra_draws_in_4d_weight"],
        )


if __name__ == "__main__":
    main()
