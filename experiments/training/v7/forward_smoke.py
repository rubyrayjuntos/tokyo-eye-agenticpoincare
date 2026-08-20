#!/usr/bin/env python3
"""Minimal Tokyo Eye v7 forward smoke — writes under checkpoints/v7/diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.data import Data

from science.tokyo_eye.TokyoEye import TokyoEye


def main() -> int:
    out = Path("checkpoints/v7/diagnostics/tokyo_eye_v7_forward_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    n, e = 8, 16
    model = TokyoEye(
        node_dim=4,
        hidden=32,
        num_layers=2,
        num_experts=2,
        hyp_mp_primary=True,
        se3_aux=False,
        hyp_mp_layers=2,
        hyperbolic_gate=True,
        hyperbolic_expert_mix=True,
    )
    model.eval()
    data = Data(
        x=torch.randn(n, 4),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, 4),
        clustering=torch.rand(n),
        degree=torch.ones(n),
        rho=torch.rand(n),
        ss_onehot=torch.zeros(n, 3),
    )
    data.ss_onehot[:, 2] = 1.0
    with torch.no_grad():
        out_fwd = model(data)

    c = float(model.curvature.detach().cpu())
    depth = out_fwd["cone_depth"].detach().cpu()
    report = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_forward_smoke",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "module": "science.tokyo_eye.TokyoEye",
        "hyp_mp_primary": True,
        "curvature_learned": c,
        "cone_depth_mean": float(depth.mean()),
        "cone_depth_std": float(depth.std()),
        "audit_hyp_mp": bool(out_fwd["audit_trail"].get("hyp_mp_primary")),
        "pass": bool(
            out_fwd["audit_trail"].get("hyp_mp_primary")
            and depth.numel() == n
            and c == c  # finite
        ),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
