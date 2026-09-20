"""End-to-end preflight smoke for tokyo_eye_equ_geoopt_restore."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT,
    assert_geoopt_restore_preflight,
    assert_pure_hyp_strict,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.loader import ensure_pdb_cached, load_structure_batch
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max


def main() -> None:
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    print("build system...")
    system, info = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    print("load_info", {k: info[k] for k in info if k != "kwargs"}, "kwargs", info.get("kwargs"))
    pf = assert_geoopt_restore_preflight(system)
    print("restore_preflight", pf)
    pure = assert_pure_hyp_strict(system)
    print("pure_hyp", pure.get("pure_hyp_pass"), "live_ok", pure.get("live_ok"))

    pdb_dir = Path("/tmp/dtie_pdb_cache")
    ensure_pdb_cached("4OBE", pdb_dir)
    batch = load_structure_batch("4OBE", "A", pdb_dir=pdb_dir, device=device)
    print("N", batch["num_nodes"], "edges", int(batch["edge_index"].shape[1]))
    system.eval()
    with torch.no_grad():
        out = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=0.8)
    print("out_keys", sorted(out.keys()))
    z = out.get("z_hyp", out.get("z"))
    if z is not None and torch.is_tensor(z):
        r = z.norm(dim=-1)
        print(
            "z",
            tuple(z.shape),
            "r_mean",
            float(r.mean()),
            "r_max",
            float(r.max()),
            "finite",
            bool(torch.isfinite(z).all()),
        )
    else:
        for k, v in out.items():
            if torch.is_tensor(v):
                print("tensor", k, tuple(v.shape), "dtype", v.dtype)
    print("PREFLIGHT_SMOKE_PASS")


if __name__ == "__main__":
    main()
