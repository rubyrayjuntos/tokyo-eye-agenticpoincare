"""
inspect_checkpoints.py — Inspect v6 checkpoint architectures
=============================================================
Loads each checkpoint with torch and reports:
  - Top-level keys (epoch, stage, loss metadata)
  - RadialHead architecture (2-layer vs 3-layer)
  - Total parameter count
  - Whether it matches the current model.py definition

Usage (inside science container):
    python -m experiments.diagnostics.inspect_checkpoints
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

CHECKPOINTS = {
    "v6_best":         "checkpoints/v6/v6_best.pt",
    "v6_phase1_ep50":  "checkpoints/v6/v6_phase1_epoch50.pt",
    "v6_phase2_ep90":  "checkpoints/v6/v6_phase2_epoch90.pt",
    "v6_phase3_ep200": "checkpoints/v6/v6_phase3_epoch200.pt",
}

def inspect(name: str, path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"  MISSING: {path}")
        return {}

    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta  = {k: ckpt[k] for k in
                 ["epoch","stage","val_loss","best_val_loss","train_loss","config"]
                 if k in ckpt}
    else:
        state = ckpt
        meta  = {}

    radial_w = {k: tuple(v.shape) for k, v in state.items()
                if "radial_head" in k and "weight" in k}
    angular_w = {k: tuple(v.shape) for k, v in state.items()
                 if "angular_head" in k and "weight" in k}

    has_net4   = any("radial_head.net.4" in k for k in state)
    arch_label = "3-layer (128→64→32→1)" if has_net4 else "2-layer (128→64→1)"

    total_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))

    # Check if node_emb input dim matches current model (node_dim=4)
    node_emb_key = next((k for k in state if "node_emb" in k and "weight" in k), None)
    node_dim = tuple(state[node_emb_key].shape)[1] if node_emb_key else "?"

    print(f"\n{'='*60}")
    print(f"  {name}  ({p.name}, {p.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*60}")
    print(f"  RadialHead arch : {arch_label}")
    print(f"  Radial weights  : {radial_w}")
    print(f"  node_dim        : {node_dim}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Metadata        : {meta}")
    if angular_w:
        first_k = next(iter(angular_w))
        print(f"  AngularHead[0]  : {angular_w[first_k]}")

    return {"arch": arch_label, "params": total_params, "meta": meta, "node_dim": node_dim}


def main():
    print("\nv6 Checkpoint Architecture Inspector")
    print("Current model.py RadialHead: 3-layer (128→64→32→1)\n")

    results = {}
    for name, path in CHECKPOINTS.items():
        results[name] = inspect(name, path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Name':<22} {'Arch':<30} {'Params':>12}  Match model.py?")
    print(f"  {'-'*72}")
    for name, r in results.items():
        if not r:
            continue
        match = "✓" if "3-layer" in r.get("arch","") else "✗ (needs patch)"
        print(f"  {name:<22} {r.get('arch','?'):<30} {r.get('params',0):>12,}  {match}")


if __name__ == "__main__":
    main()
