"""
inspect_checkpoints.py — Inspect all checkpoint architectures
=============================================================
Usage (inside science container):
    python -m experiments.diagnostics.inspect_checkpoints
"""
from __future__ import annotations
from pathlib import Path
import torch

CHECKPOINT_DIRS = [
    "checkpoints_v5",
    "checkpoints_v5_retrained",
    "checkpoints_v6",
    "checkpoints_v6_gumbel",
    "checkpoints_v6_hybrid",
    "checkpoints_v6_topo",
    "checkpoints/v5",
    "checkpoints/v6",
]


def inspect_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": str(e)}

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        return {"error": "unexpected format"}

    meta = {k: ckpt[k] for k in
            ["epoch", "stage", "val_loss", "best_val_loss", "train_loss"]
            if isinstance(ckpt, dict) and k in ckpt}

    has_net4    = any("radial_head.net.4" in k for k in state)
    arch        = "3-layer RadialHead" if has_net4 else "2-layer RadialHead"
    total       = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    node_emb_k  = next((k for k in state if "node_emb" in k and "weight" in k), None)
    node_dim    = tuple(state[node_emb_k].shape)[1] if node_emb_k else "?"
    hidden      = tuple(state[node_emb_k].shape)[0] if node_emb_k else "?"
    num_experts = len([k for k in state if k.startswith("experts.") and k.endswith(".0.weight")])

    # Check for any v6-specific keys not present in v5
    has_topo    = any("topo" in k.lower() for k in state)
    has_gumbel  = any("gumbel" in k.lower() for k in state)

    return {
        "arch":        arch,
        "params":      total,
        "node_dim":    node_dim,
        "hidden":      hidden,
        "num_experts": num_experts,
        "has_topo":    has_topo,
        "has_gumbel":  has_gumbel,
        "meta":        meta,
        "matches_modelpy": has_net4,
    }


def main():
    print("\nCheckpoint Architecture Inspector — All Versions")
    print("=" * 72)
    print(f"  Current model.py: 3-layer RadialHead (128→64→32→1)\n")

    all_results = []

    for dir_str in CHECKPOINT_DIRS:
        d = Path(dir_str)
        if not d.exists():
            continue
        pts = sorted(d.glob("*.pt"))
        if not pts:
            continue
        print(f"\n  [{dir_str}]")
        for pt in pts:
            r = inspect_file(pt)
            if r is None:
                continue
            if "error" in r:
                print(f"    {pt.name:<40} ERROR: {r['error']}")
                continue
            match = "✓" if r["matches_modelpy"] else "✗ needs patch"
            extras = []
            if r["has_topo"]:   extras.append("topo")
            if r["has_gumbel"]: extras.append("gumbel")
            extra_str = f" [{','.join(extras)}]" if extras else ""
            meta_str = ""
            if r["meta"]:
                m = r["meta"]
                parts = []
                if "epoch" in m:       parts.append(f"ep={m['epoch']}")
                if "stage" in m:       parts.append(f"stage={m['stage']}")
                if "val_loss" in m:    parts.append(f"val={m['val_loss']:.4f}")
                if "best_val_loss" in m: parts.append(f"best={m['best_val_loss']:.4f}")
                meta_str = "  " + " ".join(parts)
            print(f"    {pt.name:<40} {r['arch']:<22} "
                  f"node_dim={r['node_dim']} hidden={r['hidden']} "
                  f"experts={r['num_experts']} {match}{extra_str}{meta_str}")
            all_results.append((dir_str, pt.name, r))

    print(f"\n{'='*72}")
    print("RECOMMENDATION")
    print(f"{'='*72}")

    # Find best candidate: v6 checkpoints that match model.py
    v6_matches    = [(d, n, r) for d, n, r in all_results if "v6" in d and r.get("matches_modelpy")]
    v6_no_match   = [(d, n, r) for d, n, r in all_results if "v6" in d and not r.get("matches_modelpy")]
    best_variants = [(d, n, r) for d, n, r in all_results if "best" in n]

    if v6_matches:
        print(f"  v6 checkpoints matching current model.py (3-layer RadialHead):")
        for d, n, r in v6_matches:
            print(f"    {d}/{n}")
    elif v6_no_match:
        print(f"  All v6 checkpoints use 2-layer RadialHead (patch required, same as v5)")
        print(f"  Best candidate for diagnostics:")
        for d, n, r in best_variants:
            if "v6" in d:
                print(f"    {d}/{n}  (params={r['params']:,})")
    print()


if __name__ == "__main__":
    main()
