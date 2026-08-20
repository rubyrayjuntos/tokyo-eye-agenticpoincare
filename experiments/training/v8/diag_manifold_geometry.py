#!/usr/bin/env python3
"""Extract raw hyperbolic geometry from a v8 checkpoint (no HTML).

Example:
  PYTHONPATH=. python -m experiments.training.v8.diag_manifold_geometry \\
    --ckpt checkpoints/v8/runs/tokyo_eye_v8_stage_a_mode_b/v8_best.pt \\
    --pdb 4OBE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    apply_weight_map,
    load_weight_map,
)
from science.tokyo_eye.v8.loader import DEFAULT_CHAIN, DEFAULT_PDB_DIR, load_structure_batch
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8


def _svd_report(z: np.ndarray) -> dict:
    centered = z - z.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    ev = (s ** 2)
    ev = ev / max(float(ev.sum()), 1e-12)
    r = np.linalg.norm(z, axis=1)
    return {
        "shape": list(z.shape),
        "unique_rows_4dp": len({tuple(np.round(row, 4)) for row in z}),
        "radius_mean": float(r.mean()),
        "radius_std": float(r.std()),
        "radius_min": float(r.min()),
        "radius_max": float(r.max()),
        "top5_singular_values": [float(x) for x in s[:5]],
        "top5_explained_var_pct": [100.0 * float(x) for x in ev[:5]],
        "pc1_pct": 100.0 * float(ev[0]),
        "pc2_pct": 100.0 * float(ev[1]) if len(ev) > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="v8 manifold geometry diagnostic (no HTML)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pdb", type=str, default="4OBE")
    p.add_argument("--chain", type=str, default=DEFAULT_CHAIN)
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    cfg = load_weight_map(args.weight_map)
    device = torch.device(args.device)
    batch = load_structure_batch(
        args.pdb, args.chain, pdb_dir=args.pdb_dir, device=device
    )
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
        live_backbone=True,
    )
    equi = Path(cfg.get("checkpoint_path_default", "checkpoints/v8/pretrained/equiformer_v3_baseline.pt"))
    if equi.is_file():
        apply_weight_map(frontend, equi, cfg)
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=2,
        num_sdrp_classes=5,
        c=float(cfg.get("curvature_c", 1.0)),
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    system.load_state_dict(state, strict=False)
    system.eval()

    with torch.no_grad():
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=float(cfg.get("tau_end", 0.995)),
        )

    report = {"ckpt": str(args.ckpt), "pdb": args.pdb, "chain": args.chain, "tensors": {}}
    for key in ("z_hyp", "z_attn", "z_lift", "h_euc"):
        z = out[key].detach().cpu().numpy()
        report["tensors"][key] = _svd_report(z)
        print(f"\n[{key}]")
        for k, v in report["tensors"][key].items():
            print(f"  {k}: {v}")

    # Verdict heuristic
    zh = report["tensors"]["z_hyp"]
    za = report["tensors"]["z_attn"]
    if zh["pc1_pct"] > 95 and zh["pc2_pct"] < 3:
        verdict = "model_near_1d_collapse_on_z_hyp"
    elif za["pc1_pct"] > 95 and zh["pc2_pct"] >= 10:
        verdict = "viewer_layout_quirk_prefer_z_hyp"
    else:
        verdict = "geometry_ok_or_mild_anisotropy"
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    out_dir = args.out_dir or (args.ckpt.parent / f"diag_{args.pdb.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_hyp.npy", out["z_hyp"].cpu().numpy())
    np.save(out_dir / "z_attn.npy", out["z_attn"].cpu().numpy())
    (out_dir / "geometry_summary.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z = out["z_hyp"].cpu().numpy()
        r = np.linalg.norm(z, axis=1)
        Xc = z - z.mean(0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        pca2 = Xc @ vt[:2].T
        labels = batch["dehydron_labels"].cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].hist(r, bins=20, color="royalblue", edgecolor="black")
        axes[0].set_title(f"{args.pdb} |z_hyp|")
        axes[0].set_xlabel("radius")
        axes[1].scatter(pca2[:, 0], pca2[:, 1], c=labels, cmap="coolwarm", s=14)
        axes[1].set_title("PCA(z_hyp) — raw, not HTML disc")
        axes[1].set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_dir / "geometry_diag.png", dpi=140)
        print(f"Wrote {out_dir / 'geometry_diag.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib plot skipped: {exc}")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
