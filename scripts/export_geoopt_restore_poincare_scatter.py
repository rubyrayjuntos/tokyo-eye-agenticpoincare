"""Poincaré disc scatter panels for tokyo_eye_equ_geoopt_restore proteins."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    load_boot_split,
)
from experiments.training.v8.export_viewers import ball_to_disc
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import (
    _load_train_batch,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

BEST = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt")
OUT_DIR = Path("data/local_objects/gnn_viewer/equ_geoopt_restore")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")


def _forward_z(system, batch, tau: float):
    system.eval()
    with torch.no_grad():
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau,
        )
    z = out["z_hyp"].detach().float().cpu()
    labels = batch["dehydron_labels"].detach().float().cpu().numpy()
    r = torch.linalg.vector_norm(z, dim=-1).numpy()
    return z, labels, r, out


def _panel(ax, xy, color, title: str, *, cmap="coolwarm", vmin=0.0, vmax=1.0, cbar_label=""):
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="#888", lw=1.2))
    ax.add_patch(Circle((0, 0), 0.5, fill=False, color="#444", lw=0.6, ls="--"))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color,
        s=18,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color="#eee")
    ax.set_facecolor("#111")
    for spine in ax.spines.values():
        spine.set_color("#333")
    return sc


def main() -> None:
    set_dehydron_wrap_max(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digest = assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, _ = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(BEST, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    print("loaded", BEST, "epoch", blob.get("epoch"))

    split = load_boot_split(DEFAULT_MANIFEST)
    entries = (
        [("train", e) for e in split["train"]]
        + [("probe", e) for e in split["probe"]]
    )
    resolved = [(role, _resolve_entry(e, PDB_DIR)) for role, e in entries]

    # Per-protein dehydron-colored discs
    n = len(resolved)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), facecolor="#0d0d0d"
    )
    axes = np.atleast_2d(axes)
    payloads = []
    for i, (role, entry) in enumerate(resolved):
        ax = axes[i // ncols, i % ncols]
        batch = _load_train_batch(
            entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
        )
        z, dehydron, r_ball, out = _forward_z(system, batch, RV_TAU_END)
        xy = ball_to_disc(z)
        sid = f"{entry['pdb_id']}:{entry['chain']}"
        title = f"{sid} [{role}]\n|z|̄={float(r_ball.mean()):.3f} dh={float(dehydron.mean()):.2f}"
        sc = _panel(ax, xy, dehydron, title, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
        payloads.append(
            {
                "role": role,
                "pdb_id": entry["pdb_id"],
                "chain": entry["chain"],
                "n": int(z.shape[0]),
                "mean_ball_r": float(r_ball.mean()),
                "std_ball_r": float(r_ball.std()),
                "dehydron_frac": float(dehydron.mean()),
                "mean_disc_r": float(np.linalg.norm(xy, axis=1).mean()),
            }
        )
        print("ok", sid, payloads[-1]["n"], payloads[-1]["mean_ball_r"])
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
        axes[j // ncols, j % ncols].set_facecolor("#0d0d0d")

    fig.suptitle(
        "tokyo_eye_equ_geoopt_restore — Poincaré disc (z_hyp→PCA→disc)\n"
        f"best epoch {blob.get('epoch')} | frontend {digest[:12]}… | color=dehydron",
        color="#eee",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("dehydron", color="#ccc")
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ccc")

    panel_path = OUT_DIR / "poincare_disc_panel_dehydron.png"
    fig.savefig(panel_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("wrote", panel_path)

    # Second panel: color by ball radius
    fig2, axes2 = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), facecolor="#0d0d0d"
    )
    axes2 = np.atleast_2d(axes2)
    last_sc = None
    for i, (role, entry) in enumerate(resolved):
        ax = axes2[i // ncols, i % ncols]
        batch = _load_train_batch(
            entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
        )
        z, dehydron, r_ball, _ = _forward_z(system, batch, RV_TAU_END)
        xy = ball_to_disc(z)
        sid = f"{entry['pdb_id']}:{entry['chain']}"
        title = f"{sid} [{role}]\n|z|̄={float(r_ball.mean()):.3f}"
        last_sc = _panel(
            ax, xy, r_ball, title, cmap="viridis", vmin=0.0, vmax=max(0.5, float(r_ball.max()))
        )
    for j in range(n, nrows * ncols):
        axes2[j // ncols, j % ncols].axis("off")
        axes2[j // ncols, j % ncols].set_facecolor("#0d0d0d")
    fig2.suptitle(
        "tokyo_eye_equ_geoopt_restore — Poincaré disc colored by ‖z‖₂\n"
        f"best epoch {blob.get('epoch')} | QUALIFIED geometry HOLD",
        color="#eee",
        fontsize=12,
        y=0.995,
    )
    fig2.tight_layout(rect=[0, 0.02, 1, 0.96])
    cbar2 = fig2.colorbar(last_sc, ax=axes2.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar2.set_label("‖z‖₂", color="#ccc")
    cbar2.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="#ccc")
    radius_path = OUT_DIR / "poincare_disc_panel_radius.png"
    fig2.savefig(radius_path, dpi=160, facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close(fig2)
    print("wrote", radius_path)

    meta = {
        "gate_id": "tokyo_eye_equ_geoopt_restore",
        "ckpt": str(BEST),
        "best_epoch": blob.get("epoch"),
        "frontend_sha256": digest,
        "layout": "z_hyp → PCA whitened → Poincaré disc (ball_to_disc)",
        "tau_ceiling": RV_TAU_END,
        "proteins": payloads,
        "panel_dehydron": str(panel_path),
        "panel_radius": str(radius_path),
    }
    (OUT_DIR / "poincare_scatter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("DONE", OUT_DIR)


if __name__ == "__main__":
    main()
