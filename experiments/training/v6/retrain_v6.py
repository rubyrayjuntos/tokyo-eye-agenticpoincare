"""
retrain_v6.py — Retrain GNNv6 from v6_topo checkpoint with boundary saturation fix.

The v6_topo checkpoint was trained without a tangent-vector norm constraint, causing
100% of nodes to saturate at the Poincaré ball boundary (projection_applied_fraction=1.0).
This makes cone_depth a constant for every residue.

The fix (tanh rescaling of tangent_vector before expmap0) is already committed to
science/dtie/v5/gnn/model.py. This script warm-starts from the v6_topo checkpoint and
runs the same 3-stage curriculum so the RadialHead can recalibrate its output range
to fill the ball non-uniformly.

Usage:
    python -m experiments.training.v6.retrain_v6 \\
        --checkpoint checkpoints_v6_topo/v6_best.pt \\
        --output_dir checkpoints_v6_retrained \\
        --pdb_dir /tmp/dtie_pdb_cache \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_v6")


# ── Gate patch (mirrors shell_signal_diagnostics._patch_gate_if_needed) ──────

def _patch_gate_if_needed(state: dict) -> bool:
    """Replace TopologicalMoEGate in the model module with a v6-compatible variant."""
    import torch.nn.functional as F
    from science.dtie.v5 import gnn as _gnn_pkg
    import science.dtie.v5.gnn.model as _model_mod

    has_gate_net = any(k.startswith("gate.gate_net.") for k in state)
    has_topo = any(k.startswith("gate.topology_compressor.") for k in state)
    if not has_gate_net or has_topo:
        return False  # already compatible

    # Infer gate_net input dim and layer shapes from checkpoint
    gate_input_dim: Optional[int] = None
    if "gate.gate_net.0.weight" in state:
        gate_input_dim = state["gate.gate_net.0.weight"].shape[1]

    if gate_input_dim is None:
        logger.warning("Cannot detect gate input_dim — skipping gate patch")
        return False

    gate_layer_shapes: List[tuple] = []
    for idx in range(0, 20, 2):
        w_key = f"gate.gate_net.{idx}.weight"
        if w_key not in state:
            break
        gate_layer_shapes.append(tuple(state[w_key].shape))

    # Running-stats buffers present?
    has_stats = "gate.degree_mean" in state

    _input_dim = gate_input_dim
    _layer_shapes = gate_layer_shapes
    _has_stats = has_stats

    class _V6MoEGate(nn.Module):
        def __init__(self, hidden_dim: int, num_experts: int):
            super().__init__()
            self._input_dim = _input_dim
            layers: List[nn.Module] = []
            for i, (out_f, in_f) in enumerate(_layer_shapes):
                layers.append(nn.Linear(in_f, out_f))
                if i < len(_layer_shapes) - 1:
                    layers.append(nn.SiLU())
            self.gate_net = nn.Sequential(*layers)
            if _has_stats:
                self.register_buffer("degree_mean", torch.zeros(()))
                self.register_buffer("degree_var",  torch.ones(()))
                self.register_buffer("rho_mean",    torch.zeros(()))
                self.register_buffer("rho_var",     torch.ones(()))
                self.register_buffer("num_updates", torch.zeros(()))

        def forward(self, x_tangent: torch.Tensor,
                    clustering: torch.Tensor,
                    depth: torch.Tensor,
                    data=None) -> tuple:
            hidden_dim = x_tangent.shape[-1]

            clust = clustering.unsqueeze(-1) if clustering.dim() == 1 else clustering
            depth_f = depth if depth.dim() == 2 else depth.unsqueeze(-1)

            if data is not None and hasattr(data, "x"):
                node_feats = data.x
            else:
                node_feats = torch.zeros(x_tangent.shape[0], 4, device=x_tangent.device)

            if hasattr(self, "degree_mean"):
                rho_raw = node_feats[:, 0:1]
                deg = node_feats.shape[1]
                deg_t = torch.full((x_tangent.shape[0], 1), float(deg),
                                   device=x_tangent.device)
                dv = self.degree_var.clamp(min=1e-8)
                rv = self.rho_var.clamp(min=1e-8)
                deg_norm = (deg_t - self.degree_mean) / dv.sqrt()
                rho_norm = (rho_raw - self.rho_mean) / rv.sqrt()
                _ = rho_norm  # available but not always used below

            if self._input_dim >= hidden_dim:
                parts = [x_tangent, node_feats, clust, depth_f, deg_norm]
            else:
                parts = [node_feats, clust, depth_f, deg_norm]

            gate_in = torch.cat(parts, dim=-1)
            gate_in = gate_in[:, :self._input_dim]
            logits = self.gate_net(gate_in)
            weights = torch.softmax(logits, dim=-1)

            # Soft balance loss
            mean_w = weights.mean(dim=0)
            balance = (mean_w * torch.log(mean_w.clamp(min=1e-8))).sum()
            return weights, balance

    _model_mod.TopologicalMoEGate = _V6MoEGate
    if hasattr(_gnn_pkg, "TopologicalMoEGate"):
        _gnn_pkg.TopologicalMoEGate = _V6MoEGate

    logger.info("Gate patched to v6 stats-based variant (input_dim=%d)", gate_input_dim)
    return True


def _patch_radial_head_if_needed(state: dict) -> bool:
    """Patch to 2-layer RadialHead if checkpoint has 2-layer weights."""
    import science.dtie.v5.gnn.model as _model_mod
    import torch.nn.functional as F

    if any("radial_head.net.4" in k for k in state):
        return False  # already 3-layer

    class _RadialHead2L(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.radial_scale = nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raw = self.net(x)
            scale = F.softplus(self.radial_scale)
            return F.softplus(raw) * scale

    _model_mod.RadialHead = _RadialHead2L
    logger.info("RadialHead patched to 2-layer variant")
    return True


def load_checkpoint(checkpoint_path: Path, device: str):
    """Load v6 checkpoint with all necessary patches applied."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        weights = state["model_state_dict"]
    else:
        weights = state

    radial_patched = _patch_radial_head_if_needed(weights)
    gate_patched = _patch_gate_if_needed(weights)

    from science.dtie.v5.gnn.model import GOSPConeMapper
    model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
    model.load_state_dict(weights)

    # Re-initialise the RadialHead so it forgets the ρ-trained mapping and
    # learns SASA from scratch, while the backbone/gate/angular stay intact.
    for m in model.radial_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    nn.init.zeros_(model.radial_head.radial_scale)
    logger.info("RadialHead weights re-initialised (backbone/gate/angular preserved)")

    rs = float(model.radial_head.radial_scale.item())
    import torch.nn.functional as F
    logger.info("Loaded checkpoint | radial_scale=%.4f → softplus=%.4f | gate_patched=%s",
                rs, F.softplus(torch.tensor(rs)).item(), gate_patched)

    return model, gate_patched


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model,
    optimizer,
    proteins: List[Dict],
    loss_coeffs: Dict[str, float],
    gate_patched: bool,
    freeze_radial: bool = False,
    freeze_angular: bool = False,
    device: str = "cpu",
) -> Dict[str, float]:
    from science.dtie.v5.gnn.model import gosp_loss_v5

    model.train()
    for p in model.radial_head.parameters():
        p.requires_grad = not freeze_radial
    for p in model.angular_head.parameters():
        p.requires_grad = not freeze_angular

    epoch_losses = {k: [] for k in
                    ["total", "evidential", "balance", "cone_consistency",
                     "neighborhood_consistency", "angular_diversity",
                     "domain_separation_2d", "domain_separation_3d"]}
    grad_norms = {"radial": [], "angular": [], "backbone": []}

    for prot in proteins:
        data = prot["data"].to(device)

        # Inject data into gate forward if gate was patched
        if gate_patched:
            _orig = model.gate.forward
            model.gate.forward = lambda xt, cl, cd: _orig(xt, cl, cd, data=data)

        target_rho = prot["target_rho"].to(device)
        target_dehydron = prot["target_dehydron"].to(device)
        target_sasa = prot["target_sasa"].to(device)
        ca_coords = prot["ca_coords"].to(device)
        domain_labels = prot.get("domain_labels")
        if domain_labels is not None:
            domain_labels = domain_labels.to(device)

        optimizer.zero_grad()
        output = model(data)

        if gate_patched:
            model.gate.forward = _orig

        losses = gosp_loss_v5(
            output=output,
            target_rho=target_rho,
            target_dehydron=target_dehydron,
            ca_coords=ca_coords,
            domain_labels=domain_labels,
            target_sasa=target_sasa,
            **loss_coeffs,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for p in model.radial_head.parameters():
            if p.grad is not None:
                grad_norms["radial"].append(p.grad.norm().item())
        for p in model.angular_head.parameters():
            if p.grad is not None:
                grad_norms["angular"].append(p.grad.norm().item())
        for p in model.convs.parameters():
            if p.grad is not None:
                grad_norms["backbone"].append(p.grad.norm().item())

        for k in epoch_losses:
            if k in losses:
                v = losses[k]
                epoch_losses[k].append(v.item() if torch.is_tensor(v) else float(v))

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
    result["grad_radial"] = float(np.mean(grad_norms["radial"])) if grad_norms["radial"] else 0.0
    result["grad_angular"] = float(np.mean(grad_norms["angular"])) if grad_norms["angular"] else 0.0
    result["grad_backbone"] = float(np.mean(grad_norms["backbone"])) if grad_norms["backbone"] else 0.0
    return result


def eval_radial_health(model, proteins: List[Dict], gate_patched: bool, device: str) -> Dict:
    """Quick eval: radial_std and projection_applied_fraction."""
    model.eval()
    radial_stds, proj_fracs, cone_ranges = [], [], []

    with torch.no_grad():
        for prot in proteins:
            data = prot["data"].to(device)
            if gate_patched:
                _orig = model.gate.forward
                model.gate.forward = lambda xt, cl, cd: _orig(xt, cl, cd, data=data)

            out = model(data)

            if gate_patched:
                model.gate.forward = _orig

            rd = out["radial_features"].squeeze().cpu()
            radial_stds.append(float(rd.std()))

            at = out.get("audit_trail", {})
            pf = at.get("projection_applied_fraction", None)
            if pf is not None:
                proj_fracs.append(float(pf))

            cd = out["cone_depth"].squeeze().cpu()
            cone_ranges.append(float(cd.max() - cd.min()))

    return {
        "radial_std_mean": float(np.mean(radial_stds)),
        "proj_frac_mean": float(np.mean(proj_fracs)) if proj_fracs else -1.0,
        "cone_range_mean": float(np.mean(cone_ranges)),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain GNNv6 with boundary saturation fix")
    parser.add_argument("--checkpoint", default="checkpoints_v6_topo/v6_best.pt")
    parser.add_argument("--output_dir", default="checkpoints_v6_retrained")
    parser.add_argument("--pdb_dir", default="/tmp/dtie_pdb_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base LR (lower than fresh training since we warm-start)")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Load training data ────────────────────────────────────────────────────
    # Import data utilities directly to avoid train_v4.py's broken Gnnv4 import
    from experiments.training.v6._data import TRAINING_TARGETS, load_protein_graph

    logger.info("Loading training proteins...")
    proteins = []
    for pdb_id, info in TRAINING_TARGETS.items():
        prot = load_protein_graph(pdb_id, info["chain"], pdb_dir)
        if prot is not None:
            proteins.append(prot)
            logger.info("  %s (%s %s): %d residues", pdb_id, info["gene"],
                        info["desc"], prot["n_residues"])

    if not proteins:
        logger.error("No proteins loaded — check network access to RCSB.")
        sys.exit(1)

    logger.info("Loaded %d proteins", len(proteins))

    # ── Load checkpoint with patches ──────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    logger.info("Warm-starting from %s", checkpoint_path)
    model, gate_patched = load_checkpoint(checkpoint_path, device)
    model.to(device)

    # Sanity check — should see projection_applied_fraction < 1.0 now
    health = eval_radial_health(model, proteins[:2], gate_patched, device)
    logger.info("PRE-TRAIN health | radial_std=%.4f  proj_frac=%.4f  cone_range=%.4f",
                health["radial_std_mean"], health["proj_frac_mean"], health["cone_range_mean"])

    if health["proj_frac_mean"] > 0.95:
        logger.warning("proj_frac still near 1.0 — tangent clamp may not be active")

    # ── 3-stage curriculum (compressed epochs — warm start) ──────────────────
    # Stage 1: Re-calibrate RadialHead to use the full ball (angular frozen)
    # Stage 2: Re-align angular/gate with corrected radial (radial frozen)
    # Stage 3: Joint fine-tune everything
    stages = [
        {
            "name": "Stage 1: Radial recalibration",
            # More epochs + higher LR: radial head re-initialised from scratch,
            # needs enough gradient steps to learn SASA ordering from baseline.
            "epochs": 50,
            "lr": args.lr * 2.0,
            "freeze_radial": False,
            "freeze_angular": True,
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.01,
                "cone_coeff": 0.50,
                "neighborhood_coeff": 0.05,
                "angular_coeff": 0.0,
                "domain_sep_2d_coeff": 0.0,
                "domain_sep_3d_coeff": 0.0,
            },
        },
        {
            "name": "Stage 2: Angular re-alignment",
            "epochs": 40,
            "lr": args.lr,
            "freeze_radial": True,
            "freeze_angular": False,
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.01,
                "cone_coeff": 0.0,
                "neighborhood_coeff": 0.30,
                "angular_coeff": 0.30,
                "domain_sep_2d_coeff": 0.40,
                "domain_sep_3d_coeff": 0.40,
            },
        },
        {
            "name": "Stage 3: Joint fine-tune",
            "epochs": 30,
            "lr": args.lr * 0.3,
            "freeze_radial": False,
            "freeze_angular": False,
            "coeffs": {
                "evidential_coeff": 0.005,
                "balance_coeff": 0.005,
                "cone_coeff": 0.15,
                "neighborhood_coeff": 0.25,
                "angular_coeff": 0.20,
                "domain_sep_2d_coeff": 0.30,
                "domain_sep_3d_coeff": 0.30,
            },
        },
    ]

    from science.dtie.v5.gnn.model import build_optimizer

    metrics_log = []
    global_epoch = 0
    best_score = -math.inf

    for stage in stages:
        logger.info("\n%s", "=" * 70)
        logger.info("%s (%d epochs, lr=%.2e)", stage["name"], stage["epochs"], stage["lr"])
        logger.info("  freeze_radial=%s  freeze_angular=%s",
                    stage["freeze_radial"], stage["freeze_angular"])

        optimizer = build_optimizer(model, lr=stage["lr"])

        for epoch in range(stage["epochs"]):
            global_epoch += 1
            t0 = time.time()

            losses = train_epoch(
                model=model,
                optimizer=optimizer,
                proteins=proteins,
                loss_coeffs=stage["coeffs"],
                gate_patched=gate_patched,
                freeze_radial=stage["freeze_radial"],
                freeze_angular=stage["freeze_angular"],
                device=device,
            )

            elapsed = time.time() - t0
            c_val = float(model.curvature.item())
            import torch.nn.functional as F
            rs = float(F.softplus(model.radial_head.radial_scale).item())

            # Eval every 5 epochs
            health = {}
            if epoch % 5 == 0 or epoch == stage["epochs"] - 1:
                health = eval_radial_health(model, proteins, gate_patched, device)

            logger.info(
                "  Ep %3d | loss=%.4f (cone=%.4f ang=%.4f nbr=%.4f dom2d=%.4f) | "
                "c=%.4f rs=%.4f | proj_frac=%.3f cone_range=%.4f | "
                "∇rad=%.4f ∇ang=%.4f | %.1fs",
                global_epoch,
                losses["total"], losses["cone_consistency"],
                losses["angular_diversity"], losses["neighborhood_consistency"],
                losses["domain_separation_2d"],
                c_val, rs,
                health.get("proj_frac_mean", -1),
                health.get("cone_range_mean", -1),
                losses["grad_radial"], losses["grad_angular"],
                elapsed,
            )

            # Score: cone_range + (1 - proj_frac) — higher is better
            pf = health.get("proj_frac_mean", 1.0)
            cr = health.get("cone_range_mean", 0.0)
            score = cr + (1.0 - pf)

            if score > best_score:
                best_score = score
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "curvature": c_val,
                    "radial_scale": rs,
                    "stage": stage["name"],
                    "global_epoch": global_epoch,
                    "score": score,
                    "proj_frac": pf,
                    "cone_range": cr,
                    "architecture": {
                        "version": "v6_retrained",
                        "node_dim": 4, "hidden": 128,
                        "num_layers": 6, "num_experts": 4,
                        "projection_dim": 64,
                        "hyp_proj_dim_2d": 2, "hyp_proj_dim_3d": 3,
                        "tangent_clamp": True,
                    },
                }, output_dir / "v6_best.pt")
                logger.info("    ★ New best (score=%.4f, proj_frac=%.3f, cone_range=%.4f)",
                            score, pf, cr)

            metrics_log.append({
                "global_epoch": global_epoch,
                "stage": stage["name"],
                "losses": losses,
                "curvature": c_val,
                "radial_scale": rs,
                "health": health,
                "score": score,
                "elapsed": elapsed,
            })
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2, default=str)

        # Stage checkpoint
        slug = stage["name"].split(":")[0].strip().lower().replace(" ", "_")
        torch.save({
            "model_state_dict": model.state_dict(),
            "stage": stage["name"],
            "global_epoch": global_epoch,
        }, output_dir / f"{slug}.pt")
        logger.info("  Saved %s.pt", slug)

    logger.info("\n%s", "=" * 70)
    logger.info("RETRAIN COMPLETE | best_score=%.4f", best_score)
    logger.info("  Checkpoint: %s/v6_best.pt", output_dir)
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()
