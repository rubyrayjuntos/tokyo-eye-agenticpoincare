"""Shared v6 training loop utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from science.dtie.v6.loss import gosp_loss_v6


def attach_v6_features(data: torch.Tensor | Any) -> Any:
    """Attach degree, ss_onehot, rho required by GOSPConeMapperV6."""
    from torch_geometric.data import Data
    from torch_geometric.utils import degree as pyg_degree

    if not isinstance(data, Data):
        return data

    device = data.x.device
    num_nodes = data.x.size(0)
    if not hasattr(data, "degree") or data.degree is None:
        data.degree = pyg_degree(
            data.edge_index[0],
            num_nodes=num_nodes,
            dtype=data.x.dtype,
        ).to(device)

    if not hasattr(data, "ss_onehot") or data.ss_onehot is None:
        ss_onehot = torch.zeros(num_nodes, 3, dtype=data.x.dtype, device=device)
        ss_type = data.x[:, 2]
        for i in range(num_nodes):
            idx = int(round(float(ss_type[i].item()) * 2))
            if 0 <= idx <= 2:
                ss_onehot[i, idx] = 1.0
            else:
                ss_onehot[i, 2] = 1.0
        data.ss_onehot = ss_onehot
    elif data.ss_onehot.device != device:
        data.ss_onehot = data.ss_onehot.to(device)

    if not hasattr(data, "rho") or data.rho is None:
        data.rho = data.x[:, 0].clone()
    elif data.rho.device != device:
        data.rho = data.rho.to(device)

    if hasattr(data, "degree") and data.degree.device != device:
        data.degree = data.degree.to(device)

    return data


def set_projection_recovery_freeze(model: nn.Module) -> None:
    """Train disc projection head + gate disc readout (same modules as path alignment)."""
    set_path_alignment_freeze(model)


def set_rec_ablation_freeze(model: nn.Module) -> None:
    """Train only residual radial×angular fusion MLP and 2D disc projection head."""
    for p in model.parameters():
        p.requires_grad = False
    fusion = getattr(model, "radial_angular_fusion", None)
    if fusion is not None:
        for p in fusion.parameters():
            p.requires_grad = True
    for p in model.hyp_proj_head_2d.parameters():
        p.requires_grad = True


def set_lift_path_recovery_freeze(model: nn.Module) -> None:
    """Train radial + angular + fusion + disc projection + gate disc readout."""
    set_fusion_path_recovery_freeze(model)
    for p in model.radial_head.parameters():
        p.requires_grad = True


def set_fusion_path_recovery_freeze(model: nn.Module) -> None:
    """Train angular + fusion residual + disc projection + gate disc readout."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.angular_head.parameters():
        p.requires_grad = True
    fusion = getattr(model, "radial_angular_fusion", None)
    if fusion is not None:
        for p in fusion.parameters():
            p.requires_grad = True
    for p in model.hyp_proj_head_2d.parameters():
        p.requires_grad = True
    gate = getattr(model, "gate", None)
    if gate is not None:
        if hasattr(gate, "gate_disc_proj"):
            for p in gate.gate_disc_proj.parameters():
                p.requires_grad = True
        topo = getattr(gate, "topo_encoder", None)
        if isinstance(topo, nn.Sequential) and len(topo) > 0:
            for p in topo[-1].parameters():
                p.requires_grad = True
        if hasattr(gate, "detach_gate_input"):
            gate.detach_gate_input = False


def set_path_alignment_freeze(model: nn.Module) -> None:
    """Train only hyp_proj_head_2d, gate.gate_disc_proj, and gate topo_encoder last layer."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.hyp_proj_head_2d.parameters():
        p.requires_grad = True
    gate = getattr(model, "gate", None)
    if gate is not None:
        if hasattr(gate, "gate_disc_proj"):
            for p in gate.gate_disc_proj.parameters():
                p.requires_grad = True
        topo = getattr(gate, "topo_encoder", None)
        if isinstance(topo, nn.Sequential) and len(topo) > 0:
            for p in topo[-1].parameters():
                p.requires_grad = True
        if hasattr(gate, "detach_gate_input"):
            gate.detach_gate_input = False


def set_model_freeze(
    model: nn.Module,
    *,
    freeze_radial: bool,
    freeze_angular: bool,
    freeze_backbone: bool,
    freeze_gate: bool,
) -> None:
    """Toggle requires_grad per v6 module group."""
    for p in model.radial_head.parameters():
        p.requires_grad = not freeze_radial
    for p in model.angular_head.parameters():
        p.requires_grad = not freeze_angular
    for p in model.convs.parameters():
        p.requires_grad = not freeze_backbone
    for p in model.norms.parameters():
        p.requires_grad = not freeze_backbone
    for p in model.node_emb.parameters():
        p.requires_grad = not freeze_backbone
    for p in model.gate.parameters():
        p.requires_grad = not freeze_gate
    if hasattr(model.gate, "detach_gate_input"):
        model.gate.detach_gate_input = freeze_gate
    for p in model.experts.parameters():
        p.requires_grad = not freeze_gate or not freeze_backbone


def set_expert_dropout(model: nn.Module, p: float) -> None:
    if hasattr(model, "gate") and hasattr(model.gate, "expert_dropout_p"):
        model.gate.expert_dropout_p = p


def set_p4_uncertainty_only_freeze(model: nn.Module) -> None:
    """Phase 4 stage 1: train uncertainty_head only; freeze all other parameters."""
    for name, param in model.named_parameters():
        param.requires_grad = "uncertainty_head" in name


def build_p4_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.AdamW:
    """AdamW on uncertainty_head only — no RiemannianAdam, no AMP (Phase 4 stage 1)."""
    params = [p for p in model.uncertainty_head.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("uncertainty_head has no trainable parameters for Phase 4")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def warm_start_v5_backbone(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> int:
    """Load v5-compatible weights into v6 model (backbone + heads). Returns key count."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = state.get("model_state_dict", state)
    model_state = model.state_dict()
    loaded = 0
    skip_prefixes = ("experts.", "gate.")
    for key, tensor in weights.items():
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        if key in model_state and model_state[key].shape == tensor.shape:
            model_state[key] = tensor
            loaded += 1
    model.load_state_dict(model_state)
    return loaded


def init_v6_radial_scale(model: nn.Module, value: float = -2.0) -> None:
    """Conservative radial scale so fresh training avoids immediate boundary saturation."""
    if hasattr(model, "radial_head") and hasattr(model.radial_head, "radial_scale"):
        nn.init.constant_(model.radial_head.radial_scale, value)


def resolve_default_warm_start() -> "Path | None":
    """Find best available v5/v6 checkpoint for warm-start training."""
    from pathlib import Path

    from science.contracts.model_registry import get_checkpoint_catalog, resolve_checkpoint_file

    for cid in ("tokyo_eyes_v6", "tokyo_eyes_v5"):
        try:
            spec = get_checkpoint_catalog()[cid]
        except KeyError:
            continue
        resolved = resolve_checkpoint_file(spec.path)
        if resolved is not None:
            return resolved
    for candidate in (
        Path("/app/checkpoints_v6_retrained/v6_best.pt"),
        Path("/app/checkpoints_v6_topo/v6_best.pt"),
    ):
        if candidate.is_file():
            return candidate
    return None


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    proteins: list[dict[str, Any]],
    loss_coeffs: dict[str, float],
    *,
    freeze_radial: bool = False,
    freeze_angular: bool = False,
    freeze_backbone: bool = False,
    freeze_gate: bool = False,
    path_alignment_train: bool = False,
    rec_ablation_train: bool = False,
    projection_recovery_train: bool = False,
    fusion_path_recovery_train: bool = False,
    lift_path_recovery_train: bool = False,
    device: str = "cpu",
    use_amp: bool = True,
    v2_teacher: Any | None = None,
    v2_teacher_depth_coeff: float = 0.20,
    v2_teacher_epistemic_coeff: float = 0.10,
    epistemic_decoupling_holdouts: frozenset[str] | None = None,
    epistemic_uncertainty_only_train: bool = False,
) -> dict[str, float]:
    """Run one training epoch over all proteins (one protein per optimizer step)."""
    import logging

    logger = logging.getLogger(__name__)
    if epistemic_uncertainty_only_train:
        set_p4_uncertainty_only_freeze(model)
    elif path_alignment_train or projection_recovery_train:
        set_path_alignment_freeze(model)
    elif fusion_path_recovery_train:
        set_fusion_path_recovery_freeze(model)
    elif lift_path_recovery_train:
        set_lift_path_recovery_freeze(model)
    elif rec_ablation_train:
        set_rec_ablation_freeze(model)
    else:
        set_model_freeze(
            model,
            freeze_radial=freeze_radial,
            freeze_angular=freeze_angular,
            freeze_backbone=freeze_backbone,
            freeze_gate=freeze_gate,
        )
    model.train()

    cuda_amp = (
        not epistemic_uncertainty_only_train
        and device != "cpu"
        and use_amp
        and torch.cuda.is_available()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cuda_amp)

    epoch_losses: dict[str, list[float]] = {
        k: []
        for k in [
            "total",
            "evidential",
            "capacity_loss",
            "cone_consistency",
            "cone_depth_anticollapse",
            "shell_correlation",
            "shell_corr_depth_sasa",
            "shell_corr_epi_sasa",
            "shell_corr_proj_depth",
            "shell_corr_disc_spread",
            "shell_corr_disc_sasa",
            "shell_r_depth_sasa",
            "shell_r_epi_sasa",
            "shell_r_proj_depth",
            "shell_r_disc_sasa",
            "disc_r_std",
            "disc_depth_scale",
            "disc_path_align",
            "disc_thickness_floor",
            "disc_line_thickness_rms",
            "disc_origin_span_floor",
            "x_hyp_thickness_floor",
            "x_hyp_line_thickness_rms",
            "neighborhood_consistency",
            "angular_diversity",
            "domain_separation_2d",
            "domain_separation_3d",
            "routing_entropy",
            "v2_teacher_total",
            "v2_teacher_depth",
            "v2_teacher_epistemic",
            "epistemic_decoupling",
            "epistemic_bf_align",
            "epistemic_sasa_pen",
            "epistemic_anticollapse",
            "r_epi_bf_resid",
            "partial_epi_sasa_given_depth",
        ]
    }
    expert_load_acc: list[torch.Tensor] = []
    effective_experts_per_prot: list[float] = []
    min_routing_fracs: list[float] = []
    grad_norms = {"radial": [], "angular": [], "backbone": []}
    fold_totals: dict[str, list[float]] = {}

    for prot in proteins:
        pdb_id = prot.get("pdb_id", "?")
        n_res = prot.get("n_residues", 0)
        try:
            data = attach_v6_features(prot["data"].to(device))
            target_rho = prot["target_rho"].to(device)
            ca_coords = prot["ca_coords"].to(device)
            domain_labels = prot.get("domain_labels")
            if domain_labels is not None:
                domain_labels = domain_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            holdouts = epistemic_decoupling_holdouts or frozenset()
            b_factor_ca = None
            b_factor_present = None
            if pdb_id.upper() not in holdouts:
                raw_bf = prot.get("b_factor_ca")
                if raw_bf is not None:
                    b_factor_ca = raw_bf.to(device)
                    raw_mask = prot.get("b_factor_present")
                    if raw_mask is not None:
                        b_factor_present = raw_mask.to(device)

            def _forward_losses() -> dict[str, Any]:
                output = model(data)
                sasa = data.x[:, 3]
                losses = gosp_loss_v6(
                    output=output,
                    target_rho=target_rho.squeeze(-1) if target_rho.dim() > 1 else target_rho,
                    ca_coords=ca_coords,
                    domain_labels=domain_labels,
                    sasa=sasa,
                    b_factor_ca=b_factor_ca,
                    b_factor_present=b_factor_present,
                    **loss_coeffs,
                )
                if (
                    v2_teacher is not None
                    and not epistemic_uncertainty_only_train
                    and not path_alignment_train
                    and not rec_ablation_train
                    and not projection_recovery_train
                    and not fusion_path_recovery_train
                    and not lift_path_recovery_train
                ):
                    teacher_targets = v2_teacher.targets_for(pdb_id)
                    if teacher_targets is not None:
                        from science.dtie.v6.loss import v2_teacher_distill_loss

                        distill = v2_teacher_distill_loss(
                            output,
                            teacher_targets,
                            sasa,
                            depth_coeff=v2_teacher_depth_coeff,
                            epistemic_coeff=v2_teacher_epistemic_coeff,
                        )
                        losses["total"] = losses["total"] + distill["v2_teacher_total"]
                        losses.update(distill)
                return losses

            if cuda_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    losses = _forward_losses()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = _forward_losses()
                losses["total"].backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM on %s (%d residues) — skipping protein", pdb_id, n_res)
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        finally:
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "expert_load" in losses:
            load_t = losses["expert_load"].detach().cpu()
            expert_load_acc.append(load_t)
            from science.training.routing_metrics import effective_experts, min_routing_fraction

            effective_experts_per_prot.append(effective_experts(load_t))
            min_routing_fracs.append(min_routing_fraction(load_t))
        elif "routing_entropy" in losses:
            h = losses["routing_entropy"]
            h_val = float(h.item() if torch.is_tensor(h) else h)
            effective_experts_per_prot.append(float(np.exp(h_val)))

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

        from science.training.mlflow_governance import fold_id_to_mlflow_key

        fold_id = str(prot.get("fold_id") or "unverified")
        total_v = losses.get("total")
        if total_v is not None:
            fold_totals.setdefault(fold_id, []).append(
                float(total_v.item() if torch.is_tensor(total_v) else total_v)
            )

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
    for fold_id, vals in fold_totals.items():
        key = f"per_fold_loss.{fold_id_to_mlflow_key(fold_id)}"
        result[key] = float(np.mean(vals)) if vals else 0.0
    result["grad_radial"] = float(np.mean(grad_norms["radial"])) if grad_norms["radial"] else 0.0
    result["grad_angular"] = float(np.mean(grad_norms["angular"])) if grad_norms["angular"] else 0.0
    result["grad_backbone"] = float(np.mean(grad_norms["backbone"])) if grad_norms["backbone"] else 0.0
    if expert_load_acc:
        mean_load = torch.stack(expert_load_acc).mean(dim=0)
        for i, load in enumerate(mean_load.tolist()):
            result[f"expert_load_{i}"] = float(load)
        result["expert_starvation_count"] = float(sum(1 for x in mean_load.tolist() if x < 0.05))
    if effective_experts_per_prot:
        result["effective_experts"] = float(np.mean(effective_experts_per_prot))
        result["effective_experts_min"] = float(np.min(effective_experts_per_prot))
    if min_routing_fracs:
        result["min_routing_fraction"] = float(np.min(min_routing_fracs))
    return result


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _accumulate_expert_geometry(
    out: dict[str, Any],
    data: Any,
    *,
    num_experts: int,
    depth_sums: list[float],
    depth_counts: list[int],
    disc_sums: list[float],
    disc_counts: list[int],
    r_ds_pairs: list[tuple[list[float], list[float]]],
) -> None:
    """Per-expert cone_depth / disc_r / depth×SASA from dominant routing assignment."""
    weights = out.get("expert_weights")
    if weights is None:
        return
    cd = out["cone_depth"].squeeze().detach().cpu().numpy()
    sasa = data.x[:, 3].detach().cpu().numpy()
    hyp = out["hyp_projections_2d"].detach().cpu().numpy()
    disc_r = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)
    assign = weights.argmax(dim=1).detach().cpu().numpy()
    for e in range(num_experts):
        mask = assign == e
        n = int(mask.sum())
        if n < 3:
            continue
        depth_sums[e] += float(cd[mask].sum())
        depth_counts[e] += n
        disc_sums[e] += float(disc_r[mask].sum())
        disc_counts[e] += n
        r_ds_pairs[e][0].extend(cd[mask].tolist())
        r_ds_pairs[e][1].extend(sasa[mask].tolist())


def measure_geometry_health(
    model: nn.Module, proteins: list[dict[str, Any]], device: str, *, max_proteins: int = 32
) -> dict[str, float]:
    """Geometry health + shell probe correlations on a capped protein subset."""
    model.train(False)
    radial_stds, proj_fracs, cone_ranges = [], [], []
    cone_depth_stds, cone_depth_means = [], []
    disc_r_means, disc_r_stds = [], []
    probe_depth_sasa, probe_epi_sasa, probe_proj_depth, probe_disc_sasa = [], [], [], []
    disc_sigma_ratios, disc_eff_ranks, disc_thickness = [], [], []
    disc_thickness_pre, disc_thickness_post = [], []
    x_hyp_thickness = []
    num_experts = len(getattr(model, "experts", [])) or 4
    depth_sums = [0.0] * num_experts
    depth_counts = [0] * num_experts
    disc_sums = [0.0] * num_experts
    disc_counts = [0] * num_experts
    r_ds_pairs: list[tuple[list[float], list[float]]] = [( [], []) for _ in range(num_experts)]
    sample = proteins if len(proteins) <= max_proteins else proteins[:max_proteins]

    with torch.no_grad():
        for prot in sample:
            data = attach_v6_features(prot["data"].to(device))
            out = model(data)
            rd = out["radial_features"].squeeze().cpu().numpy()
            radial_stds.append(float(rd.std()))
            at = out.get("audit_trail", {})
            pf = at.get("projection_applied_fraction")
            if pf is not None:
                proj_fracs.append(float(pf))
            cd = out["cone_depth"].squeeze().cpu().numpy()
            cone_ranges.append(float(cd.max() - cd.min()))
            cone_depth_stds.append(float(cd.std()))
            cone_depth_means.append(float(cd.mean()))

            sasa = data.x[:, 3].cpu().numpy()
            epi = out["uncertainty"]["epistemic"].squeeze().cpu().numpy()
            hyp = out["hyp_projections_2d"].cpu().numpy()
            disc_r = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)
            disc_r_means.append(float(disc_r.mean()))
            disc_r_stds.append(float(disc_r.std()))
            probe_depth_sasa.append(_pearson_np(cd, sasa))
            probe_epi_sasa.append(_pearson_np(epi, sasa))
            probe_proj_depth.append(_pearson_np(disc_r, cd))
            probe_disc_sasa.append(_pearson_np(disc_r, sasa))
            from science.training.disc_occupancy import disc_occupancy_from_numpy, disc_line_thickness_from_tensor

            occ = disc_occupancy_from_numpy(hyp)
            disc_sigma_ratios.append(occ["disc_sigma2_sigma1"])
            disc_eff_ranks.append(occ["disc_effective_rank"])
            disc_thickness.append(occ["disc_line_thickness_rms"])
            x_hyp_thickness.append(
                float(disc_line_thickness_from_tensor(out["x_hyp"].float()).cpu().item())
            )
            hyp_pre = out.get("hyp_projections_2d_pre")
            hyp_post = out.get("hyp_projections_2d_post")
            if hyp_pre is not None and hyp_post is not None:
                occ_pre = disc_occupancy_from_numpy(hyp_pre.detach().cpu().numpy())
                occ_post = disc_occupancy_from_numpy(hyp_post.detach().cpu().numpy())
                disc_thickness_pre.append(occ_pre["disc_line_thickness_rms"])
                disc_thickness_post.append(occ_post["disc_line_thickness_rms"])
            else:
                hyp_routed = out.get("hyp_projections_2d_routed")
                if hyp_routed is not None and not bool(getattr(model, "legacy_disc_projection", False)):
                    occ_pre = disc_occupancy_from_numpy(hyp)
                    occ_post = disc_occupancy_from_numpy(hyp_routed.detach().cpu().numpy())
                    disc_thickness_pre.append(occ_pre["disc_line_thickness_rms"])
                    disc_thickness_post.append(occ_post["disc_line_thickness_rms"])
            _accumulate_expert_geometry(
                out,
                data,
                num_experts=num_experts,
                depth_sums=depth_sums,
                depth_counts=depth_counts,
                disc_sums=disc_sums,
                disc_counts=disc_counts,
                r_ds_pairs=r_ds_pairs,
            )

    def _nanmean(values: list[float]) -> float:
        arr = np.array(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        return float(np.nanmean(arr))

    result = {
        "radial_std_mean": float(np.mean(radial_stds)),
        "proj_frac_mean": float(np.mean(proj_fracs)) if proj_fracs else -1.0,
        "cone_range_mean": float(np.mean(cone_ranges)),
        "cone_depth_std_mean": float(np.mean(cone_depth_stds)),
        "cone_depth_mean_mean": float(np.mean(cone_depth_means)),
        "disc_r_mean": float(np.mean(disc_r_means)),
        "disc_r_std_mean": float(np.mean(disc_r_stds)),
        "probe_r_depth_sasa": _nanmean(probe_depth_sasa),
        "probe_r_epi_sasa": _nanmean(probe_epi_sasa),
        "probe_r_proj_depth": _nanmean(probe_proj_depth),
        "probe_r_disc_sasa": _nanmean(probe_disc_sasa),
        "disc_sigma2_sigma1_mean": _nanmean(disc_sigma_ratios),
        "disc_effective_rank_mean": _nanmean(disc_eff_ranks),
        "disc_line_thickness_rms_mean": _nanmean(disc_thickness),
        "disc_line_thickness_pre_mean": _nanmean(disc_thickness_pre),
        "disc_line_thickness_post_mean": _nanmean(disc_thickness_post),
        "x_hyp_line_thickness_mean": _nanmean(x_hyp_thickness),
    }
    for e in range(num_experts):
        if depth_counts[e] > 0:
            result[f"expert_{e}_depth_mean"] = depth_sums[e] / depth_counts[e]
        if disc_counts[e] > 0:
            result[f"expert_{e}_disc_r_mean"] = disc_sums[e] / disc_counts[e]
        if len(r_ds_pairs[e][0]) >= 3:
            result[f"expert_{e}_r_depth_sasa"] = _pearson_np(
                np.array(r_ds_pairs[e][0]), np.array(r_ds_pairs[e][1])
            )
    return result
