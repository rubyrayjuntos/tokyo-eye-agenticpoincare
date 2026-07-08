"""Shared v6 training loop utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from science.dtie.common.residue_features import residue_sasa_from_data
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


def prepare_training_batch(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    structural_disc_frozen: bool = False,
) -> Any:
    """Attach v6 features and optional structural disc SSOT before a training forward."""
    # Clone so structural attach does not mutate cached prot["data"] across checkpoints.
    data = attach_v6_features(prot["data"].clone().to(device))
    if structural_disc_frozen:
        from science.dtie.common.structural_disc_compose import attach_structural_disc_for_forward

        curvature_c = float(model.curvature.detach().cpu().item())
        data = attach_structural_disc_for_forward(data, prot, curvature_c)
    return data


def set_slim_moe_structural_ssot_freeze(model: nn.Module) -> None:
    """Train MoE routing + uncertainty on frozen structural disc; geometry heads read-only."""
    geometry_prefixes = (
        "radial_head.",
        "angular_head.",
        "hyp_proj_head_2d.",
        "hyp_proj_head_3d.",
    )
    for name, param in model.named_parameters():
        if name.startswith(geometry_prefixes) or name == "expert_depth_bias":
            param.requires_grad = False
        else:
            param.requires_grad = True
    fusion = getattr(model, "radial_angular_fusion", None)
    if fusion is not None:
        for p in fusion.parameters():
            p.requires_grad = False
    if hasattr(model, "gate") and hasattr(model.gate, "detach_gate_input"):
        model.gate.detach_gate_input = False
    set_uncertainty_from_backbone(model, enabled=True)


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


def set_gate_only_freeze(model: nn.Module, *, train_experts: bool = True) -> None:
    """Train MoE gate (+ optional expert MLPs); freeze backbone and geometry heads."""
    for name, param in model.named_parameters():
        if name.startswith("gate."):
            param.requires_grad = True
        elif train_experts and name.startswith("experts."):
            param.requires_grad = True
        else:
            param.requires_grad = False
    if getattr(model, "expert_depth_bias", None) is not None:
        model.expert_depth_bias.requires_grad = False
    if hasattr(model, "gate") and hasattr(model.gate, "detach_gate_input"):
        model.gate.detach_gate_input = False


def set_topology_gate_disc_recovery_freeze(model: nn.Module) -> None:
    """Gate + 2D disc projection only; freeze expert_depth_bias and all routing geometry."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gate.") or name.startswith("hyp_proj_head_2d.")
    if getattr(model, "expert_depth_bias", None) is not None:
        model.expert_depth_bias.requires_grad = False
    if hasattr(model, "gate") and hasattr(model.gate, "detach_gate_input"):
        model.gate.detach_gate_input = False


def set_topology_crescent_recovery_freeze(model: nn.Module) -> None:
    """Angular + fusion + disc wedge; freeze radial depth and expert_depth_bias."""
    set_fusion_path_recovery_freeze(model)
    if getattr(model, "radial_head", None) is not None:
        for p in model.radial_head.parameters():
            p.requires_grad = False
    if getattr(model, "expert_depth_bias", None) is not None:
        model.expert_depth_bias.requires_grad = False


def set_uncertainty_from_backbone(model: nn.Module, *, enabled: bool) -> None:
    """Switch uncertainty head input between backbone and post-routing tangent."""
    if hasattr(model, "uncertainty_from_backbone"):
        model.uncertainty_from_backbone = enabled


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
    gate_only_train: bool = False,
    topology_gate_disc_recovery_train: bool = False,
    topology_crescent_recovery_train: bool = False,
    slim_moe_structural_ssot_train: bool = False,
    structural_disc_frozen: bool = False,
    topology_depth: bool = False,
) -> dict[str, float]:
    """Run one training epoch over all proteins (one protein per optimizer step)."""
    import logging

    logger = logging.getLogger(__name__)
    if epistemic_uncertainty_only_train:
        set_p4_uncertainty_only_freeze(model)
        set_uncertainty_from_backbone(model, enabled=False)
    elif topology_gate_disc_recovery_train:
        set_topology_gate_disc_recovery_freeze(model)
        set_uncertainty_from_backbone(model, enabled=True)
    elif topology_crescent_recovery_train:
        set_topology_crescent_recovery_freeze(model)
        set_uncertainty_from_backbone(model, enabled=True)
    elif slim_moe_structural_ssot_train:
        set_slim_moe_structural_ssot_freeze(model)
    elif gate_only_train:
        set_gate_only_freeze(model)
        set_uncertainty_from_backbone(model, enabled=True)
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
            "routing_load_floor",
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
            "pocket_bce",
            "interface_bce",
            "leak_bce",
        ]
    }
    expert_load_acc: list[torch.Tensor] = []
    effective_experts_per_prot: list[float] = []
    min_routing_fracs: list[float] = []
    grad_norms: dict[str, list[float]] = {
        "grad_radial": [],
        "grad_angular": [],
        "grad_backbone": [],
    }
    fold_totals: dict[str, list[float]] = {}

    for prot in proteins:
        pdb_id = prot.get("pdb_id", "?")
        n_res = prot.get("n_residues", 0)
        try:
            data = prepare_training_batch(
                model,
                prot,
                device,
                structural_disc_frozen=structural_disc_frozen,
            )
            target_rho = prot["target_rho"].to(device)
            target_dehydron = prot.get("target_dehydron")
            if target_dehydron is not None:
                target_dehydron = target_dehydron.to(device)
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

                target_pocket = prot.get("target_pocket")
                if target_pocket is not None:
                    target_pocket = target_pocket.to(device)
                target_interface = prot.get("target_interface")
                if target_interface is not None:
                    target_interface = target_interface.to(device)
                pocket_label_mask = prot.get("pocket_label_mask")
                if pocket_label_mask is not None:
                    pocket_label_mask = pocket_label_mask.to(device)
                interface_label_mask = prot.get("interface_label_mask")
                if interface_label_mask is not None:
                    interface_label_mask = interface_label_mask.to(device)
                target_leak = prot.get("target_leak")
                if target_leak is not None:
                    target_leak = target_leak.to(device)
                leak_label_mask = prot.get("leak_label_mask")
                if leak_label_mask is not None:
                    leak_label_mask = leak_label_mask.to(device)

            def _forward_losses() -> dict[str, Any]:
                output = model(data)
                sasa = None
                if not topology_depth:
                    sasa = residue_sasa_from_data(data).squeeze(-1)
                losses = gosp_loss_v6(
                    output=output,
                    target_rho=target_rho.squeeze(-1) if target_rho.dim() > 1 else target_rho,
                    target_dehydron=(
                        target_dehydron.squeeze(-1)
                        if target_dehydron is not None and target_dehydron.dim() > 1
                        else target_dehydron
                    ),
                    ca_coords=ca_coords,
                    domain_labels=domain_labels,
                    sasa=sasa,
                    b_factor_ca=b_factor_ca,
                    b_factor_present=b_factor_present,
                    target_pocket=target_pocket,
                    target_interface=target_interface,
                    target_leak=target_leak,
                    pocket_label_mask=pocket_label_mask,
                    interface_label_mask=interface_label_mask,
                    leak_label_mask=leak_label_mask,
                    topology_depth=topology_depth,
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
                    and not slim_moe_structural_ssot_train
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

            from science.training.grad_probe import append_subsystem_grad_norms

            if cuda_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    losses = _forward_losses()
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                append_subsystem_grad_norms(model, grad_norms)
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = _forward_losses()
                losses["total"].backward()
                append_subsystem_grad_norms(model, grad_norms)
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

    from science.training.grad_probe import finalize_subsystem_grad_norms

    result = {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}
    for fold_id, vals in fold_totals.items():
        key = f"per_fold_loss.{fold_id_to_mlflow_key(fold_id)}"
        result[key] = float(np.mean(vals)) if vals else 0.0
    result.update(finalize_subsystem_grad_norms(grad_norms))
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
    disc_r_values: list[list[float]],
    r_dt_pairs: list[tuple[list[float], list[float]]],
    tau_sums: list[float],
    rho_sums: list[float],
    helix_counts: list[int],
    sheet_counts: list[int],
    coil_counts: list[int],
    bio_counts: list[int],
) -> None:
    """Per-expert geometry + topology from dominant routing assignment."""
    weights = out.get("expert_weights")
    if weights is None:
        return
    cd = out["cone_depth"].squeeze().detach().cpu().numpy()
    x = data.x.detach().cpu().numpy()
    rho = x[:, 0]
    tau = x[:, 1]
    ss = x[:, 2]
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
        disc_r_values[e].extend(disc_r[mask].tolist())
        r_dt_pairs[e][0].extend(cd[mask].tolist())
        r_dt_pairs[e][1].extend(tau[mask].tolist())
        tau_sums[e] += float(tau[mask].sum())
        rho_sums[e] += float(rho[mask].sum())
        ss_m = ss[mask]
        helix_counts[e] += int((ss_m < 0.25).sum())
        sheet_counts[e] += int(((ss_m >= 0.25) & (ss_m < 0.75)).sum())
        coil_counts[e] += int((ss_m >= 0.75).sum())
        bio_counts[e] += n


def measure_geometry_health(
    model: nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    max_proteins: int = 32,
    topology_depth: bool = False,
    structural_disc_frozen: bool = False,
    edge_telemetry: bool = True,
) -> dict[str, float]:
    """Geometry health + shell probe correlations on a capped protein subset."""
    from science.dtie.common.residue_features import TAU
    from science.training.uncertainty_diagnostics import (
        NODE_ALE_INFORMATIVE_FLOOR,
        NODE_ALE_STD_FLOOR,
        NODE_EPI_STD_FLOOR,
    )

    model.train(False)
    radial_stds, proj_fracs, cone_ranges = [], [], []
    cone_depth_stds, cone_depth_means = [], []
    disc_r_means, disc_r_stds = [], []
    probe_depth_sasa, probe_epi_sasa, probe_proj_depth, probe_disc_sasa = [], [], [], []
    probe_depth_tau, probe_depth_rho = [], []
    probe_epi_ale: list[float] = []
    all_epi_vals: list[np.ndarray] = []
    all_ale_vals: list[np.ndarray] = []
    all_rho_vals: list[np.ndarray] = []
    all_nu_vals: list[np.ndarray] = []
    edge_telemetry_records: list[Any] = []
    disc_sigma_ratios, disc_eff_ranks, disc_thickness = [], [], []
    disc_thickness_pre, disc_thickness_post = [], []
    x_hyp_thickness = []
    num_experts = len(getattr(model, "experts", [])) or 4
    depth_sums = [0.0] * num_experts
    depth_counts = [0] * num_experts
    disc_sums = [0.0] * num_experts
    disc_counts = [0] * num_experts
    disc_r_values: list[list[float]] = [[] for _ in range(num_experts)]
    r_dt_pairs: list[tuple[list[float], list[float]]] = [( [], []) for _ in range(num_experts)]
    tau_sums = [0.0] * num_experts
    rho_sums = [0.0] * num_experts
    helix_counts = [0] * num_experts
    sheet_counts = [0] * num_experts
    coil_counts = [0] * num_experts
    bio_counts = [0] * num_experts
    sample = proteins if len(proteins) <= max_proteins else proteins[:max_proteins]

    with torch.no_grad():
        for prot in sample:
            data = prepare_training_batch(
                model,
                prot,
                device,
                structural_disc_frozen=structural_disc_frozen,
            )
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

            tau = data.x[:, 1].cpu().numpy()
            rho = data.x[:, 0].cpu().numpy()
            if not topology_depth:
                sasa = residue_sasa_from_data(data).squeeze(-1).cpu().numpy()
            epi = out["uncertainty"]["epistemic"].squeeze().cpu().numpy()
            ale = out["uncertainty"]["aleatoric"].squeeze().cpu().numpy()
            all_epi_vals.append(np.asarray(epi, dtype=float).reshape(-1))
            all_ale_vals.append(np.asarray(ale, dtype=float).reshape(-1))
            all_rho_vals.append(np.asarray(rho, dtype=float).reshape(-1))
            evidence = out.get("evidence") or {}
            nu_t = evidence.get("nu")
            if nu_t is not None:
                all_nu_vals.append(nu_t.detach().cpu().numpy().reshape(-1))
            probe_epi_ale.append(_pearson_np(epi, ale))
            hyp = out["hyp_projections_2d"].cpu().numpy()
            disc_r = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)
            disc_r_means.append(float(disc_r.mean()))
            disc_r_stds.append(float(disc_r.std()))
            if not topology_depth:
                probe_depth_sasa.append(_pearson_np(cd, sasa))
                probe_epi_sasa.append(_pearson_np(epi, sasa))
                probe_disc_sasa.append(_pearson_np(disc_r, sasa))
            probe_depth_tau.append(_pearson_np(cd, tau))
            probe_depth_rho.append(_pearson_np(cd, rho))
            probe_proj_depth.append(_pearson_np(disc_r, cd))
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
                disc_r_values=disc_r_values,
                r_dt_pairs=r_dt_pairs,
                tau_sums=tau_sums,
                rho_sums=rho_sums,
                helix_counts=helix_counts,
                sheet_counts=sheet_counts,
                coil_counts=coil_counts,
                bio_counts=bio_counts,
            )
            if edge_telemetry:
                from science.training.edge_telemetry import collect_edge_telemetry

                ca = prot.get("ca_coords")
                ca_np = ca.detach().cpu().numpy() if ca is not None else None
                edge_telemetry_records.append(
                    collect_edge_telemetry(
                        model,
                        data,
                        out,
                        structure_id=str(prot.get("pdb_id", "?")),
                        chain=str(prot.get("chain", "A")),
                        ca_coords=ca_np,
                        include_per_edge=False,
                    )
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
        "probe_r_depth_tau": _nanmean(probe_depth_tau),
        "probe_r_depth_rho": _nanmean(probe_depth_rho),
        "probe_r_proj_depth": _nanmean(probe_proj_depth),
        "probe_r_epi_ale": _nanmean(probe_epi_ale),
        "epistemic_std_mean": float(np.std(np.concatenate(all_epi_vals))) if all_epi_vals else 0.0,
        "aleatoric_std_mean": float(np.std(np.concatenate(all_ale_vals))) if all_ale_vals else 0.0,
        "disc_sigma2_sigma1_mean": _nanmean(disc_sigma_ratios),
        "disc_effective_rank_mean": _nanmean(disc_eff_ranks),
        "disc_line_thickness_rms_mean": _nanmean(disc_thickness),
        "disc_line_thickness_pre_mean": _nanmean(disc_thickness_pre),
        "disc_line_thickness_post_mean": _nanmean(disc_thickness_post),
        "x_hyp_line_thickness_mean": _nanmean(x_hyp_thickness),
    }
    if not topology_depth:
        result["probe_r_depth_sasa"] = _nanmean(probe_depth_sasa)
        result["probe_r_epi_sasa"] = _nanmean(probe_epi_sasa)
        result["probe_r_disc_sasa"] = _nanmean(probe_disc_sasa)
    for e in range(num_experts):
        if depth_counts[e] > 0:
            result[f"expert_{e}_depth_mean"] = depth_sums[e] / depth_counts[e]
        if disc_counts[e] > 0:
            result[f"expert_{e}_disc_r_mean"] = disc_sums[e] / disc_counts[e]
        if len(disc_r_values[e]) >= 3:
            result[f"expert_{e}_disc_r_std"] = float(np.std(np.asarray(disc_r_values[e], dtype=float)))
        if len(r_dt_pairs[e][0]) >= 3:
            result[f"expert_{e}_r_depth_tau"] = _pearson_np(
                np.array(r_dt_pairs[e][0]), np.array(r_dt_pairs[e][1])
            )
        if bio_counts[e] > 0:
            n_bio = bio_counts[e]
            result[f"expert_{e}_tau_mean"] = tau_sums[e] / n_bio
            result[f"expert_{e}_rho_mean"] = rho_sums[e] / n_bio
            result[f"expert_{e}_ss_helix_frac"] = helix_counts[e] / n_bio
            result[f"expert_{e}_ss_sheet_frac"] = sheet_counts[e] / n_bio
            result[f"expert_{e}_ss_coil_frac"] = coil_counts[e] / n_bio

    if all_epi_vals and all_ale_vals:
        epi_cat = np.concatenate(all_epi_vals)
        ale_cat = np.concatenate(all_ale_vals)
        epi_std = float(np.std(epi_cat))
        ale_std = float(np.std(ale_cat))
        result["uncertainty_probe_alive_epi"] = (
            1.0 if epi_std >= NODE_EPI_STD_FLOOR else 0.0
        )
        result["uncertainty_probe_alive_ale"] = (
            1.0 if ale_std >= NODE_ALE_STD_FLOOR else 0.0
        )
        result["uncertainty_informative_ale"] = (
            1.0 if ale_std >= NODE_ALE_INFORMATIVE_FLOOR else 0.0
        )
        from science.training.evidential_validation import EPISTEMIC_CORPUS_STD_FLOOR

        result["uncertainty_epistemic_non_degenerate"] = (
            1.0 if epi_std >= EPISTEMIC_CORPUS_STD_FLOOR else 0.0
        )
        if all_nu_vals:
            nu_cat = np.concatenate(all_nu_vals)
            nu_mean = float(np.mean(nu_cat))
            if nu_mean > 1e-12:
                result["evidence_nu_cv_mean"] = float(np.std(nu_cat) / nu_mean)
        if all_rho_vals:
            rho_cat = np.concatenate(all_rho_vals)
            tau_mask = np.abs(rho_cat - TAU) <= 1.0
            if tau_mask.any() and (~tau_mask).any():
                ale_tau = float(np.mean(ale_cat[tau_mask]))
                ale_non = float(np.mean(ale_cat[~tau_mask]))
                lift = ale_tau - ale_non
                result["node_aleatoric_tau_lift"] = lift
                result["uncertainty_tau_ale_elevated"] = 1.0 if lift > 0.0 else 0.0

    if edge_telemetry and edge_telemetry_records:
        from science.training.edge_telemetry import (
            aggregate_corpus_records,
            corpus_aggregate_to_health,
        )

        result.update(
            corpus_aggregate_to_health(aggregate_corpus_records(edge_telemetry_records))
        )

    return result
