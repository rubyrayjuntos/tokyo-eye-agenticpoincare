"""Inference liveness probes — fail silent no-op feature / MP channels."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Max abs delta below which a channel is treated as dead.
DEFAULT_DEAD_EPS = 1e-6


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _forward_outputs(
    model: nn.Module,
    data: Any,
) -> dict[str, torch.Tensor]:
    out = model(data)
    depth = out["cone_depth"]
    disc = out.get("hyp_projections_2d")
    if disc is None:
        disc = out.get("hyp_projections_2d_pre")
    weights = out.get("expert_weights")
    if weights is None:
        weights = out.get("routing_weights")
    payload: dict[str, torch.Tensor] = {"cone_depth": depth}
    if disc is not None:
        payload["hyp_projections_2d"] = disc
    if weights is not None:
        payload["expert_weights"] = weights
    unc = out.get("uncertainty") or {}
    if isinstance(unc, dict) and "epistemic" in unc:
        payload["epistemic"] = unc["epistemic"]
    return payload


@torch.no_grad()
def probe_barcode_liveness(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    structural_disc_frozen: bool = False,
    eps: float = DEFAULT_DEAD_EPS,
) -> dict[str, float | bool]:
    """Compare forward with barcode vs barcode columns zeroed.

    Returns max abs deltas and ``alive`` if any tracked output moves.
    """
    from experiments.training.v6.train_loop import prepare_training_batch

    from science.dtie.common.residue_features import gnn_input_dim

    data = prepare_training_batch(
        model,
        prot,
        device,
        structural_disc_frozen=structural_disc_frozen,
    )
    # topology_three_vector → base 3; legacy_four_vector → base 4.
    # Barcode scalars (+ missing) begin immediately after the base block.
    base_dim = int(gnn_input_dim())
    if data.x.size(1) <= base_dim:
        return {
            "barcode_cols": 0.0,
            "alive": False,
            "skipped": True,
            "reason": f"node_dim_le_{base_dim}",
            "base_dim": float(base_dim),
        }

    model.eval()
    out_full = _forward_outputs(model, data)
    data_z = data.clone()
    data_z.x = data_z.x.clone()
    data_z.x[:, base_dim:] = 0
    out_zero = _forward_outputs(model, data_z)

    deltas: dict[str, float] = {}
    for key in out_full:
        deltas[f"delta_{key}"] = _max_abs(out_full[key], out_zero[key])
    alive = any(v > eps for v in deltas.values())
    return {
        "barcode_cols": float(data.x.size(1) - base_dim),
        "base_dim": float(base_dim),
        "alive": alive,
        "skipped": False,
        **deltas,
    }


@torch.no_grad()
def probe_dehydron_edge_barcode_liveness(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    structural_disc_frozen: bool = False,
    eps: float = DEFAULT_DEAD_EPS,
) -> dict[str, float | bool]:
    """Compare forward with vs without local dehydron edge barcode columns."""
    from science.dtie.v66.role_edge_graph import DEHYDRON_BARCODE_COL, EDGE_BARCODE_DIM
    from experiments.training.v6.train_loop import prepare_training_batch

    if not getattr(model, "dehydron_edge_barcode", False):
        return {"alive": False, "skipped": True, "reason": "dehydron_edge_barcode_off"}

    data = prepare_training_batch(
        model,
        prot,
        device,
        structural_disc_frozen=structural_disc_frozen,
    )
    end = DEHYDRON_BARCODE_COL + EDGE_BARCODE_DIM
    if data.edge_attr is None or data.edge_attr.size(-1) < end:
        return {
            "alive": False,
            "skipped": True,
            "reason": "edge_attr_too_narrow",
        }

    model.eval()
    out_full = _forward_outputs(model, data)
    data_z = data.clone()
    data_z.edge_attr = data_z.edge_attr.clone()
    data_z.edge_attr[:, DEHYDRON_BARCODE_COL:end] = 0.0
    out_zero = _forward_outputs(model, data_z)

    deltas: dict[str, float] = {}
    for key in out_full:
        deltas[f"delta_{key}"] = _max_abs(out_full[key], out_zero[key])
    alive = any(v > eps for v in deltas.values())
    return {
        "edge_barcode_cols": float(EDGE_BARCODE_DIM),
        "alive": alive,
        "skipped": False,
        **deltas,
    }


@torch.no_grad()
def probe_mp_liveness(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    structural_disc_frozen: bool = False,
    noise_scale: float = 0.05,
    eps: float = DEFAULT_DEAD_EPS,
) -> dict[str, float | bool]:
    """Perturb node_emb input slightly; outputs should move when MP→geometry is live."""
    from experiments.training.v6.train_loop import prepare_training_batch

    data = prepare_training_batch(
        model,
        prot,
        device,
        structural_disc_frozen=structural_disc_frozen,
    )
    model.eval()
    out_a = _forward_outputs(model, data)
    data_b = data.clone()
    data_b.x = data_b.x.clone()
    data_b.x = data_b.x + noise_scale * torch.randn_like(data_b.x)
    out_b = _forward_outputs(model, data_b)

    deltas: dict[str, float] = {}
    for key in out_a:
        deltas[f"delta_{key}"] = _max_abs(out_a[key], out_b[key])
    alive = any(v > eps for v in deltas.values())
    return {
        "alive": alive,
        "structural_disc_frozen": bool(structural_disc_frozen),
        "noise_scale": noise_scale,
        **deltas,
    }


@torch.no_grad()
def probe_chem_edge_liveness(
    model: nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    structural_disc_frozen: bool = False,
    eps: float = DEFAULT_DEAD_EPS,
) -> dict[str, float | bool | str]:
    """Compare forward with vs without chem (disulf/covale) one-hot edges.

    Picks the first corpus protein that carries mapped chem edges after attach.
    Empty chem sets (e.g. 4OBE) are skipped — sparsity is expected, not dead.
    """
    from experiments.training.v6.train_loop import prepare_training_batch
    from science.dtie.v66.chem_edge_graph import (
        NUM_ROLE_RELATIONS_WITH_CHEM,
        ROLE_COVALE,
        ROLE_DISULF,
    )
    from science.dtie.v66.thermo_edge_features import GEO_DIM

    if not getattr(model, "chem_edge_mp", False):
        return {"alive": False, "skipped": True, "reason": "chem_edge_mp_off"}

    chosen: dict[str, Any] | None = None
    data_full: Any | None = None
    for prot in proteins:
        data = prepare_training_batch(
            model,
            prot,
            device,
            structural_disc_frozen=structural_disc_frozen,
        )
        if data.edge_attr is None or data.edge_attr.size(-1) < GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM:
            continue
        oh = data.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM]
        n_disulf = int((oh[:, ROLE_DISULF] > 0.5).sum().item())
        n_covale = int((oh[:, ROLE_COVALE] > 0.5).sum().item())
        if n_disulf + n_covale == 0:
            continue
        chosen = prot
        data_full = data
        break

    if chosen is None or data_full is None:
        return {
            "alive": False,
            "skipped": True,
            "reason": "no_chem_edges_in_corpus_sample",
            "n_disulf_edges": 0.0,
            "n_covale_edges": 0.0,
        }

    oh = data_full.edge_attr[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM]
    n_disulf = int((oh[:, ROLE_DISULF] > 0.5).sum().item())
    n_covale = int((oh[:, ROLE_COVALE] > 0.5).sum().item())

    model.eval()
    out_full = _forward_outputs(model, data_full)
    data_z = data_full.clone()
    data_z.edge_attr = data_z.edge_attr.clone()
    # Drop chem rows entirely (cleaner than zeroing one-hots, which would
    # reassign them to relation-0 via EquivariantConvMultiRel fallback).
    keep = ~(
        (data_z.edge_attr[:, GEO_DIM + ROLE_DISULF] > 0.5)
        | (data_z.edge_attr[:, GEO_DIM + ROLE_COVALE] > 0.5)
    )
    data_z.edge_index = data_z.edge_index[:, keep]
    data_z.edge_attr = data_z.edge_attr[keep]
    out_zero = _forward_outputs(model, data_z)

    deltas: dict[str, float] = {}
    for key in out_full:
        deltas[f"delta_{key}"] = _max_abs(out_full[key], out_zero[key])
    alive = any(v > eps for v in deltas.values())

    # Per-relation radial-MLP output variance on chem edges (utilization signal).
    rel_stats: dict[str, float] = {}
    try:
        conv0 = model.convs[0]
        dist = data_full.edge_attr[:, 3:4]
        for name, rel_id in (("disulf", ROLE_DISULF), ("covale", ROLE_COVALE)):
            mask = data_full.edge_attr[:, GEO_DIM + rel_id] > 0.5
            if not bool(mask.any()):
                rel_stats[f"radial_var_{name}"] = 0.0
                continue
            mlp = conv0.radial_mlps[rel_id]
            y = mlp(dist[mask])
            rel_stats[f"radial_var_{name}"] = float(y.float().var().item())
    except Exception as exc:  # pragma: no cover - defensive
        rel_stats["radial_var_error"] = 1.0
        logger.debug("chem radial var probe failed: %s", exc)

    return {
        "structure": f"{chosen.get('pdb_id')}:{chosen.get('chain')}",
        "n_disulf_edges": float(n_disulf),
        "n_covale_edges": float(n_covale),
        "alive": alive,
        "skipped": False,
        **deltas,
        **rel_stats,
    }


def run_feature_liveness_probes(
    model: nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    use_dehydron_barcode: bool,
    structural_disc_frozen: bool,
    fail_if_dead: bool = True,
    min_epochs_before_fail: int = 0,
    epoch_in_phase: int = 0,
    dehydron_edge_barcode: bool = False,
    chem_edge_mp: bool = False,
) -> dict[str, Any]:
    """Run barcode (if enabled) + MP + chem probes on the corpus."""
    if not proteins:
        return {"ok": True, "skipped": True, "reason": "no_proteins"}

    prot = proteins[0]
    report: dict[str, Any] = {
        "structure": f"{prot.get('pdb_id')}:{prot.get('chain')}",
        "ok": True,
    }

    mp = probe_mp_liveness(
        model,
        prot,
        device,
        structural_disc_frozen=structural_disc_frozen,
    )
    report["mp"] = mp
    if not structural_disc_frozen and not mp.get("alive"):
        msg = (
            "MP liveness dead: perturbing data.x did not move inference outputs "
            f"while structural_disc_frozen=False ({mp})"
        )
        report["ok"] = False
        report["mp_error"] = msg
        logger.warning(msg)

    if use_dehydron_barcode:
        bc = probe_barcode_liveness(
            model,
            prot,
            device,
            structural_disc_frozen=structural_disc_frozen,
        )
        report["barcode"] = bc
        if not bc.get("skipped") and not bc.get("alive"):
            msg = (
                "Barcode liveness dead: zeroing barcode columns did not move "
                f"inference outputs ({bc})"
            )
            report["barcode_error"] = msg
            if fail_if_dead and epoch_in_phase >= min_epochs_before_fail:
                report["ok"] = False
                logger.error(msg)
            else:
                logger.warning("%s (epoch %s < fail threshold %s)", msg, epoch_in_phase, min_epochs_before_fail)

    if dehydron_edge_barcode:
        bc = probe_dehydron_edge_barcode_liveness(
            model,
            prot,
            device,
            structural_disc_frozen=structural_disc_frozen,
        )
        report["barcode_edge"] = bc
        if not bc.get("skipped") and not bc.get("alive"):
            msg = (
                "Dehydron edge barcode liveness dead: zeroing edge barcode cols "
                f"did not move inference outputs ({bc})"
            )
            report["barcode_edge_error"] = msg
            if fail_if_dead and epoch_in_phase >= min_epochs_before_fail:
                report["ok"] = False
                logger.error(msg)
            else:
                logger.warning("%s (epoch %s < fail threshold %s)", msg, epoch_in_phase, min_epochs_before_fail)

    if chem_edge_mp:
        chem = probe_chem_edge_liveness(
            model,
            proteins,
            device,
            structural_disc_frozen=structural_disc_frozen,
        )
        report["chem"] = chem
        if not chem.get("skipped") and not chem.get("alive"):
            msg = (
                "Chem-edge liveness dead: removing disulf/covale edges did not "
                f"move inference outputs ({chem})"
            )
            report["chem_error"] = msg
            if fail_if_dead and epoch_in_phase >= min_epochs_before_fail:
                report["ok"] = False
                logger.error(msg)
            else:
                logger.warning(
                    "%s (epoch %s < fail threshold %s)",
                    msg,
                    epoch_in_phase,
                    min_epochs_before_fail,
                )

    return report
