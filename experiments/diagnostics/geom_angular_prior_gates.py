"""Fix-1 geometric angular prior gates (1F88 gap, 4OBE circ-R, corr(r,depth))."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _largest_gap_deg(theta: np.ndarray) -> float:
    th = np.sort(np.mod(theta, 2.0 * np.pi))
    if th.size < 2:
        return float("nan")
    d = np.diff(th)
    d = np.append(d, th[0] + 2.0 * np.pi - th[-1])
    return float(np.degrees(d.max()))


def _circ_r(theta: np.ndarray) -> float:
    if theta.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * theta))))


def _match_x(model: torch.nn.Module, data: Any) -> Any:
    in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
    if data.x.size(-1) > in_f:
        data.x = data.x[:, :in_f].contiguous()
    return data


def measure_fix1_anchor_gates(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    anchors: tuple[str, ...] = ("1F88", "4OBE"),
    min_r: float = 0.05,
) -> dict[str, Any]:
    """Per-structure Fix-1 disc metrics on the learned pre-routing disc."""
    from experiments.training.v6.train_loop import prepare_training_batch

    by_id = {str(p.get("pdb_id", "")).upper(): p for p in proteins}
    out: dict[str, Any] = {"anchors": {}}
    model.eval()
    with torch.no_grad():
        for pdb_id in anchors:
            prot = by_id.get(pdb_id)
            if prot is None:
                out["anchors"][pdb_id] = {"missing": True}
                continue
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=False
            )
            data = _match_x(model, data)
            pred = model(data)
            xy = pred["hyp_projections_2d"].detach().cpu().numpy()
            depth = pred["cone_depth"].detach().cpu().numpy().reshape(-1)
            r = np.linalg.norm(xy, axis=1)
            theta = np.arctan2(xy[:, 1], xy[:, 0])
            mask = r >= min_r
            if int(mask.sum()) < 8:
                mask = np.ones_like(r, dtype=bool)
            r_m = r[mask]
            th_m = theta[mask]
            d_m = depth[mask]
            corr = float("nan")
            if r_m.size > 2 and float(np.std(r_m)) > 1e-8 and float(np.std(d_m)) > 1e-8:
                corr = float(np.corrcoef(r_m, d_m)[0, 1])
            out["anchors"][pdb_id] = {
                "missing": False,
                "n_vis": int(mask.sum()),
                "r_mean": float(r_m.mean()),
                "gap_deg": _largest_gap_deg(th_m),
                "circ_R": _circ_r(th_m),
                "corr_r_depth": corr,
            }
    a = out["anchors"]
    f88 = a.get("1F88") or {}
    obe = a.get("4OBE") or {}
    out["summary"] = {
        "1f88_gap_deg": f88.get("gap_deg"),
        "4obe_circ_R": obe.get("circ_R"),
        "1f88_corr_r_depth": f88.get("corr_r_depth"),
        "4obe_corr_r_depth": obe.get("corr_r_depth"),
        "4obe_circ_R_le_0p60": (
            float(obe["circ_R"]) <= 0.60
            if obe.get("circ_R") is not None and obe.get("missing") is False
            else None
        ),
    }
    return out
