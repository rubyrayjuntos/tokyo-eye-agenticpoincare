"""Per-channel node feature transforms (T1a: z-score + optional τ rewrite)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import TAU


def rewrite_tau_channel(x4: torch.Tensor | np.ndarray, *, mode: str = "abs_dist") -> Any:
    """Replace binary tau_flag column with a ρ-referenced continuous channel.

    mode:
      - ``abs_dist``: |ρ − TAU|
      - ``flag``: leave as-is
    """
    if mode == "flag":
        return x4
    if mode != "abs_dist":
        raise ValueError(f"unknown tau rewrite mode: {mode}")
    if isinstance(x4, np.ndarray):
        out = np.array(x4, copy=True, dtype=np.float64)
        out[:, 1] = np.abs(out[:, 0] - float(TAU))
        return out
    out = x4.clone()
    out[:, 1] = torch.abs(out[:, 0] - float(TAU))
    return out


def fit_input_feature_stats(
    proteins: list[dict[str, Any]],
    *,
    replace_tau_with_abs_dist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Corpus mean/std over active node channels (3 topology or 4 legacy).

    Uses ``data.x`` width as-is (capped at 4). Channel 1 remains τ for both
    ``topology_three_vector`` ``[ρ, τ, ss]`` and legacy ``[ρ, τ, ss, SASA]``.
    """
    rows: list[np.ndarray] = []
    n_feat: int | None = None
    for prot in proteins:
        x = prot["data"].x.detach().cpu().numpy()
        d = int(x.shape[-1])
        if d < 3:
            raise ValueError(f"expected ≥3 node features for T1a fit, got {x.shape}")
        width = min(d, 4)
        if n_feat is None:
            n_feat = width
        elif width != n_feat:
            raise ValueError(
                f"mixed node feature widths in corpus: saw {n_feat} and {width}"
            )
        x_core = np.asarray(x[:, :width], dtype=np.float64).copy()
        if replace_tau_with_abs_dist:
            x_core = rewrite_tau_channel(x_core, mode="abs_dist")
        rows.append(x_core)
    X = np.concatenate(rows, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float64), std.astype(np.float64)


def transform_node_features(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    zscore: bool = True,
    replace_tau_with_abs_dist: bool = False,
) -> torch.Tensor:
    """Apply T1a transforms to ``data.x`` (preserves trailing dims beyond mean/std).

    Supports ``topology_three_vector`` (width 3) and ``legacy_four_vector`` (width 4).
    ``mean``/``std`` length must match the active core width.
    """
    if x.size(-1) < 3:
        raise ValueError(f"expected ≥3 node features, got {tuple(x.shape)}")
    mean_t = mean.to(device=x.device, dtype=x.dtype).reshape(-1)
    std_t = std.to(device=x.device, dtype=x.dtype).reshape(-1)
    if mean_t.numel() != std_t.numel():
        raise ValueError(
            f"mean/std length mismatch: mean={mean_t.numel()} std={std_t.numel()}"
        )
    width = int(mean_t.numel())
    if width < 3 or width > 4:
        raise ValueError(f"T1a mean/std width must be 3 or 4, got {width}")
    if x.size(-1) < width:
        raise ValueError(
            f"node features width {x.size(-1)} < fitted stats width {width}"
        )
    x_core = x[:, :width]
    if replace_tau_with_abs_dist:
        x_core = rewrite_tau_channel(x_core, mode="abs_dist")
    if zscore:
        x_core = (x_core - mean_t) / std_t
    if x.size(-1) > width:
        return torch.cat([x_core, x[:, width:]], dim=-1)
    return x_core


def apply_input_feature_norm_to_data(
    data: Data,
    model: torch.nn.Module,
) -> Data:
    """In-place-safe transform using buffers/flags on ``model``."""
    zscore = bool(getattr(model, "input_feature_zscore", False))
    replace = bool(getattr(model, "replace_tau_with_abs_dist", False))
    if not zscore and not replace:
        return data
    mean = getattr(model, "input_feat_mean", None)
    std = getattr(model, "input_feat_std", None)
    if mean is None or std is None:
        raise RuntimeError(
            "input_feature_zscore/replace_tau set but input_feat_mean/std buffers missing — "
            "call fit_and_install_input_feature_norm(...) after loading the corpus"
        )
    data = data.clone() if hasattr(data, "clone") else data
    data.x = transform_node_features(
        data.x,
        mean=mean,
        std=std,
        zscore=zscore or replace,  # abs-dist still needs scale match to fitted stats
        replace_tau_with_abs_dist=replace,
    )
    return data


def fit_and_install_input_feature_norm(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    *,
    zscore: bool = True,
    replace_tau_with_abs_dist: bool = False,
) -> dict[str, Any]:
    """Fit corpus stats and install buffers/flags on the model."""
    mean, std = fit_input_feature_stats(
        proteins, replace_tau_with_abs_dist=replace_tau_with_abs_dist
    )
    device = next(model.parameters()).device
    model.input_feature_zscore = bool(zscore)
    model.replace_tau_with_abs_dist = bool(replace_tau_with_abs_dist)
    # Re-register if already present.
    for name, arr in (("input_feat_mean", mean), ("input_feat_std", std)):
        t = torch.as_tensor(arr, dtype=torch.float32, device=device)
        if hasattr(model, name):
            getattr(model, name).copy_(t)
        else:
            model.register_buffer(name, t)
    return {
        "input_feature_zscore": bool(zscore),
        "replace_tau_with_abs_dist": bool(replace_tau_with_abs_dist),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "TAU": float(TAU),
    }
