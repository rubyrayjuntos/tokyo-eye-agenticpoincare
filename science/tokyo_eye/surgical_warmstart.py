"""Surgical B′ warm-start: Fix-1 donor → TokyoEye without Euc trunk transfer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from science.tokyo_eye.TokyoEye import (
    TokyoEye,
    infer_tokyo_eye_kwargs,
    load_tokyo_eye_state_dict,
)

# SE(3) Euc message-passing trunk — denied under Hyp MP primary.
DEFAULT_DENY_PREFIXES: tuple[str, ...] = ("convs.", "norms.")


@dataclass(frozen=True)
class SurgicalWarmstartReport:
    donor: str
    out: str
    transferred_keys: int
    denied_keys: int
    missing_after_load: int
    unexpected_after_load: int
    hyp_mp_primary: bool
    se3_aux: bool
    denied_prefixes: tuple[str, ...]


def filter_donor_state(
    state: dict[str, Any],
    *,
    deny_prefixes: tuple[str, ...] = DEFAULT_DENY_PREFIXES,
) -> tuple[dict[str, Any], list[str]]:
    """Drop Euc-trunk keys; leave hyp_mp absent so it stays cold-init."""
    kept: dict[str, Any] = {}
    denied: list[str] = []
    for key, value in state.items():
        if any(key.startswith(p) for p in deny_prefixes):
            denied.append(key)
            continue
        if key.startswith("hyp_mp."):
            # Donor should not have these; never transfer if present.
            denied.append(key)
            continue
        kept[key] = value
    return kept, denied


def build_tokyo_eye_from_donor_kwargs(
    state: dict[str, Any],
    *,
    architecture: dict[str, Any] | None,
    training_config: dict[str, Any] | None,
) -> TokyoEye:
    kwargs = infer_tokyo_eye_kwargs(state, architecture, training_config)
    kwargs["hyp_mp_primary"] = True
    kwargs["se3_aux"] = False
    node_dim = int(kwargs.pop("node_dim", 4))
    return TokyoEye(node_dim=node_dim, **kwargs)


def surgical_warmstart_tokyo_eye(
    donor_path: Path,
    *,
    out_path: Path,
    deny_prefixes: tuple[str, ...] = DEFAULT_DENY_PREFIXES,
    device: str = "cpu",
) -> SurgicalWarmstartReport:
    """Build TokyoEye, transfer allowed Fix-1 weights, write v7 warmstart ckpt."""
    raw = torch.load(donor_path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Donor checkpoint is not a dict: {donor_path}")
    state = raw.get("model_state_dict", raw)
    if not isinstance(state, dict):
        raise ValueError("Donor missing model_state_dict")
    architecture = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    training_config = (
        raw.get("training_config") if isinstance(raw.get("training_config"), dict) else {}
    )

    filtered, denied = filter_donor_state(state, deny_prefixes=deny_prefixes)
    model = build_tokyo_eye_from_donor_kwargs(
        filtered,
        architecture=architecture,
        training_config=training_config,
    )

    # Re-register T1a buffers before load when present on donor.
    if "input_feat_mean" in filtered and "input_feat_std" in filtered:
        mean = filtered["input_feat_mean"]
        std = filtered["input_feat_std"]
        if not hasattr(model, "input_feat_mean"):
            model.register_buffer("input_feat_mean", mean.clone())
            model.register_buffer("input_feat_std", std.clone())
        model.input_feature_zscore = True

    missing, unexpected = load_tokyo_eye_state_dict(model, filtered)
    model.hyp_mp_primary = True
    model.se3_aux = False
    model.hyperbolic_mp_graph = False
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    donor_tc = dict(training_config)
    payload = {
        "model_state_dict": model.state_dict(),
        "global_epoch": 0,
        "phase": 1,
        "phase_name": "bprime_warmstart",
        "score": float("-inf"),
        "architecture": {
            "version": "v7",
            "class": "TokyoEye",
            "hyp_mp_primary": True,
            "se3_aux": False,
            "bprime_surgical_warmstart": True,
            "donor": str(donor_path),
            "denied_prefixes": list(deny_prefixes),
        },
        "training_config": {
            **donor_tc,
            "gnn_lineage": "v7",
            "model_version": "TokyoEye-v7",
            "hyp_mp_primary": True,
            "se3_aux": False,
            "hyperbolic_mp_graph": False,
            "bprime_surgical_warmstart": True,
            "bprime_donor": str(donor_path),
        },
        "curvature": float(model.curvature.detach().cpu().item()),
        "meta": {
            "kind": "bprime_surgical_warmstart",
            "transferred_keys": len(filtered),
            "denied_keys": len(denied),
            "missing_after_load": len(missing),
            "unexpected_after_load": len(unexpected),
        },
    }
    torch.save(payload, out_path)
    return SurgicalWarmstartReport(
        donor=str(donor_path),
        out=str(out_path),
        transferred_keys=len(filtered),
        denied_keys=len(denied),
        missing_after_load=len(missing),
        unexpected_after_load=len(unexpected),
        hyp_mp_primary=True,
        se3_aux=False,
        denied_prefixes=deny_prefixes,
    )
