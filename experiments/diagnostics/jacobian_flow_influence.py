#!/usr/bin/env python3
"""Jacobian node-to-node influence probe — learned flow / hub / resistance.

Option B pivot: measure directional influence through the trained GNN rather than
reshaping disc geometry. Pre-registered in
``docs/specs/learned-flow-influence/ablation.md``.

For each target residue B:
  s_B = score(layer[B])   # default ``||layer[B]||²``; use ``pc1_sq`` = ``(h·û)²``
                          # when input z-norm makes magnitude an unreliable proxy
  influence(A → B) = ||∂s_B / ∂feat_A||
where ``feat`` is ``raw_x`` (default), ``post_zscore`` (``node_emb`` input), or
``post_node_emb`` — see ``--grad-site`` / JACOBIAN_ZNORM_DEFECT.md.

One backward pass per B; layers reported separately (trunk ``encoder_h`` vs disc
``hyp_projections_2d``) — never pooled (see DISC_PROJECTION_NOT_TRUNK_PROXY).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.training.gnn_lineage import load_model_from_checkpoint

# ---------------------------------------------------------------------------
# Numerical safety — single constant, reused everywhere (Gram-logdet style).
# ---------------------------------------------------------------------------
EPS = 1e-6
FLOW_CV_FLOOR = 0.02  # same shape as NU_CV_FLOOR
ASYMMETRY_FLOOR = 0.05  # mean pairwise asymmetry; near-zero ⇒ symmetric smoothing
ORIGIN_FLAG_EPS = EPS  # flag nodes within this of the origin in the scored layer


LayerName = Literal["encoder_h", "hyp_projections_2d"]
ScoreMode = Literal["norm_sq", "pc1_sq"]
GradSite = Literal["raw_x", "post_zscore", "post_node_emb"]


def _attach_grad_site_hooks(
    model: nn.Module, grad_site: GradSite
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Capture the tensor that influence norms will differentiate w.r.t.

    ``post_zscore`` / ``post_node_emb`` use ``node_emb`` pre/post hooks so the
    site is exactly the model's feature path (not a re-applied transform).
    """
    holders: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    if grad_site == "raw_x":
        return holders, handles
    if not hasattr(model, "node_emb"):
        raise RuntimeError(
            f"grad_site={grad_site!r} requires model.node_emb "
            "(raw_x works on hook-only stubs)"
        )
    if grad_site == "post_zscore":

        def _pre(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            holders["tensor"] = inputs[0]

        handles.append(model.node_emb.register_forward_pre_hook(_pre))
    elif grad_site == "post_node_emb":

        def _post(
            _mod: nn.Module, _inputs: tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            holders["tensor"] = out

        handles.append(model.node_emb.register_forward_hook(_post))
    else:
        raise ValueError(f"unknown grad_site: {grad_site}")
    return holders, handles


def safe_norm_sq(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Squared L2 with epsilon floor on the sum-of-squares (sqrt-safe upstream)."""
    return torch.clamp((x * x).sum(dim=dim), min=EPS)


def pc1_unit_direction(layer: torch.Tensor) -> torch.Tensor:
    """Unit PC1 of centered rows (detached SVD — direction is a fixed probe axis)."""
    H = layer.detach()
    if H.ndim != 2 or H.shape[0] < 2:
        raise ValueError(f"pc1_unit_direction expects (N,D) with N≥2, got {tuple(H.shape)}")
    Hc = H - H.mean(dim=0, keepdim=True)
    # torch.linalg.svd on float64; take Vh[0]
    _u, _s, vh = torch.linalg.svd(Hc, full_matrices=False)
    v = vh[0]
    nrm = torch.linalg.vector_norm(v)
    if float(nrm) < EPS:
        raise RuntimeError("degenerate PC1 direction (near-zero singular vector)")
    return v / nrm


def score_scalar(
    row: torch.Tensor,
    *,
    score_mode: ScoreMode,
    pc1_dir: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar energy for one residue used as autograd target.

    ``norm_sq``: ``||h||²`` (legacy; magnitude-sensitive under input z-norm).
    ``pc1_sq``: ``(h·û)²`` with fixed corpus-batch PC1 ``û`` (norm-invariant axis).
    """
    if score_mode == "norm_sq":
        return safe_norm_sq(row)
    if score_mode == "pc1_sq":
        if pc1_dir is None:
            raise ValueError("pc1_sq requires pc1_dir")
        proj = (row * pc1_dir).sum()
        return torch.clamp(proj * proj, min=EPS)
    raise ValueError(f"unknown score_mode: {score_mode}")


def coefficient_of_variation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < EPS:
        return float("nan")
    return float(np.std(arr) / abs(mean))


def asymmetry_index(influence: np.ndarray) -> dict[str, float]:
    """Mean |I_ab − I_ba| / (I_ab + I_ba) over off-diagonal finite pairs."""
    n = influence.shape[0]
    vals: list[float] = []
    for a in range(n):
        for b in range(a + 1, n):
            i_ab = influence[a, b]
            i_ba = influence[b, a]
            if not (np.isfinite(i_ab) and np.isfinite(i_ba)):
                continue
            denom = abs(i_ab) + abs(i_ba)
            if denom < EPS:
                continue
            vals.append(abs(i_ab - i_ba) / denom)
    if not vals:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "n_pairs": 0.0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n_pairs": float(arr.size),
    }


def flow_centralities(influence: np.ndarray) -> dict[str, np.ndarray]:
    """out(A)=Σ_B I(A→B), in(B)=Σ_A I(A→B); NaN entries excluded via nansum."""
    out_c = np.nansum(influence, axis=1)
    in_c = np.nansum(influence, axis=0)
    return {"out": out_c, "in": in_c, "total": out_c + in_c}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xr = np.argsort(np.argsort(x[mask])).astype(np.float64)
    yr = np.argsort(np.argsort(y[mask])).astype(np.float64)
    if float(np.std(xr)) < EPS or float(np.std(yr)) < EPS:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[mask], y[mask]
    point = spearman_corr(xx, yy)
    if xx.size < 5 or not np.isfinite(point):
        return {
            "spearman": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": float(xx.size),
        }
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, xx.size, size=xx.size)
        boots.append(spearman_corr(xx[idx], yy[idx]))
    arr = np.asarray(boots, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "spearman": point,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": float(xx.size),
        }
    return {
        "spearman": point,
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n": float(xx.size),
    }


def _cast_module_double(model: nn.Module) -> nn.Module:
    return model.double()


def _cast_batch_float64(data: Any, device: str) -> Any:
    """Promote all floating batch tensors to float64 (hyp-MP / side-channels too).

    ``compute_influence_matrix`` doubles the module for Jacobian stability; leaving
    ``hyperbolic_edge_attr`` (or ``ca_coords`` / priors) in float32 causes
    ``mat1 Float / mat2 Double`` failures inside EquivariantConv.
    """
    keys = list(data.keys()) if hasattr(data, "keys") else []
    for key in keys:
        val = data[key]
        if not torch.is_tensor(val):
            continue
        if val.is_floating_point():
            data[key] = val.detach().to(device=device, dtype=torch.float64)
        else:
            data[key] = val.to(device=device)
    return data


def _capture_layers(
    model: nn.Module,
) -> tuple[list[Any], dict[str, torch.Tensor]]:
    """Hook trunk without detaching — graph must stay live for autograd."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _radial_hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0]

    def _angular_hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if "encoder_h" not in captured and inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0]

    handles.append(model.radial_head.register_forward_hook(_radial_hook))
    if hasattr(model, "angular_head"):
        handles.append(model.angular_head.register_forward_hook(_angular_hook))
    return handles, captured


def jacobian_probe_node_count(
    data: Any,
    prot: dict[str, Any] | None = None,
) -> int:
    """Residue count for Jacobian / classical GT pools (excludes Path B parents)."""
    return residue_node_count(data, prot)


def compute_influence_matrix(
    model: nn.Module,
    data: Any,
    *,
    layer: LayerName,
    device: str,
    prot: dict[str, Any] | None = None,
    score_mode: ScoreMode = "norm_sq",
    grad_site: GradSite = "raw_x",
) -> dict[str, Any]:
    """Full N×N influence matrix for one layer via N backward passes.

    When ``data.n_residue_nodes`` (or ``prot["n_residues"]``) is set, the probe
    pool and matrix are restricted to leaf residues ``0:n_residue_nodes`` so
    Path B parent rows never enter asymmetry or classical GT correlation.

    ``score_mode``:
      - ``norm_sq``: ``s_B = ||layer[B]||²`` (legacy; magnitude-sensitive under z-norm)
      - ``pc1_sq``: ``s_B = (layer[B]·û)²`` with fixed PC1 ``û`` of the residue pool
        (norm-invariant; preferred when ``input_feature_zscore`` is on)

    ``grad_site``:
      - ``raw_x``: ``||∂s_B/∂data.x_A||`` (legacy default)
      - ``post_zscore``: ``||∂s_B/∂z_A||`` where ``z`` is ``node_emb`` input
        (after ``transform_node_features`` when z-norm is on)
      - ``post_node_emb``: ``||∂s_B/∂h0_A||`` where ``h0 = node_emb(z)``

    Excludes NaN/Inf entries rather than zero-filling. Flags near-origin nodes
    (``norm_sq``: small ``||h||``; ``pc1_sq``: small ``|h·û|``).
    """
    if score_mode not in ("norm_sq", "pc1_sq"):
        raise ValueError(f"unknown score_mode: {score_mode}")
    if grad_site not in ("raw_x", "post_zscore", "post_node_emb"):
        raise ValueError(f"unknown grad_site: {grad_site}")
    model.eval()
    model = _cast_module_double(model)
    data = _cast_batch_float64(data, device)
    # Re-bind x with grad after full-batch cast (detach cleared requires_grad).
    x = data.x.detach().to(device=device, dtype=torch.float64).requires_grad_(True)
    data.x = x

    handles, captured = _capture_layers(model)
    site_holders, site_handles = _attach_grad_site_hooks(model, grad_site)
    try:
        out = model(data)
    finally:
        for h in handles + site_handles:
            h.remove()

    if layer == "encoder_h":
        if "encoder_h" not in captured:
            raise RuntimeError("failed to capture encoder_h via radial/angular hooks")
        layer_t = captured["encoder_h"]
    else:
        disc = out.get("hyp_projections_2d")
        if disc is None:
            raise RuntimeError("model output missing hyp_projections_2d")
        layer_t = disc

    if layer_t.dtype != torch.float64:
        # Should already be float64 if model is double; keep reference for grad.
        pass

    if grad_site == "raw_x":
        grad_input = x
    else:
        if "tensor" not in site_holders:
            raise RuntimeError(
                f"grad_site={grad_site!r}: failed to capture intermediate via node_emb hook"
            )
        grad_input = site_holders["tensor"]

    n_total = int(layer_t.shape[0])
    n = jacobian_probe_node_count(data, prot)
    if n > n_total:
        raise ValueError(
            f"n_residue_nodes={n} exceeds layer rows {n_total} for {layer}"
        )
    layer_t = layer_t[:n]
    # Influence rows must match residue pool; intermediates may include parents.
    if int(grad_input.shape[0]) < n:
        raise ValueError(
            f"grad_site tensor rows {int(grad_input.shape[0])} < n_residues={n}"
        )
    pc1_dir: torch.Tensor | None = None
    if score_mode == "pc1_sq":
        pc1_dir = pc1_unit_direction(layer_t)

    influence = np.full((n, n), np.nan, dtype=np.float64)
    excluded: list[dict[str, Any]] = []
    norms = layer_t.detach().norm(dim=-1).cpu().numpy()
    if score_mode == "pc1_sq" and pc1_dir is not None:
        proj = (layer_t.detach() * pc1_dir.unsqueeze(0)).sum(dim=-1).abs().cpu().numpy()
        near_origin = [int(i) for i, r in enumerate(proj) if float(r) < ORIGIN_FLAG_EPS]
        score_magnitudes = proj
    else:
        near_origin = [int(i) for i, r in enumerate(norms) if float(r) < ORIGIN_FLAG_EPS]
        score_magnitudes = norms

    for b in range(n):
        s_b = score_scalar(layer_t[b], score_mode=score_mode, pc1_dir=pc1_dir)
        retain = b < n - 1
        try:
            (grad_feat,) = torch.autograd.grad(
                s_b,
                grad_input,
                retain_graph=retain,
                create_graph=False,
                allow_unused=False,
            )
        except RuntimeError as exc:
            excluded.append(
                {
                    "target_b": b,
                    "reason": f"autograd_failed:{exc}",
                    "scope": "column",
                }
            )
            continue

        if grad_feat is None:
            excluded.append(
                {"target_b": b, "reason": "grad_none", "scope": "column"}
            )
            continue

        g = grad_feat.detach()[:n]
        col_norm = torch.sqrt(torch.clamp((g * g).sum(dim=-1), min=EPS))
        col = col_norm.cpu().numpy()
        for a in range(n):
            val = float(col[a])
            if not np.isfinite(val):
                excluded.append(
                    {
                        "source_a": a,
                        "target_b": b,
                        "reason": "non_finite_influence",
                        "value": val,
                        "scope": "pair",
                    }
                )
                continue
            influence[a, b] = val

        del grad_feat, s_b

    # Free retained graph.
    del out, layer_t, captured, x, grad_input, site_holders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    cents = flow_centralities(influence)
    asym = asymmetry_index(influence)
    out_cv = coefficient_of_variation(cents["out"])
    in_cv = coefficient_of_variation(cents["in"])
    total_cv = coefficient_of_variation(cents["total"])

    liveness = {
        "flow_centrality_cv_out": out_cv,
        "flow_centrality_cv_in": in_cv,
        "flow_centrality_cv_total": total_cv,
        "flow_centrality_nondegenerate": bool(
            np.isfinite(total_cv) and total_cv >= FLOW_CV_FLOOR
        ),
        "asymmetry_mean": asym["mean"],
        "asymmetry_non_zero": bool(
            np.isfinite(asym["mean"]) and asym["mean"] >= ASYMMETRY_FLOOR
        ),
        "n_excluded_pairs": sum(1 for e in excluded if e.get("scope") == "pair"),
        "n_excluded_columns": sum(1 for e in excluded if e.get("scope") == "column"),
        "n_near_origin_nodes": len(near_origin),
        "near_origin_nodes": near_origin,
        "cv_floor": FLOW_CV_FLOOR,
        "asymmetry_floor": ASYMMETRY_FLOOR,
        "score_mode": score_mode,
        "grad_site": grad_site,
    }
    liveness["alive"] = bool(
        liveness["flow_centrality_nondegenerate"]
        and liveness["n_excluded_columns"] == 0
        and liveness["n_near_origin_nodes"] == 0
    )
    # Symmetric-but-nondegenerate is informative, not a fail — report separately.
    liveness["symmetric_but_nondegenerate"] = bool(
        liveness["flow_centrality_nondegenerate"]
        and not liveness["asymmetry_non_zero"]
        and liveness["n_excluded_columns"] == 0
    )

    return {
        "layer": layer,
        "score_mode": score_mode,
        "grad_site": grad_site,
        "n_residues": n,
        "n_total_nodes": n_total,
        "influence": influence,
        "centralities": {
            "out": cents["out"],
            "in": cents["in"],
            "total": cents["total"],
        },
        "asymmetry": asym,
        "layer_norms": norms.astype(np.float64),
        "score_magnitudes": np.asarray(score_magnitudes, dtype=np.float64),
        "min_layer_norm": float(np.min(norms)) if norms.size else float("nan"),
        "liveness": liveness,
        "excluded": excluded[:200],  # cap report size
        "excluded_total": len(excluded),
    }


def correlate_with_classical(
    centralities: Mapping[str, np.ndarray],
    classical: Mapping[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-structure Spearman + bootstrap CI vs classical metrics."""
    flow = np.asarray(centralities["total"], dtype=np.float64)
    out: dict[str, Any] = {}
    for key in (
        "betweenness",
        "current_flow_betweenness",
        "fiedler_abs",
        "anm_msf",
    ):
        y = np.asarray(classical[key], dtype=np.float64)
        out[key] = bootstrap_spearman(flow, y, seed=seed)
    return out


def evaluate_structure(
    *,
    prot: dict[str, Any],
    checkpoint: Path,
    device: str,
    layers: Sequence[LayerName] = ("encoder_h", "hyp_projections_2d"),
    score_mode: ScoreMode = "norm_sq",
    grad_site: GradSite = "raw_x",
) -> dict[str, Any]:
    pdb_id = str(prot.get("pdb_id") or prot.get("structure_id") or "").upper()
    ca = prot.get("ca_coords")
    if ca is None:
        raise ValueError(f"{pdb_id}: missing ca_coords")
    ca_np_full = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data = prepare_training_batch(model, prot, device)
    n_res = jacobian_probe_node_count(data, prot)
    ca_np = ca_np_full[:n_res] if ca_np_full.shape[0] > n_res else ca_np_full
    classical = classical_network_metrics(ca_np)
    # Serialize classical arrays as lists for JSON.
    classical_json = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in classical.items()
    }

    layer_reports: dict[str, Any] = {}
    for layer in layers:
        # Fresh batch per layer so prior autograd teardown cannot poison the next.
        model = load_model_from_checkpoint(checkpoint, device)
        model.eval()
        data = prepare_training_batch(model, prot, device)
        report = compute_influence_matrix(
            model,
            data,
            layer=layer,
            device=device,
            prot=prot,
            score_mode=score_mode,
            grad_site=grad_site,
        )
        corr = None
        if report["liveness"]["flow_centrality_nondegenerate"]:
            corr = correlate_with_classical(
                report["centralities"],
                classical,
                seed=hash(pdb_id) % (2**31),
            )
        # Drop full N×N matrix from JSON (huge); keep summary stats + diagonal-ish.
        influence = report["influence"]
        finite = influence[np.isfinite(influence)]
        layer_reports[layer] = {
            "n_residues": report["n_residues"],
            "score_mode": report.get("score_mode", score_mode),
            "grad_site": report.get("grad_site", grad_site),
            "min_layer_norm": report["min_layer_norm"],
            "liveness": report["liveness"],
            "asymmetry": report["asymmetry"],
            "influence_summary": {
                "finite_frac": float(finite.size / max(influence.size, 1)),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
            },
            "centralities": {
                "out": report["centralities"]["out"].tolist(),
                "in": report["centralities"]["in"].tolist(),
                "total": report["centralities"]["total"].tolist(),
            },
            "classical_correlation": corr,
            "excluded_total": report["excluded_total"],
            "excluded_sample": report["excluded"][:20],
            "near_origin_nodes": report["liveness"]["near_origin_nodes"],
        }
        del report, influence, model, data
        gc.collect()

    return {
        "pdb_id": pdb_id,
        "chain": prot.get("chain"),
        "checkpoint": str(checkpoint),
        "score_mode": score_mode,
        "grad_site": grad_site,
        "classical": {
            k: classical_json[k]
            for k in (
                "n_residues",
                "n_contact_edges",
                "contact_cutoff_angstrom",
                "lcc_size",
                "fiedler_value",
            )
        },
        "layers": layer_reports,
    }


def load_proteins_from_cache(
    cache_path: Path,
    pdb_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    wanted = {p.upper() for p in pdb_ids}
    out: dict[str, dict[str, Any]] = {}
    for prot in blob.get("proteins") or []:
        pid = str(prot.get("pdb_id") or prot.get("structure_id") or "").upper()
        if pid in wanted:
            out[pid] = prot
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"{cache_path}: missing structures {missing}")
    return out


# Stage A-12 floors — locked in ablation.md before scale-up.
HOLD_SPEARMAN_MIN = 0.30
HOLD_CI_LO_MIN = 0.0
WIN_MIN_HOLDS = 10
PARTIAL_MIN_HOLDS = 6
ANM_SECONDARY_MAX = -0.30


def structure_holds_trunk(layer_report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply locked Stage A-12 per-structure hold predicate (trunk only)."""
    live = layer_report.get("liveness") or {}
    corr = (layer_report.get("classical_correlation") or {}).get("betweenness") or {}
    anm = (layer_report.get("classical_correlation") or {}).get("anm_msf") or {}
    rho = corr.get("spearman")
    ci_lo = corr.get("ci_lo")
    anm_rho = anm.get("spearman")
    anm_ci_hi = anm.get("ci_hi")
    liveness_ok = bool(live.get("alive") and live.get("asymmetry_non_zero"))
    spearman_ok = (
        rho is not None
        and ci_lo is not None
        and np.isfinite(rho)
        and np.isfinite(ci_lo)
        and float(rho) >= HOLD_SPEARMAN_MIN
        and float(ci_lo) > HOLD_CI_LO_MIN
    )
    anm_ok = (
        anm_rho is not None
        and anm_ci_hi is not None
        and np.isfinite(anm_rho)
        and np.isfinite(anm_ci_hi)
        and float(anm_rho) <= ANM_SECONDARY_MAX
        and float(anm_ci_hi) < 0.0
    )
    return {
        "holds": bool(liveness_ok and spearman_ok),
        "liveness_ok": liveness_ok,
        "spearman_ok": spearman_ok,
        "anm_secondary_ok": anm_ok,
        "spearman_betweenness": rho,
        "ci_lo": ci_lo,
        "spearman_anm_msf": anm_rho,
    }


def aggregate_stage_a12_verdict(
    structures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Corpus outcome from per-structure holds — no pooling."""
    rows: list[dict[str, Any]] = []
    for s in structures:
        pid = str(s.get("pdb_id", "")).upper()
        trunk = (s.get("layers") or {}).get("encoder_h") or {}
        hold = structure_holds_trunk(trunk)
        rows.append({"pdb_id": pid, **hold})
    n_hold = sum(1 for r in rows if r["holds"])
    n = len(rows)
    if n_hold >= WIN_MIN_HOLDS:
        outcome = "win_graph_scaffolded_flow"
    elif n_hold >= PARTIAL_MIN_HOLDS:
        outcome = "partial"
    else:
        outcome = "fail"
    return {
        "outcome": outcome,
        "n_structures": n,
        "n_holds": n_hold,
        "win_min_holds": WIN_MIN_HOLDS,
        "partial_min_holds": PARTIAL_MIN_HOLDS,
        "hold_spearman_min": HOLD_SPEARMAN_MIN,
        "hold_ci_lo_min": HOLD_CI_LO_MIN,
        "claim_framing": (
            "graph-scaffolded, training-neutral unless ep1Δρ study shows training-additive"
        ),
        "per_structure": rows,
    }


def main() -> None:
    from experiments.training.v66.healthy_fix1 import HEALTHY_FIX1_CKPT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HEALTHY_FIX1_CKPT,
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    parser.add_argument(
        "--pdb-id",
        action="append",
        dest="pdb_ids",
        help="Repeatable. Default pilot: 4OBE only.",
    )
    parser.add_argument(
        "--stage-a12",
        action="store_true",
        help="Score all 12 enabled Stage A structures (pre-registered floors).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--ep0-control",
        type=Path,
        default=None,
        help="Optional near-epoch-0 checkpoint for known-degenerate control.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("norm_sq", "pc1_sq"),
        default="norm_sq",
        help=(
            "Autograd target: norm_sq=||h||² (legacy); "
            "pc1_sq=(h·û)² with fixed batch PC1 (norm-invariant under z-norm)."
        ),
    )
    parser.add_argument(
        "--grad-site",
        choices=("raw_x", "post_zscore", "post_node_emb"),
        default="raw_x",
        help=(
            "Differentiate influence w.r.t. raw input, post-zscore "
            "(node_emb input), or post-node_emb activations. "
            "Workaround for z-norm wash-out at raw_x — see JACOBIAN_ZNORM_DEFECT.md."
        ),
    )
    parser.add_argument(
        "--layers",
        choices=("encoder_h", "hyp_projections_2d", "both"),
        default="both",
        help="Which probe layers to score (default both; encoder_h for cheap sweeps).",
    )
    args = parser.parse_args()

    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    score_mode: ScoreMode = args.score_mode  # type: ignore[assignment]
    grad_site: GradSite = args.grad_site  # type: ignore[assignment]
    if args.layers == "both":
        layers: tuple[LayerName, ...] = ("encoder_h", "hyp_projections_2d")
    else:
        layers = (args.layers,)  # type: ignore[assignment]
    stage_a12 = (
        "1MBN",
        "1LYZ",
        "1BG1",
        "1F88",
        "2Z6H",
        "1HHP",
        "1TEN",
        "1UBQ",
        "1TIM",
        "4OBE",
        "1IVO",
        "2SHP",
    )
    if args.stage_a12:
        pdb_ids = list(stage_a12)
        default_out = Path(
            "checkpoints/v66/diagnostics/learned_flow_influence/stage_a12.json"
        )
    else:
        pdb_ids = [p.upper() for p in (args.pdb_ids or ["4OBE"])]
        default_out = Path(
            "checkpoints/v66/diagnostics/learned_flow_influence/pilot_4obe.json"
        )
    out_path = args.output or default_out
    proteins = load_proteins_from_cache(args.corpus_cache, pdb_ids)

    structures: list[dict[str, Any]] = []
    for pid in pdb_ids:
        print(
            f"evaluating {pid} on {args.checkpoint} "
            f"(score_mode={score_mode}, grad_site={grad_site}, layers={layers}) ...",
            flush=True,
        )
        structures.append(
            evaluate_structure(
                prot=proteins[pid],
                checkpoint=args.checkpoint,
                device=args.device,
                score_mode=score_mode,
                grad_site=grad_site,
                layers=layers,
            )
        )
        if args.ep0_control is not None:
            print(f"ep0-control {pid} on {args.ep0_control} ...", flush=True)
            ctrl = evaluate_structure(
                prot=proteins[pid],
                checkpoint=args.ep0_control,
                device=args.device,
                score_mode=score_mode,
                grad_site=grad_site,
                layers=layers,
            )
            ctrl["role"] = "ep0_degenerate_control"
            structures.append(ctrl)

    primary = [s for s in structures if s.get("role", "primary") == "primary"]
    verdict = aggregate_stage_a12_verdict(primary) if args.stage_a12 else None

    feat_label = {
        "raw_x": "x_A",
        "post_zscore": "z_A (node_emb input)",
        "post_node_emb": "h0_A (node_emb output)",
    }[grad_site]
    score_note = (
        f"influence(A→B)=||∂ s_B / ∂{feat_label}|| with s_B=||layer[B]||² (norm_sq)."
        if score_mode == "norm_sq"
        else (
            f"influence(A→B)=||∂ s_B / ∂{feat_label}|| with s_B=(layer[B]·û)² "
            "(pc1_sq); û = fixed PC1 of the residue pool "
            "(norm-invariant under input z-norm)."
        )
    )
    report = {
        "schema_version": 1,
        "probe": "jacobian_flow_influence",
        "score_mode": score_mode,
        "grad_site": grad_site,
        "layers": list(layers),
        "interpretation": (
            f"{score_note} Negative findings on asymmetry (near-zero) mean "
            "symmetric smoothing, not a probe failure. Trunk and disc are never "
            "pooled. Stage A-12 win uses per-structure holds only (ablation.md "
            "floors locked before scale-up)."
        ),
        "numerical_safety": {
            "EPS": EPS,
            "dtype": "float64",
            "flow_cv_floor": FLOW_CV_FLOOR,
            "asymmetry_floor": ASYMMETRY_FLOOR,
            "origin_flag_eps": ORIGIN_FLAG_EPS,
            "nan_policy": "exclude_and_count_never_zero_fill",
        },
        "checkpoint": str(args.checkpoint),
        "corpus_cache": str(args.corpus_cache),
        "structures": structures,
        "stage_a12_verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # Compact stdout summary
    summary = []
    for s in primary:
        row: dict[str, Any] = {"pdb_id": s["pdb_id"], "role": s.get("role", "primary")}
        for layer, lr in s["layers"].items():
            live = lr["liveness"]
            row[layer] = {
                "alive": live["alive"],
                "cv_total": live["flow_centrality_cv_total"],
                "asymmetry_mean": live["asymmetry_mean"],
                "symmetric_but_nondegenerate": live["symmetric_but_nondegenerate"],
                "n_near_origin": live["n_near_origin_nodes"],
                "n_excluded_pairs": live["n_excluded_pairs"],
                "corr_betweenness": (lr.get("classical_correlation") or {})
                .get("betweenness", {})
                .get("spearman"),
            }
        if "encoder_h" in s["layers"]:
            row["hold"] = structure_holds_trunk(s["layers"]["encoder_h"])
        summary.append(row)
    print(
        json.dumps(
            {
                "summary": summary,
                "stage_a12_verdict": verdict,
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
