"""Read-only edge-level telemetry for v6 GNN (MVP instrumentation).

Computes four quantities from existing physics + forward outputs without new
learned parameters. Run the telemetry-alive check before interpreting any
edge-flow or hub signal.

See ``docs/audit/EDGE_TELEMETRY.md`` and ``EDGE_TELEMETRY_MVP_SPEC.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import torch

from science.dtie.common.residue_features import TAU
from science.dtie.v5.phases.phase4_resistance import (
    CONTACT_CUTOFF,
    _build_conductance_graph,
)
from science.dtie.v6.gnn.model import resolve_message_passing_edges

# Guards against flat-std / dead-probe failure mode (see spec §3).
EDGE_EPI_STD_FLOOR = 1e-4
EDGE_ALE_STD_FLOOR = 1e-4
# Relative spread guard (optional); endpoint-mean dampens CV vs node-level epi.
EDGE_UNC_CV_FLOOR = 1e-5

MLFLOW_TAG_TELEMETRY = "telemetry/track"
MLFLOW_TAG_NOT_GATE = "not_p_entry_gate"


@dataclass
class EdgeTelemetryRecord:
    """Per-structure edge telemetry (aggregates + optional per-edge arrays)."""

    structure_id: str
    chain: str
    n_nodes: int
    n_edges: int
    edge_set: str  # "hyperbolic" | "ca_contact"
    edge_embed_resistance_corr: float
    edge_epistemic_var_mean: float
    edge_epistemic_var_std: float
    edge_aleatoric_var_mean: float
    edge_aleatoric_var_std: float
    same_expert_rate: float
    same_expert_null_rate: float
    same_expert_excess: float
    dominant_expert_share: float
    edge_flow_score_mean: float
    edge_flow_score_std: float
    tau_boundary_edge_fraction: float
    tau_boundary_edge_aleatoric_mean: float
    non_tau_boundary_edge_aleatoric_mean: float
    tau_boundary_aleatoric_lift: float
    flow_stratification: dict[str, Any] | None
    telemetry_alive: bool
    telemetry_alive_reason: str
    # Per-edge (optional export)
    edge_flow_score: np.ndarray | None = field(default=None, repr=False)
    edge_epistemic_var: np.ndarray | None = field(default=None, repr=False)
    edge_aleatoric_var: np.ndarray | None = field(default=None, repr=False)
    edge_same_expert: np.ndarray | None = field(default=None, repr=False)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _edge_effective_resistances(G: nx.Graph) -> dict[tuple[int, int], float]:
    """Pairwise effective resistance for every graph edge via Laplacian pseudoinverse."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return {}
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G, weight="conductance", nodelist=nodes).astype(np.float64).toarray()
    try:
        L_pinv = np.linalg.pinv(L)
    except np.linalg.LinAlgError:
        return {}
    out: dict[tuple[int, int], float] = {}
    for i, j in G.edges():
        ii, jj = idx[i], idx[j]
        r_eff = float(L_pinv[ii, ii] + L_pinv[jj, jj] - 2.0 * L_pinv[ii, jj])
        key = (min(i, j), max(i, j))
        out[key] = max(r_eff, 1e-12)
    return out


def _pair_resistance(
    r_map: dict[tuple[int, int], float], src: int, dst: int
) -> float:
    key = (min(src, dst), max(src, dst))
    if key in r_map:
        return r_map[key]
    return float("nan")


def edge_embed_signal_from_conv(
    model: torch.nn.Module,
    edge_attr: torch.Tensor,
) -> np.ndarray:
    """Learned edge signal: radial MLP weight norm on active MP edges (no attention in v6)."""
    conv = model.convs[-1]
    with torch.no_grad():
        dist = edge_attr[:, 3:4]
        weights = conv.radial_mlp(dist)
        signal = weights.norm(dim=-1).detach().cpu().numpy()
    return signal


def edge_uncertainty_from_nodes(
    edge_index: torch.Tensor,
    epistemic: torch.Tensor,
    aleatoric: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint-mean epistemic / aleatoric variance proxy per edge."""
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    epi = epistemic.detach().cpu().numpy().reshape(-1)
    ale = aleatoric.detach().cpu().numpy().reshape(-1)
    edge_epi = 0.5 * (epi[src] + epi[dst])
    edge_ale = 0.5 * (ale[src] + ale[dst])
    return edge_epi, edge_ale


def same_expert_stats(
    edge_index: torch.Tensor,
    expert_weights: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Same-expert edge rate vs independent-assignment null (sum p_e^2)."""
    w = expert_weights.detach()
    if w.dim() == 1:
        w = w.unsqueeze(-1)
    assign = w.argmax(dim=1).cpu().numpy()
    load = w.mean(dim=0).cpu().numpy()
    null_rate = float(np.sum(load * load))
    dominant = float(load.max()) if load.size else 0.0
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    if src.size == 0:
        return 0.0, null_rate, 0.0, dominant
    same_rate = float(np.mean(assign[src] == assign[dst]))
    excess = same_rate - null_rate
    return same_rate, null_rate, excess, dominant


def edge_flow_scores(
    G: nx.Graph,
    edge_index: torch.Tensor,
    *,
    cone_depths: np.ndarray | None = None,
) -> np.ndarray:
    """Betweenness × conductance flow proxy per active MP edge.

    Contact-graph edges use ``G`` conductance; hyperbolic-only pairs fall back to
    ``exp(depth_i + depth_j)`` so median stratification is not degenerate at zero.
    """
    if edge_index.size(1) == 0:
        return np.zeros(0, dtype=np.float64)
    n_nodes = int(cone_depths.shape[0]) if cone_depths is not None else 0
    bet: dict[int, float] = {}
    if G.number_of_edges() > 0:
        bet = nx.betweenness_centrality(G, weight="conductance")
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    scores = np.zeros(len(src), dtype=np.float64)
    for k, (i, j) in enumerate(zip(src, dst, strict=True)):
        if G.has_edge(int(i), int(j)):
            c = float(G.edges[int(i), int(j)]["conductance"])
        elif cone_depths is not None and n_nodes > max(int(i), int(j)):
            c = float(np.exp(cone_depths[int(i)] + cone_depths[int(j)]))
        else:
            c = 0.0
        bi = float(bet.get(int(i), 0.0))
        bj = float(bet.get(int(j), 0.0))
        scores[k] = c * 0.5 * (bi + bj)
    return scores


def _tau_boundary_mask(rho: np.ndarray, *, tau: float = TAU, band: float = 1.0) -> np.ndarray:
    return np.abs(rho - tau) <= band


def _unc_spread_ok(values: np.ndarray, *, std_floor: float, label: str) -> tuple[bool, str | None]:
    if values.size == 0:
        return False, f"{label}_no_edges"
    if not np.all(np.isfinite(values)):
        return False, f"{label}_non_finite"
    std = float(np.std(values))
    mean = float(np.mean(values))
    if std < std_floor:
        return False, f"{label}_flat_std={std:.2e}"
    if mean > 1e-12 and std / mean < EDGE_UNC_CV_FLOOR:
        return False, f"{label}_low_cv={std/mean:.2e}"
    return True, None


def assess_telemetry_alive(
    edge_epistemic_var: np.ndarray,
    edge_aleatoric_var: np.ndarray,
    edge_embed_resistance_corr: float,
) -> tuple[bool, str]:
    """Both epistemic and aleatoric edge spreads must be non-degenerate."""
    reasons: list[str] = []
    for arr, floor, label in (
        (edge_epistemic_var, EDGE_EPI_STD_FLOOR, "edge_epistemic"),
        (edge_aleatoric_var, EDGE_ALE_STD_FLOOR, "edge_aleatoric"),
    ):
        ok, reason = _unc_spread_ok(arr, std_floor=floor, label=label)
        if not ok and reason:
            reasons.append(reason)
    if not math.isfinite(edge_embed_resistance_corr):
        reasons.append("resistance_corr_non_finite")
    if reasons:
        return False, ";".join(reasons)
    return True, "ok"


def stratify_same_expert_by_flow(
    edge_index: torch.Tensor,
    expert_weights: torch.Tensor,
    flow_scores: np.ndarray,
    *,
    split: str = "median",
) -> dict[str, Any]:
    """High vs low edge_flow_score strata — healthy signature: excess_high > excess_low."""
    flow = np.asarray(flow_scores, dtype=np.float64)
    w = expert_weights.detach()
    if w.dim() == 1:
        w = w.unsqueeze(-1)
    assign = w.argmax(dim=1).cpu().numpy()
    load = w.mean(dim=0).cpu().numpy()
    null_rate = float(np.sum(load * load))
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    if src.size == 0 or flow.size == 0:
        return {
            "flow_split": split,
            "flow_threshold": float("nan"),
            "high_flow": None,
            "low_flow": None,
            "same_expert_excess_high_minus_low": float("nan"),
            "healthy_flow_alignment": False,
        }

    same = assign[src] == assign[dst]
    if split == "median":
        threshold = float(np.median(flow))
        high_m = flow >= threshold
        low_m = flow < threshold
        if not np.any(low_m) or not np.any(high_m):
            q75, q25 = float(np.percentile(flow, 75)), float(np.percentile(flow, 25))
            threshold = q75
            high_m = flow >= q75
            low_m = flow <= q25
    else:
        q75, q25 = float(np.percentile(flow, 75)), float(np.percentile(flow, 25))
        threshold = q75
        high_m = flow >= q75
        low_m = flow <= q25

    def _stratum(mask: np.ndarray) -> dict[str, float] | None:
        if not np.any(mask):
            return None
        rate = float(np.mean(same[mask]))
        return {
            "n_edges": int(np.sum(mask)),
            "same_expert_rate": rate,
            "same_expert_null_rate": null_rate,
            "same_expert_excess": rate - null_rate,
            "edge_flow_score_mean": float(np.mean(flow[mask])),
        }

    high = _stratum(high_m)
    low = _stratum(low_m)
    delta = float("nan")
    healthy = False
    if high is not None and low is not None:
        delta = high["same_expert_excess"] - low["same_expert_excess"]
        healthy = delta > 0.0

    return {
        "flow_split": split,
        "flow_threshold": threshold,
        "high_flow": high,
        "low_flow": low,
        "same_expert_excess_high_minus_low": delta,
        "healthy_flow_alignment": healthy,
    }


def _residue_only_mp_edges(
    data: Any,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Drop Path B parent endpoints from telemetry MP edges.

    When ``data.n_residue_nodes`` is set, containment parents sit at
    ``[n_residue_nodes:]``. Residue metrics (uncertainty, expert weights)
    are leaf-aligned — parent indices IndexError unless filtered.
    """
    n_res = getattr(data, "n_residue_nodes", None)
    if n_res is None:
        return edge_index, edge_attr, int(data.x.size(0))
    n_res = int(n_res)
    if n_res <= 0 or edge_index.numel() == 0:
        return edge_index, edge_attr, n_res
    src, dst = edge_index[0], edge_index[1]
    mask = (src < n_res) & (dst < n_res)
    if bool(mask.all()):
        return edge_index, edge_attr, n_res
    return edge_index[:, mask], edge_attr[mask], n_res


def collect_edge_telemetry(
    model: torch.nn.Module,
    data: Any,
    output: dict[str, Any],
    *,
    structure_id: str = "",
    chain: str = "A",
    ca_coords: np.ndarray | None = None,
    include_per_edge: bool = False,
) -> EdgeTelemetryRecord:
    """Compute edge telemetry for one forward pass (read-only)."""
    mp_ei, mp_ea = resolve_message_passing_edges(data)
    mp_ei, mp_ea, n_nodes = _residue_only_mp_edges(data, mp_ei, mp_ea)
    edge_set = (
        "hyperbolic"
        if bool(getattr(data, "hyperbolic_graph", False))
        and getattr(data, "hyperbolic_edge_index", None) is not None
        else "ca_contact"
    )

    unc = output["uncertainty"]
    epi = unc["epistemic"].squeeze(-1)
    ale = unc["aleatoric"].squeeze(-1)
    expert_weights = output["expert_weights"]
    # Defensive: callers may pass unsliced parent-inclusive forwards.
    if epi.shape[0] > n_nodes:
        epi = epi[:n_nodes]
        ale = ale[:n_nodes]
    if expert_weights.shape[0] > n_nodes:
        expert_weights = expert_weights[:n_nodes]
    edge_epi, edge_ale = edge_uncertainty_from_nodes(mp_ei, epi, ale)

    embed_signal = edge_embed_signal_from_conv(model, mp_ea)

    cone_depth = output["cone_depth"].squeeze(-1).detach().cpu().numpy()
    if cone_depth.shape[0] > n_nodes:
        cone_depth = cone_depth[:n_nodes]
    if ca_coords is None:
        ca_coords = getattr(data, "pos", None)
        if ca_coords is not None:
            ca_coords = ca_coords.detach().cpu().numpy()
    if ca_coords is None:
        ca_coords = np.full((cone_depth.shape[0], 3), np.nan)
    ca_coords = np.asarray(ca_coords, dtype=np.float64)
    if ca_coords.shape[0] > n_nodes:
        ca_coords = ca_coords[:n_nodes]

    G = _build_conductance_graph(
        ca_coords,
        cone_depth,
        cutoff=CONTACT_CUTOFF,
    )
    r_map = _edge_effective_resistances(G)
    src, dst = mp_ei[0].cpu().numpy(), mp_ei[1].cpu().numpy()
    r_eff = np.array(
        [_pair_resistance(r_map, int(i), int(j)) for i, j in zip(src, dst, strict=True)],
        dtype=np.float64,
    )
    corr = _safe_pearson(embed_signal, r_eff)

    flow = edge_flow_scores(G, mp_ei, cone_depths=cone_depth)
    same_rate, null_rate, excess, dominant = same_expert_stats(
        mp_ei, expert_weights
    )
    flow_strat = stratify_same_expert_by_flow(mp_ei, expert_weights, flow)

    rho = data.x[:n_nodes, 0].detach().cpu().numpy()
    tau_mask = _tau_boundary_mask(rho)
    boundary_edges = float(
        np.mean(tau_mask[src] | tau_mask[dst])
    ) if src.size else 0.0
    if src.size:
        edge_tau = tau_mask[src] | tau_mask[dst]
        ale_tau_mean = float(np.mean(edge_ale[edge_tau])) if edge_tau.any() else float("nan")
        ale_non_mean = float(np.mean(edge_ale[~edge_tau])) if (~edge_tau).any() else float("nan")
    else:
        ale_tau_mean = ale_non_mean = float("nan")
    ale_lift = (
        ale_tau_mean - ale_non_mean
        if math.isfinite(ale_tau_mean) and math.isfinite(ale_non_mean)
        else float("nan")
    )

    assign = expert_weights.argmax(dim=1).cpu().numpy()
    edge_same = (assign[src] == assign[dst]).astype(np.int8) if src.size else np.array([], dtype=np.int8)

    alive, alive_reason = assess_telemetry_alive(edge_epi, edge_ale, corr)

    return EdgeTelemetryRecord(
        structure_id=structure_id.lower(),
        chain=chain,
        n_nodes=n_nodes,
        n_edges=int(mp_ei.size(1)),
        edge_set=edge_set,
        edge_embed_resistance_corr=corr,
        edge_epistemic_var_mean=float(np.mean(edge_epi)) if edge_epi.size else 0.0,
        edge_epistemic_var_std=float(np.std(edge_epi)) if edge_epi.size else 0.0,
        edge_aleatoric_var_mean=float(np.mean(edge_ale)) if edge_ale.size else 0.0,
        edge_aleatoric_var_std=float(np.std(edge_ale)) if edge_ale.size else 0.0,
        same_expert_rate=same_rate,
        same_expert_null_rate=null_rate,
        same_expert_excess=excess,
        dominant_expert_share=dominant,
        edge_flow_score_mean=float(np.mean(flow)) if flow.size else 0.0,
        edge_flow_score_std=float(np.std(flow)) if flow.size else 0.0,
        tau_boundary_edge_fraction=boundary_edges,
        tau_boundary_edge_aleatoric_mean=ale_tau_mean,
        non_tau_boundary_edge_aleatoric_mean=ale_non_mean,
        tau_boundary_aleatoric_lift=ale_lift,
        flow_stratification=flow_strat,
        telemetry_alive=alive,
        telemetry_alive_reason=alive_reason,
        edge_flow_score=flow if include_per_edge else None,
        edge_epistemic_var=edge_epi if include_per_edge else None,
        edge_aleatoric_var=edge_ale if include_per_edge else None,
        edge_same_expert=edge_same if include_per_edge else None,
    )


def record_to_summary_dict(rec: EdgeTelemetryRecord) -> dict[str, Any]:
    """JSON-serializable summary (drops per-edge arrays)."""
    d = asdict(rec)
    for key in (
        "edge_flow_score",
        "edge_epistemic_var",
        "edge_aleatoric_var",
        "edge_same_expert",
    ):
        d.pop(key, None)
    return d


def record_to_export_dict(rec: EdgeTelemetryRecord) -> dict[str, Any]:
    """Full export including per-edge arrays for baseline JSON."""
    d = record_to_summary_dict(rec)
    if rec.edge_flow_score is not None:
        d["per_edge"] = {
            "edge_flow_score": rec.edge_flow_score.tolist(),
            "edge_epistemic_var": rec.edge_epistemic_var.tolist()
            if rec.edge_epistemic_var is not None
            else [],
            "edge_aleatoric_var": rec.edge_aleatoric_var.tolist()
            if rec.edge_aleatoric_var is not None
            else [],
            "edge_same_expert": rec.edge_same_expert.tolist()
            if rec.edge_same_expert is not None
            else [],
        }
    return d


def aggregate_corpus_records(records: list[EdgeTelemetryRecord]) -> dict[str, float]:
    """Corpus-level means for baseline JSON."""
    if not records:
        return {}
    keys = [
        "edge_embed_resistance_corr",
        "edge_epistemic_var_std",
        "edge_aleatoric_var_std",
        "same_expert_rate",
        "same_expert_null_rate",
        "same_expert_excess",
        "dominant_expert_share",
        "edge_flow_score_mean",
        "tau_boundary_edge_fraction",
        "tau_boundary_aleatoric_lift",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [getattr(r, k) for r in records if math.isfinite(getattr(r, k))]
        out[f"corpus_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["corpus_telemetry_alive_fraction"] = float(
        np.mean([1.0 if r.telemetry_alive else 0.0 for r in records])
    )
    out["corpus_all_finite"] = float(
        all(
            math.isfinite(r.edge_embed_resistance_corr)
            and math.isfinite(r.edge_epistemic_var_std)
            and math.isfinite(r.edge_aleatoric_var_std)
            for r in records
        )
    )
    deltas = [
        (r.flow_stratification or {}).get("same_expert_excess_high_minus_low")
        for r in records
    ]
    deltas_f = [d for d in deltas if d is not None and math.isfinite(d)]
    out["corpus_flow_excess_high_minus_low_mean"] = (
        float(np.mean(deltas_f)) if deltas_f else float("nan")
    )
    out["corpus_healthy_flow_alignment_fraction"] = float(
        np.mean(
            [
                1.0
                if (r.flow_stratification or {}).get("healthy_flow_alignment")
                else 0.0
                for r in records
            ]
        )
    )
    return out


# Maps ``aggregate_corpus_records`` keys → flat health dict keys for MLflow / epoch logs.
_CORPUS_AGGREGATE_TO_HEALTH: dict[str, str] = {
    "corpus_edge_embed_resistance_corr_mean": "edge_embed_resistance_corr_mean",
    "corpus_edge_epistemic_var_std_mean": "edge_epistemic_var_std_mean",
    "corpus_edge_aleatoric_var_std_mean": "edge_aleatoric_var_std_mean",
    "corpus_same_expert_rate_mean": "same_expert_rate_mean",
    "corpus_same_expert_null_rate_mean": "same_expert_null_rate_mean",
    "corpus_same_expert_excess_mean": "same_expert_excess_mean",
    "corpus_flow_excess_high_minus_low_mean": "flow_excess_high_minus_low_mean",
    "corpus_telemetry_alive_fraction": "edge_telemetry_alive_fraction",
    "corpus_healthy_flow_alignment_fraction": "healthy_flow_alignment_fraction",
    "corpus_tau_boundary_aleatoric_lift_mean": "edge_tau_boundary_aleatoric_lift_mean",
}


def corpus_aggregate_to_health(agg: dict[str, float]) -> dict[str, float]:
    """Flatten corpus aggregate into ``measure_geometry_health`` / MLflow health keys."""
    out: dict[str, float] = {}
    for corp_key, health_key in _CORPUS_AGGREGATE_TO_HEALTH.items():
        val = agg.get(corp_key)
        if val is not None and math.isfinite(float(val)):
            out[health_key] = float(val)
    return out
