"""Deterministic structural Poincaré disc placement (SSOT for viewer + slim GNN).

Composes per-residue disc coordinates from governed Tier-1 features (ρ, τ, Cα)
using hyperbolic Möbius addition — macro hub ⊕ micro offset — rather than
Euclidean stitching or GNN-learned layout.

Contract:
  - **Layout geometry** comes from this module (structural SSOT).
  - **τ vocabulary**: ``TAU`` (13.0) is the dehydron-density threshold in residue
    features. ``tau_flag`` is the binary governed feature (1 = sub-threshold
    dehydron, 0 = wrapped). It is **not** a separate rim-depth hyperparameter.
  - **Physics overlays** (L† conductance, hub_score, persistence) remain separate.
  - **GNN** should consume ``z_disc`` / ``z_ball`` as fixed input for MoE +
    uncertainty only (not re-learn radial/angular placement).

See ``hyperbolic_lorentz_ops`` for the n-dimensional Möbius / Lorentz primitives.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.common.graph_builder import GraphBuilder, ProteinGraph, ResidueFeatures
from science.dtie.common.hyperbolic_lorentz_ops import (
    ball_radius,
    clamp_to_ball,
    expmap0_tangent_at_origin,
    hyperbolic_distance_cap,
    hyperbolic_distance_from_origin,
    logmap0_at_origin,
    lorentzian_barycenter,
    mobius_add,
    poincare_distance,
)
from science.dtie.common.residue_features import TAU

logger = logging.getLogger(__name__)

DISC_DIM = 2
DEFAULT_MAX_DISC_FRAC = 0.92
LAYOUT_METHOD = "structural_tau_rho_pca_expmap0"
MACRO_NAMES = (
    "Folded core",
    "Secondary structure",
    "Allosteric linker",
    "Disordered periphery",
    "Surface rim",
)


class GraphDB(Protocol):
    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class StructuralResidueInput:
    residue_id: str
    residue_index: int
    chain_label: str
    rho: float
    tau_flag: float
    ss_type: float
    ca_xyz: np.ndarray  # shape (3,)


@dataclass
class MacroHub:
    macro_id: str
    name: str
    z_disc: np.ndarray  # shape (2,)
    member_indices: list[int] = field(default_factory=list)


@dataclass
class StructuralDiscNode:
    residue_id: str
    residue_index: int
    chain_label: str
    kind: str  # "macro_anchor" | "dehydron" | "residue"
    parent_macro_id: str
    z_disc: np.ndarray  # absolute 2D ball position
    z_micro: np.ndarray  # tangent micro offset: macro ⊕ micro = z_disc
    rho: float
    tau_flag: float
    ss_type: float
    phi: float
    depth_norm: float
    hyperbolic_r: float = 0.0


@dataclass
class StructuralDiscArtifact:
    structure_id: str
    curvature_c: float
    macros: list[MacroHub]
    nodes: list[StructuralDiscNode]
    layout_method: str = LAYOUT_METHOD

    @property
    def z_disc_matrix(self) -> np.ndarray:
        return np.stack([n.z_disc for n in self.nodes], axis=0)

    @property
    def residue_ids(self) -> list[str]:
        return [n.residue_id for n in self.nodes]


def residue_inputs_from_protein_graph(graph: ProteinGraph) -> list[StructuralResidueInput]:
    out: list[StructuralResidueInput] = []
    for r in graph.residues:
        out.append(
            StructuralResidueInput(
                residue_id=r.residue_id,
                residue_index=r.residue_index,
                chain_label=r.chain_label,
                rho=float(r.rho),
                tau_flag=float(r.tau_flag),
                ss_type=float(r.ss_type),
                ca_xyz=np.array([r.ca_x, r.ca_y, r.ca_z], dtype=np.float64),
            )
        )
    return out


def _structural_depth_norm(rho: np.ndarray, tau_flag: np.ndarray) -> np.ndarray:
    """0 = wrapped core (hyperbolic origin), 1 = τ-rim exposed (disc boundary).

    Tier-1 only (ρ, τ_flag) — no SASA. Continuous at ρ=TAU even when ``tau_flag``
    flips: ``rim_exposure`` depends on ρ alone; ``tau_flag`` only scales rim pull
  below TAU.
    """
    r = np.asarray(rho, dtype=np.float64)
    t = np.asarray(tau_flag, dtype=np.float64)
    buried = np.exp(-2.5 * r / TAU)
    rim_exposure = np.clip(1.0 - r / TAU, 0.0, 1.0)
    rim_pull = t * rim_exposure
    return np.clip((1.0 - rim_pull) * buried + rim_pull * rim_exposure, 0.0, 1.0)


def _pca_angles(ca_xyz: np.ndarray) -> np.ndarray:
    centered = ca_xyz - ca_xyz.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return np.linspace(0.0, 2.0 * np.pi, centered.shape[0], endpoint=False)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    return np.arctan2(plane[:, 1], plane[:, 0])


def _place_disc_from_depth_angle(
    depth_norm: float,
    phi: float,
    c: float,
    *,
    max_frac: float = DEFAULT_MAX_DISC_FRAC,
) -> np.ndarray:
    """Place a disc point via expmap₀ from origin (geodesic radius × direction).

    ``depth_norm`` maps linearly to geodesic distance in [0, d_max]; direction is
    the Cα PCA angle in the tangent plane at the origin (rotation is an isometry).
    """
    d_max = hyperbolic_distance_cap(c, max_frac=max_frac)
    d_h = float(np.clip(depth_norm, 0.0, 1.0)) * d_max
    tangent = np.array([d_h * np.cos(phi), d_h * np.sin(phi)], dtype=np.float64)
    z = expmap0_tangent_at_origin(tangent, c)
    return clamp_to_ball(z, c)


def _initial_disc_positions(
    residues: list[StructuralResidueInput],
    c: float,
    *,
    max_frac: float = DEFAULT_MAX_DISC_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(residues)
    ca = np.stack([r.ca_xyz for r in residues], axis=0)
    rho = np.array([r.rho for r in residues], dtype=np.float64)
    tau = np.array([r.tau_flag for r in residues], dtype=np.float64)
    depth = _structural_depth_norm(rho, tau)
    phi = _pca_angles(ca)
    z = np.stack(
        [
            _place_disc_from_depth_angle(depth[i], float(phi[i]), c, max_frac=max_frac)
            for i in range(n)
        ],
        axis=0,
    )
    hyp_r = np.array([hyperbolic_distance_from_origin(z[i], c) for i in range(n)])
    return z, depth, phi, hyp_r


def _pick_macro_count(n_residues: int) -> int:
    if n_residues < 8:
        return 1
    return int(min(5, max(2, n_residues // 40)))


def _farthest_point_seeds(z: np.ndarray, c: float, k: int) -> list[int]:
    n = z.shape[0]
    if k <= 1:
        return [0]
    chosen = [int(np.argmin(np.linalg.norm(z, axis=1)))]
    while len(chosen) < k:
        best_idx = 0
        best_dist = -1.0
        for i in range(n):
            if i in chosen:
                continue
            dist = min(poincare_distance(z[i], z[j], c) for j in chosen)
            if dist > best_dist:
                best_dist = dist
                best_idx = i
        chosen.append(best_idx)
    return chosen


def _assign_macros(
    z: np.ndarray,
    macro_z: np.ndarray,
    c: float,
) -> np.ndarray:
    n = z.shape[0]
    k = macro_z.shape[0]
    assign = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dists = [poincare_distance(z[i], macro_z[j], c) for j in range(k)]
        assign[i] = int(np.argmin(dists))
    return assign


def _refine_macros(
    z: np.ndarray,
    c: float,
    k: int,
    *,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = _farthest_point_seeds(z, c, k)
    macro_z = z[seeds].copy()
    assign = _assign_macros(z, macro_z, c)
    for _ in range(iterations):
        new_macros = []
        for j in range(k):
            members = z[assign == j]
            if len(members) == 0:
                new_macros.append(macro_z[j])
                continue
            weights = [1.0] * len(members)
            new_macros.append(clamp_to_ball(lorentzian_barycenter(members, weights, c), c))
        macro_z = np.stack(new_macros, axis=0)
        assign = _assign_macros(z, macro_z, c)
    return macro_z, assign


def compose_structural_disc(
    residues: list[StructuralResidueInput],
    curvature_c: float,
    *,
    structure_id: str = "",
    k_macros: int | None = None,
    max_disc_frac: float = DEFAULT_MAX_DISC_FRAC,
) -> StructuralDiscArtifact:
    """Build macro ⊕ micro structural disc layout from Tier-1 residue features."""
    c = require_learned_curvature(curvature_c, context="structural disc compose")
    if not residues:
        raise ValueError("compose_structural_disc requires at least one residue")

    z, depth, phi, hyp_r = _initial_disc_positions(
        residues, c, max_frac=max_disc_frac
    )
    k = k_macros if k_macros is not None else _pick_macro_count(len(residues))
    macro_z, assign = _refine_macros(z, c, k)

    macros: list[MacroHub] = []
    for j in range(k):
        member_idx = [i for i in range(len(residues)) if assign[i] == j]
        name = MACRO_NAMES[j] if j < len(MACRO_NAMES) else f"Macro {j + 1}"
        macros.append(
            MacroHub(
                macro_id=f"macro_{j}",
                name=name,
                z_disc=macro_z[j].copy(),
                member_indices=member_idx,
            )
        )

    nodes: list[StructuralDiscNode] = []
    for i, res in enumerate(residues):
        macro_id = f"macro_{int(assign[i])}"
        z_abs = z[i].copy()
        z_micro = mobius_add(-macro_z[assign[i]], z_abs, c)
        z_micro = clamp_to_ball(z_micro, c)
        kind = "dehydron" if res.tau_flag >= 0.5 else "residue"
        nodes.append(
            StructuralDiscNode(
                residue_id=res.residue_id,
                residue_index=res.residue_index,
                chain_label=res.chain_label,
                kind=kind,
                parent_macro_id=macro_id,
                z_disc=z_abs,
                z_micro=z_micro,
                rho=res.rho,
                tau_flag=res.tau_flag,
                ss_type=res.ss_type,
                phi=float(phi[i]),
                depth_norm=float(depth[i]),
                hyperbolic_r=float(hyp_r[i]),
            )
        )

    logger.info(
        "Structural disc composed structure=%s n=%d macros=%d c=%.6f",
        structure_id or "?",
        len(nodes),
        len(macros),
        c,
    )
    return StructuralDiscArtifact(
        structure_id=structure_id,
        curvature_c=c,
        macros=macros,
        nodes=nodes,
    )


def compose_from_protein_graph(
    graph: ProteinGraph,
    curvature_c: float,
    *,
    k_macros: int | None = None,
) -> StructuralDiscArtifact:
    return compose_structural_disc(
        residue_inputs_from_protein_graph(graph),
        curvature_c,
        structure_id=graph.structure_id,
        k_macros=k_macros,
    )


async def fetch_learned_curvature(
    db: GraphDB,
    structure_id: str,
) -> float:
    """Resolve learned curvature from the latest hyperbolic embedding_space row."""
    rows = await db.fetch_all(
        """
        SELECT es.curvature
        FROM fact_gnn_node_embedding e
        JOIN embedding_space es ON es.space_id = e.space_id
        JOIN provenance_run p ON p.run_id = e.run_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
          AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
        ORDER BY p.started_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if not rows:
        raise ValueError(f"No hyperbolic embedding_space curvature for {structure_id}")
    return require_learned_curvature(
        rows[0].get("curvature"),
        context=f"structure {structure_id}",
    )


async def load_and_compose(
    db: GraphDB,
    structure_id: str,
    *,
    curvature_c: float | None = None,
    chain_filter: str | None = None,
    k_macros: int | None = None,
) -> StructuralDiscArtifact:
    """Load governed residues via GraphBuilder and compose structural disc SSOT."""
    builder = GraphBuilder(db=db)
    graph = await builder.build_graph(structure_id, chain_filter=chain_filter)
    c = (
        require_learned_curvature(curvature_c, context="load_and_compose")
        if curvature_c is not None
        else await fetch_learned_curvature(db, structure_id)
    )
    return compose_from_protein_graph(graph, c, k_macros=k_macros)


def export_viewer_json(artifact: StructuralDiscArtifact) -> dict[str, Any]:
    """JSON payload for scale-invariant / MVP disc viewers."""
    return {
        "metadata": {
            "structure_id": artifact.structure_id,
            "curvature_c": artifact.curvature_c,
            "layout": artifact.layout_method,
            "layout_label": "structural SSOT (Tier-1 ρ, τ); not GNN-learned",
            "n_macros": len(artifact.macros),
            "n_residues": len(artifact.nodes),
            "disc_dim": DISC_DIM,
        },
        "macros": [
            {
                "id": m.macro_id,
                "name": m.name,
                "z_disc": m.z_disc.tolist(),
                "member_count": len(m.member_indices),
            }
            for m in artifact.macros
        ],
        "nodes": [
            {
                "id": n.residue_id,
                "residue_index": n.residue_index,
                "chain_label": n.chain_label,
                "kind": n.kind,
                "parent_macro_id": n.parent_macro_id,
                "z_disc": n.z_disc.tolist(),
                "z_micro": n.z_micro.tolist(),
                "rho": n.rho,
                "tau_flag": n.tau_flag,
                "ss_type": n.ss_type,
                "phi": n.phi,
                "depth_norm": n.depth_norm,
                "hyperbolic_r": n.hyperbolic_r,
                "cone_depth": n.hyperbolic_r,
            }
            for n in artifact.nodes
        ],
    }


def export_viewer_json_string(artifact: StructuralDiscArtifact, *, indent: int = 2) -> str:
    return json.dumps(export_viewer_json(artifact), indent=indent)


def verify_mobius_composition(artifact: StructuralDiscArtifact, *, atol: float = 1e-8) -> bool:
    """True when every node satisfies macro ⊕ micro ≈ z_disc."""
    c = artifact.curvature_c
    macro_by_id = {m.macro_id: m.z_disc for m in artifact.macros}
    for node in artifact.nodes:
        macro_z = macro_by_id[node.parent_macro_id]
        recomposed = mobius_add(macro_z, node.z_micro, c)
        if np.max(np.abs(recomposed - node.z_disc)) > atol:
            return False
    return True


def z_disc_matrix_for_residue_ids(
    artifact: StructuralDiscArtifact,
    residue_ids: list[str],
) -> np.ndarray:
    """Align artifact node order to ``residue_ids`` from GraphBuilder / PyG."""
    lookup = {n.residue_id: n.z_disc for n in artifact.nodes}
    missing = [rid for rid in residue_ids if rid not in lookup]
    if missing:
        raise ValueError(
            f"Structural disc missing {len(missing)} residues "
            f"(first: {missing[0]!r})"
        )
    return np.stack([lookup[rid] for rid in residue_ids], axis=0)


def _pyg_data_device(graph_data: Any) -> "torch.device":
    """Device of existing PyG tensors (attach must match for CUDA training)."""
    import torch

    for key in ("x", "edge_index", "edge_attr"):
        tensor = getattr(graph_data, key, None)
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


def attach_structural_disc_to_pyg(
    graph_data: Any,
    artifact: StructuralDiscArtifact,
    *,
    residue_ids: list[str] | None = None,
    use_hyperbolic_graph: bool = True,
    k_neighbors: int = 8,
) -> Any:
    """Attach frozen structural disc + optional hyperbolic k-NN graph to PyG ``Data``.

    **Mutates ``graph_data`` in place.** Callers comparing checkpoints or curvatures
    must pass ``prot["data"].clone()`` (see ``docs/audit/PYG_ATTACH_MUTATION.md``).

    Sets:
      - ``structural_z_disc`` [N, 2] float tensor
      - ``structural_z_disc_frozen`` True
      - ``structural_disc_layout`` str
      - ``structural_cone_depth`` hyperbolic geodesic depth per node (viewer SSOT)
      - When ``use_hyperbolic_graph``: sets ``hyperbolic_edge_index`` /
        ``hyperbolic_edge_attr`` (Cα ``edge_index`` / ``edge_attr`` unchanged).
    """
    import torch

    from science.dtie.common.hyperbolic_disc_graph import (
        build_hyperbolic_disc_graph,
        cone_depth_from_disc,
    )

    ids = residue_ids or list(getattr(graph_data, "residue_ids", artifact.residue_ids))
    z = z_disc_matrix_for_residue_ids(artifact, ids)
    c = float(artifact.curvature_c)
    device = _pyg_data_device(graph_data)
    graph_data.structural_z_disc = torch.tensor(z, dtype=torch.float32, device=device)
    graph_data.structural_z_disc_frozen = True
    graph_data.structural_disc_layout = artifact.layout_method
    graph_data.structural_disc_curvature_c = c
    hyp_depth = cone_depth_from_disc(z, c)
    graph_data.structural_cone_depth = torch.tensor(hyp_depth, dtype=torch.float32, device=device)

    if use_hyperbolic_graph:
        hyp_ei, hyp_ea = build_hyperbolic_disc_graph(z, c, k_neighbors=k_neighbors)
        graph_data.hyperbolic_edge_index = torch.tensor(hyp_ei, dtype=torch.long, device=device)
        graph_data.hyperbolic_edge_attr = torch.tensor(hyp_ea, dtype=torch.float32, device=device)
        graph_data.hyperbolic_graph = True

        num_nodes = z.shape[0]
        hyp_degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if hyp_ei.size > 0:
            src = graph_data.hyperbolic_edge_index[0]
            hyp_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.long))
        graph_data.hyperbolic_degree = hyp_degree
        graph_data.degree = hyp_degree

    return graph_data


def residue_inputs_from_training_prot(prot: dict[str, Any]) -> list[StructuralResidueInput]:
    """Build structural compose inputs from a training ``load_protein_graph`` dict."""
    data_x = prot["data"].x.detach().cpu().numpy()
    ca = prot["ca_coords"].detach().cpu().numpy()
    residue_ids = list(prot["residue_ids"])
    default_chain = str(prot.get("chain", "A"))
    out: list[StructuralResidueInput] = []
    for i, rid in enumerate(residue_ids):
        parts = str(rid).split(":")
        res_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else i + 1
        chain_label = parts[0] if parts and parts[0] else default_chain
        out.append(
            StructuralResidueInput(
                residue_id=str(rid),
                residue_index=res_index,
                chain_label=chain_label,
                rho=float(data_x[i, 0]),
                tau_flag=float(data_x[i, 1]),
                ss_type=float(data_x[i, 2]),
                ca_xyz=np.asarray(ca[i], dtype=np.float64),
            )
        )
    return out


def compose_from_training_prot(
    prot: dict[str, Any],
    curvature_c: float,
    *,
    k_macros: int | None = None,
) -> StructuralDiscArtifact:
    """Compose structural SSOT from a training protein dict (no DB round-trip)."""
    structure_id = str(prot.get("pdb_id") or prot.get("structure_id") or "")
    return compose_structural_disc(
        residue_inputs_from_training_prot(prot),
        curvature_c,
        structure_id=structure_id,
        k_macros=k_macros,
    )


def attach_structural_disc_for_forward(
    graph_data: Any,
    prot: dict[str, Any],
    curvature_c: float,
) -> Any:
    """Compose + attach structural disc before a v6 forward pass (training/diagnostics)."""
    artifact = compose_from_training_prot(prot, curvature_c)
    return attach_structural_disc_to_pyg(
        graph_data,
        artifact,
        residue_ids=list(prot.get("residue_ids") or []),
    )
