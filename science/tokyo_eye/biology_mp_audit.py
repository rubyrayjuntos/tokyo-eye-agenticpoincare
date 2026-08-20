"""Audit counters for hyp_biology_mp — prove zero Cα leakage into Hyp MP."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

ALLOWED_BIOLOGY_EDGE_TYPES = ("hbond", "dehydron", "pi_stack", "salt_bridge")
FORBIDDEN_MP_EDGE_TYPES = ("ca_contact", "packing_euc", "se3_rel_xyz")

# Integer codes aligned with biology_graph.BIO_* constants.
BIO_TYPE_NAMES: dict[int, str] = {
    0: "hbond",
    1: "dehydron",
    2: "pi_stack",
    3: "salt_bridge",
}


def edge_type_counts(
    edge_types: Sequence[int] | np.ndarray | torch.Tensor | None,
    *,
    directed: bool = True,
) -> dict[str, int]:
    """Count undirected biology edges by type name (divide by 2 if directed)."""
    counts = {name: 0 for name in ALLOWED_BIOLOGY_EDGE_TYPES}
    if edge_types is None:
        return counts
    if torch.is_tensor(edge_types):
        arr = edge_types.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        arr = np.asarray(edge_types, dtype=np.int64).reshape(-1)
    denom = 2 if directed else 1
    for code, name in BIO_TYPE_NAMES.items():
        n = int(np.sum(arr == int(code)))
        counts[name] = n // denom
    return counts


def degree_zero_stats(
    edge_index: torch.Tensor | np.ndarray | None,
    num_nodes: int,
) -> dict[str, Any]:
    """Option A: degree-0 nodes are allowed; report count/fraction."""
    n = int(num_nodes)
    if n <= 0:
        return {"num_nodes": 0, "degree_zero_count": 0, "degree_zero_frac": 0.0}
    deg = np.zeros(n, dtype=np.int64)
    if edge_index is not None:
        if torch.is_tensor(edge_index):
            ei = edge_index.detach().cpu().numpy()
        else:
            ei = np.asarray(edge_index)
        if ei.size > 0:
            for i in ei.reshape(2, -1)[0]:
                ii = int(i)
                if 0 <= ii < n:
                    deg[ii] += 1
    z = int(np.sum(deg == 0))
    return {
        "num_nodes": n,
        "degree_zero_count": z,
        "degree_zero_frac": float(z) / float(n),
    }


def assert_biology_mp_ontology(
    edge_types: Sequence[int] | np.ndarray | torch.Tensor | None,
    *,
    ca_in_mp: bool,
    forbidden_labels: Sequence[str] | None = None,
) -> None:
    """Hard-fail if Cα leakage or forbidden labels appear."""
    if ca_in_mp:
        raise AssertionError("ca_in_mp must be false for hyp_biology_mp")
    for label in forbidden_labels or ():
        if label in FORBIDDEN_MP_EDGE_TYPES:
            raise AssertionError(f"forbidden MP edge type present: {label}")
    if edge_types is None:
        return
    if torch.is_tensor(edge_types):
        arr = edge_types.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        arr = np.asarray(edge_types, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return
    if int(arr.min()) < 0 or int(arr.max()) > max(BIO_TYPE_NAMES):
        raise AssertionError(
            f"edge_types out of biology ontology range: min={arr.min()} max={arr.max()}"
        )


def build_biology_mp_audit(
    *,
    edge_index: torch.Tensor | np.ndarray | None,
    edge_types: Sequence[int] | np.ndarray | torch.Tensor | None,
    num_nodes: int,
    ca_in_mp: bool,
    allow_ca_fallback: bool,
) -> dict[str, Any]:
    """Structured audit payload for trail / smoke grade."""
    counts = edge_type_counts(edge_types, directed=True)
    dz = degree_zero_stats(edge_index, num_nodes)
    audit = {
        "hyp_mp_edges": "biology",
        "ca_in_mp": bool(ca_in_mp),
        "allow_ca_fallback": bool(allow_ca_fallback),
        "degree_zero_nodes_allowed": True,
        "expected_edge_ontology": list(ALLOWED_BIOLOGY_EDGE_TYPES),
        "forbidden_edge_ontology": list(FORBIDDEN_MP_EDGE_TYPES),
        "edge_counts": counts,
        "n_biology_edges_undirected": int(sum(counts.values())),
        **dz,
    }
    assert_biology_mp_ontology(edge_types, ca_in_mp=ca_in_mp)
    if allow_ca_fallback:
        raise AssertionError("allow_ca_fallback must be false for hyp_biology_mp audit")
    return audit


def merge_audit_into_trail(
    audit_trail: dict[str, Any],
    biology_audit: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(audit_trail)
    out["hyp_biology_mp"] = True
    out["biology_mp"] = dict(biology_audit)
    out["ca_in_mp"] = bool(biology_audit.get("ca_in_mp", False))
    return out


__all__ = [
    "ALLOWED_BIOLOGY_EDGE_TYPES",
    "FORBIDDEN_MP_EDGE_TYPES",
    "BIO_TYPE_NAMES",
    "assert_biology_mp_ontology",
    "build_biology_mp_audit",
    "degree_zero_stats",
    "edge_type_counts",
    "merge_audit_into_trail",
]
