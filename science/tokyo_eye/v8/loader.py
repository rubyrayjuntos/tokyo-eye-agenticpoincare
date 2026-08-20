"""TokyoEye-v8 curated PDB dataset (Sprint 6 + Sprint 8 graph cache).

Mode A: single structure (default ``4OBE:A``).
Mode B/C: Stage A-style manifest ``proteins[]`` with ``enabled`` filter.

Isolated: no v66 ``load_training_proteins`` / Normalizer.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from science.tokyo_eye.v8.biophysics import (
    BIOPHYS_CACHE_VERSION,
    DSSP_ENERGY_CUTOFF,
    DSSP_FQQ,
    WRAP_CONE_HALF_ANGLE_DEG,
    WRAPPING_RADIUS,
    parse_residue_records_from_pdb_chain,
)
from science.tokyo_eye.v8.r0_r5_graph import (
    R0_COVALENT,
    R1_HBOND,
    R2_DEHYDRON,
    R3_HYDROPHOBIC_PI,
    R4_SALT_BRIDGE,
    R5_LOCAL_NEIGHBORHOOD,
    WRAPPING_RADIUS_A,
    build_r0_r5_graph,
    get_dehydron_wrap_max,
)
from science.tokyo_eye.v8.types import ResidueRecord

DEFAULT_PDB_ID = "4OBE"
DEFAULT_CHAIN = "A"
DEFAULT_PDB_DIR = Path("pdb_cache")
DEFAULT_GRAPH_CACHE_DIR = Path("pdb_cache/v8_graph_cache")
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Experimental 5-way SDRP heuristic class ids (not ground truth)
SDRP_CORE = 0
SDRP_DEHYDRON_RIM = 1
SDRP_SALT = 2
SDRP_HYDROPHOBE = 3
SDRP_NEIGHBORHOOD = 4
NUM_SDRP_HEURISTIC = 5


def ensure_pdb_cached(pdb_id: str, pdb_dir: Path | str) -> Path:
    """Return local ``{pdb_id}.pdb``, downloading from RCSB if missing."""
    import shutil
    import subprocess

    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.strip().upper()
    local = pdb_dir / f"{pdb_id}.pdb"
    if local.is_file() and local.stat().st_size > 0:
        return local
    url = RCSB_DOWNLOAD.format(pdb_id=pdb_id)
    last_err: Exception | None = None
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            local.write_bytes(resp.read())
        return local
    except Exception as exc:  # noqa: BLE001 — fall through to curl
        last_err = exc
    if shutil.which("curl"):
        tmp = local.with_suffix(".pdb.partial")
        try:
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url],
                check=True,
                capture_output=True,
                text=True,
            )
            if tmp.is_file() and tmp.stat().st_size > 0:
                tmp.replace(local)
                return local
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {pdb_id} from RCSB: {last_err}")


def parse_enabled_manifest(manifest_path: Path | str) -> list[dict[str, str]]:
    """Strict Stage A schema: ``proteins[]`` filtered by ``enabled``."""
    path = Path(manifest_path)
    with path.open() as f:
        manifest_data = json.load(f)
    proteins = manifest_data.get("proteins", [])
    if not isinstance(proteins, list):
        raise ValueError(f"manifest {path} missing proteins[] list")
    enabled_proteins = [p for p in proteins if p.get("enabled", True)]
    out: list[dict[str, str]] = []
    for p in enabled_proteins:
        pdb_id = str(p.get("pdb_id") or p.get("pdb") or "").strip().upper()
        chain = str(p.get("chain") or "A").strip()
        if not pdb_id:
            raise ValueError(f"enabled protein missing pdb_id in {path}: {p}")
        out.append({"pdb_id": pdb_id, "chain": chain})
    if not out:
        raise ValueError(f"no enabled proteins in manifest {path}")
    return out


def graph_cache_hash(
    *,
    wrap_max: int | None = None,
    dssp_cutoff: float = DSSP_ENERGY_CUTOFF,
    cone_deg: float = WRAP_CONE_HALF_ANGLE_DEG,
    wrap_radius: float = WRAPPING_RADIUS,
    version: str = BIOPHYS_CACHE_VERSION,
) -> str:
    """Stable short hash of biophysics hyperparameters (cache key fragment)."""
    payload = {
        "version": version,
        "dssp_energy_cutoff": float(dssp_cutoff),
        "dssp_fqq": float(DSSP_FQQ),
        "wrap_cone_half_angle_deg": float(cone_deg),
        "wrapping_radius": float(wrap_radius),
        "dehydron_wrap_max": int(
            get_dehydron_wrap_max() if wrap_max is None else wrap_max
        ),
        "wrapping_radius_a": float(WRAPPING_RADIUS_A),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def graph_cache_path(
    pdb_id: str,
    chain: str,
    *,
    cache_dir: Path | str = DEFAULT_GRAPH_CACHE_DIR,
    wrap_max: int | None = None,
) -> Path:
    h = graph_cache_hash(wrap_max=wrap_max)
    return Path(cache_dir) / f"{pdb_id.upper()}_{chain}_{h}.pt"


def ca_coords(records: Sequence[ResidueRecord]) -> np.ndarray:
    rows = []
    for r in records:
        ca = r.get_atom("CA")
        if ca is None:
            raise ValueError(f"residue {r.chain_label}{r.residue_index} missing CA")
        rows.append(ca.coord.astype(np.float32))
    return np.stack(rows, axis=0)


def dehydron_labels_from_edges(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
) -> np.ndarray:
    """Binary node labels: 1 iff incident to any R2 dehydron edge."""
    labels = np.zeros(num_nodes, dtype=np.float32)
    if edge_type.size == 0:
        return labels
    mask = edge_type == R2_DEHYDRON
    if not np.any(mask):
        return labels
    ei = edge_index[:, mask]
    labels[ei[0]] = 1.0
    labels[ei[1]] = 1.0
    return labels


def sdrp_heuristic_from_edges(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
) -> np.ndarray:
    """Experimental 5-way class from local edge-type mix (argmax counts)."""
    counts = np.zeros((num_nodes, NUM_SDRP_HEURISTIC), dtype=np.float32)
    if edge_type.size == 0:
        return np.zeros(num_nodes, dtype=np.int64)
    src, dst = edge_index[0], edge_index[1]
    for i, t in enumerate(edge_type.tolist()):
        a, b = int(src[i]), int(dst[i])
        if t == R2_DEHYDRON:
            counts[a, SDRP_DEHYDRON_RIM] += 1
            counts[b, SDRP_DEHYDRON_RIM] += 1
        elif t == R1_HBOND or t == R0_COVALENT:
            counts[a, SDRP_CORE] += 1
            counts[b, SDRP_CORE] += 1
        elif t == R4_SALT_BRIDGE:
            counts[a, SDRP_SALT] += 1
            counts[b, SDRP_SALT] += 1
        elif t == R3_HYDROPHOBIC_PI:
            counts[a, SDRP_HYDROPHOBE] += 1
            counts[b, SDRP_HYDROPHOBE] += 1
        elif t == R5_LOCAL_NEIGHBORHOOD:
            counts[a, SDRP_NEIGHBORHOOD] += 1
            counts[b, SDRP_NEIGHBORHOOD] += 1
    # Prefer dehydron rim when tied with core
    order = [
        SDRP_DEHYDRON_RIM,
        SDRP_SALT,
        SDRP_HYDROPHOBE,
        SDRP_NEIGHBORHOOD,
        SDRP_CORE,
    ]
    # Reorder columns for stable argmax priority
    prioritized = counts[:, order]
    winners = np.argmax(prioritized, axis=1)
    return np.asarray([order[int(w)] for w in winners], dtype=np.int64)


def mechanism_soft_targets(
    dehydron_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft mechanism strengths: dehydron nodes high pos, wrapped-core high neg."""
    d = dehydron_labels.astype(np.float32)
    pos = 0.35 + 0.55 * d
    neg = 0.45 * (1.0 - d) + 0.10 * d
    return pos, neg


def batch_from_records(
    records: Sequence[ResidueRecord],
    *,
    pdb_id: str,
    chain: str,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build harness batch dict from residue records."""
    graph = build_r0_r5_graph(records)
    coords = ca_coords(records)
    dehydron = dehydron_labels_from_edges(
        graph.num_nodes, graph.edge_index, graph.edge_type
    )
    sdrp = sdrp_heuristic_from_edges(
        graph.num_nodes, graph.edge_index, graph.edge_type
    )
    pos, neg = mechanism_soft_targets(dehydron)
    return _batch_from_arrays(
        coords=coords,
        edge_index=graph.edge_index,
        edge_type=graph.edge_type,
        dehydron=dehydron,
        sdrp=sdrp,
        pos=pos,
        neg=neg,
        num_nodes=graph.num_nodes,
        pdb_id=pdb_id,
        chain=chain,
        graph_meta=graph.meta,
        device=device,
    )


def _batch_from_arrays(
    *,
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    dehydron: np.ndarray,
    sdrp: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    num_nodes: int,
    pdb_id: str,
    chain: str,
    graph_meta: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    return {
        "x": torch.from_numpy(np.asarray(coords, dtype=np.float32)).to(device),
        "edge_index": torch.tensor(edge_index, dtype=torch.long, device=device),
        "edge_type": torch.tensor(edge_type, dtype=torch.long, device=device),
        "dehydron_labels": torch.from_numpy(dehydron.astype(np.float32)).to(device),
        "sdrp_target": torch.from_numpy(sdrp.astype(np.int64)).to(device),
        "mechanism_pos": torch.from_numpy(pos.astype(np.float32)).to(device),
        "mechanism_neg": torch.from_numpy(neg.astype(np.float32)).to(device),
        "num_nodes": int(num_nodes),
        "pdb_id": pdb_id.upper(),
        "chain": chain,
        "graph_meta": dict(graph_meta),
        "dehydron_frac": float(dehydron.mean()) if dehydron.size else 0.0,
    }


def _save_graph_cache(path: Path, batch: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return  # non-writable mount (e.g. docker cwd) — skip cache
    payload = {
        "x": batch["x"].detach().cpu(),
        "edge_index": batch["edge_index"].detach().cpu(),
        "edge_type": batch["edge_type"].detach().cpu(),
        "dehydron_labels": batch["dehydron_labels"].detach().cpu(),
        "sdrp_target": batch["sdrp_target"].detach().cpu(),
        "mechanism_pos": batch["mechanism_pos"].detach().cpu(),
        "mechanism_neg": batch["mechanism_neg"].detach().cpu(),
        "num_nodes": int(batch["num_nodes"]),
        "pdb_id": batch["pdb_id"],
        "chain": batch["chain"],
        "graph_meta": batch["graph_meta"],
        "dehydron_frac": float(batch["dehydron_frac"]),
        "cache_hash": graph_cache_hash(),
        "cache_version": BIOPHYS_CACHE_VERSION,
    }
    try:
        torch.save(payload, path)
    except OSError:
        return


def _load_graph_cache(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — corrupt cache → rebuild
        return None
    if payload.get("cache_hash") != graph_cache_hash():
        return None
    if payload.get("cache_version") != BIOPHYS_CACHE_VERSION:
        return None
    device = torch.device(device)
    dehydron = payload["dehydron_labels"]
    return {
        "x": payload["x"].to(device),
        "edge_index": payload["edge_index"].to(device),
        "edge_type": payload["edge_type"].to(device),
        "dehydron_labels": dehydron.to(device),
        "sdrp_target": payload["sdrp_target"].to(device),
        "mechanism_pos": payload["mechanism_pos"].to(device),
        "mechanism_neg": payload["mechanism_neg"].to(device),
        "num_nodes": int(payload["num_nodes"]),
        "pdb_id": str(payload["pdb_id"]).upper(),
        "chain": str(payload["chain"]),
        "graph_meta": dict(payload.get("graph_meta") or {}),
        "dehydron_frac": float(
            payload.get(
                "dehydron_frac",
                float(dehydron.float().mean()) if dehydron.numel() else 0.0,
            )
        ),
        "from_graph_cache": True,
    }


def load_structure_batch(
    pdb_id: str,
    chain: str,
    *,
    pdb_dir: Path | str = DEFAULT_PDB_DIR,
    device: torch.device | str = "cpu",
    use_graph_cache: bool = True,
    graph_cache_dir: Path | str = DEFAULT_GRAPH_CACHE_DIR,
) -> dict[str, Any]:
    path = ensure_pdb_cached(pdb_id, pdb_dir)
    cache_path = graph_cache_path(pdb_id, chain, cache_dir=graph_cache_dir)
    if use_graph_cache:
        cached = _load_graph_cache(cache_path, device=device)
        if cached is not None:
            return cached

    records = parse_residue_records_from_pdb_chain(path, chain)
    records = [r for r in records if r.get_atom("CA") is not None]
    if not records:
        raise ValueError(f"no CA-complete residues for {pdb_id}:{chain}")
    batch = batch_from_records(records, pdb_id=pdb_id, chain=chain, device=device)
    batch["from_graph_cache"] = False
    if use_graph_cache:
        _save_graph_cache(cache_path, batch)
    return batch


class TokyoEyeCuratedDataset(Dataset):
    """Folder-agnostic curated set: single PDB smoke or multi-structure manifest."""

    def __init__(
        self,
        pdb_code: str = DEFAULT_PDB_ID,
        chain: str = DEFAULT_CHAIN,
        manifest_path: str | Path | None = None,
        pdb_dir: str | Path = DEFAULT_PDB_DIR,
        *,
        use_graph_cache: bool = True,
        graph_cache_dir: str | Path = DEFAULT_GRAPH_CACHE_DIR,
    ) -> None:
        super().__init__()
        self.pdb_dir = Path(pdb_dir)
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.use_graph_cache = bool(use_graph_cache)
        self.graph_cache_dir = Path(graph_cache_dir)
        if manifest_path is not None and Path(manifest_path).is_file():
            self.entries = parse_enabled_manifest(manifest_path)
            self.mode = "manifest"
            self.manifest_path = str(manifest_path)
        else:
            self.entries = [
                {"pdb_id": pdb_code.strip().upper(), "chain": chain.strip()}
            ]
            self.mode = "single"
            self.manifest_path = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[int(idx) % len(self.entries)]
        return load_structure_batch(
            entry["pdb_id"],
            entry["chain"],
            pdb_dir=self.pdb_dir,
            device="cpu",
            use_graph_cache=self.use_graph_cache,
            graph_cache_dir=self.graph_cache_dir,
        )

    def get_on_device(self, idx: int, device: torch.device | str) -> dict[str, Any]:
        batch = self[idx]
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        return batch


__all__ = [
    "DEFAULT_CHAIN",
    "DEFAULT_GRAPH_CACHE_DIR",
    "DEFAULT_PDB_DIR",
    "DEFAULT_PDB_ID",
    "NUM_SDRP_HEURISTIC",
    "TokyoEyeCuratedDataset",
    "batch_from_records",
    "dehydron_labels_from_edges",
    "ensure_pdb_cached",
    "graph_cache_hash",
    "graph_cache_path",
    "load_structure_batch",
    "parse_enabled_manifest",
    "sdrp_heuristic_from_edges",
]
