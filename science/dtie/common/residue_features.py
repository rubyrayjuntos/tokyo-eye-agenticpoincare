"""Single source of truth for GNN per-residue input features (rho, tau, ss, sasa).

FeatureMode.TRAINING_CURRENT (pre-retrain parity gate):
  - rho: dehydron wrapping count (N–O midpoint, 6.5 Å)
  - tau_flag: 1.0 if rho < TAU else 0.0
  - ss_type: geometric Cα-angle encoding (0.0 / 0.5 / 1.0)
  - sasa: Cα-neighbor inverse proxy [0, 1]

FeatureMode.MASTER (retrain + DB repopulation target):
  - rho: same dehydron wrapping count
  - tau_flag: same TAU-referenced rule
  - sse_code: H/E/C via biotite annotate_sse (P-SEA on Cα trace; requires res_name)
  - ss_type: encode_sse_type_from_sse_code(H/E/C) → 0.0 / 0.5 / 1.0
  - sasa: FreeSASA per-residue Å², heavy atoms only, calcCoord + fixed VdW radii
    (matches experiments/diagnostics/evaluate_9est_prereg.py baseline spec)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

TAU = 13.0
WRAPPING_RADIUS = 6.5
SASA_PROXY_CUTOFF = 10.0

# FreeSASA master spec: heavy atoms only, fixed VdW radii (Å), calcCoord API.
FREESASA_ELEMENT_RADII: dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
}
FREESASA_DEFAULT_RADIUS = 1.70

# DSSP H/E/C → GNN ss_type scalar (legacy gnn_payload map).
SSE_CODE_TO_SS_TYPE: dict[str, float] = {"H": 0.0, "E": 0.5, "C": 1.0}

POLAR_SIDECHAINS = frozenset(
    {"ARG", "ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR", "TRP"}
)


class FeatureMode(str, Enum):
    """Active feature definitions for MASTER ingest / P_FEATURE_01 (four-vector persistence)."""

    TRAINING_CURRENT = "training_current"  # geometric SS + SASA proxy (pre-retrain gate)
    MASTER = "master"  # DSSP-class SSE + FreeSASA Å² (retrain target)


class GnnInputMode(str, Enum):
    """What the GNN node_emb sees — independent of DB MASTER persistence.

    TOPOLOGY_THREE_VECTOR (target): [ρ, τ, ss_type] only. SASA stays on dim_residue /
    data.sasa for binding-site accessibility, not hyperbolic depth.
    LEGACY_FOUR_VECTOR: production checkpoints through v6.0 — [ρ, τ, ss, sasa].
    """

    LEGACY_FOUR_VECTOR = "legacy_four_vector"
    TOPOLOGY_THREE_VECTOR = "topology_three_vector"


def resolve_gnn_input_mode(explicit: GnnInputMode | str | None = None) -> GnnInputMode:
    """Resolve GNN input layout from explicit arg or ``GNN_INPUT_MODE`` env."""
    import os

    if explicit is not None:
        if isinstance(explicit, GnnInputMode):
            return explicit
        return GnnInputMode(str(explicit).strip().lower())
    raw = os.environ.get("GNN_INPUT_MODE", GnnInputMode.LEGACY_FOUR_VECTOR.value).strip().lower()
    if raw in ("topology_three_vector", "topology", "three_vector", "3"):
        return GnnInputMode.TOPOLOGY_THREE_VECTOR
    return GnnInputMode.LEGACY_FOUR_VECTOR


def gnn_input_dim(mode: GnnInputMode | None = None) -> int:
    """Input width for ``GOSPConeMapperV6.node_emb``."""
    mode = resolve_gnn_input_mode(mode)
    return 3 if mode == GnnInputMode.TOPOLOGY_THREE_VECTOR else 4


def gnn_feature_set_id(mode: GnnInputMode | None = None) -> str:
    """MLflow / contract feature_set tag."""
    mode = resolve_gnn_input_mode(mode)
    return (
        "master_topology_three_vector"
        if mode == GnnInputMode.TOPOLOGY_THREE_VECTOR
        else "master_four_vector"
    )


def stack_gnn_node_features(
    rho: np.ndarray,
    tau_flag: np.ndarray,
    ss_type: np.ndarray,
    sasa: np.ndarray | None = None,
    *,
    mode: GnnInputMode | None = None,
) -> np.ndarray:
    """Build ``data.x`` for PyG — strips SASA when ``TOPOLOGY_THREE_VECTOR``."""
    mode = resolve_gnn_input_mode(mode)
    rho = np.asarray(rho, dtype=np.float64)
    tau_flag = np.asarray(tau_flag, dtype=np.float64)
    ss_type = np.asarray(ss_type, dtype=np.float64)
    if mode == GnnInputMode.TOPOLOGY_THREE_VECTOR:
        return np.stack([rho, tau_flag, ss_type], axis=1).astype(np.float32)
    if sasa is None:
        raise ValueError("sasa array required for GnnInputMode.LEGACY_FOUR_VECTOR")
    sasa = np.asarray(sasa, dtype=np.float64)
    return np.stack([rho, tau_flag, ss_type, sasa], axis=1).astype(np.float32)


def residue_sasa_from_data(data: Any) -> "torch.Tensor":
    """SASA side-channel for training probes — not part of topology ``data.x``."""
    import torch

    side = getattr(data, "sasa", None)
    if side is not None:
        if side.dim() == 1:
            return side.unsqueeze(-1)
        return side
    if data.x.size(1) > 3:
        return data.x[:, 3:4]
    raise ValueError(
        "SASA unavailable: set data.sasa or use GnnInputMode.LEGACY_FOUR_VECTOR"
    )


@dataclass(frozen=True)
class AtomRecord:
    atom_name: str
    element: str
    coord: np.ndarray  # shape (3,)
    parent_residue_name: str = ""


@dataclass(frozen=True)
class ResidueRecord:
    chain_label: str
    residue_index: int
    residue_name: str
    atoms: tuple[AtomRecord, ...]
    residue_id: str = ""

    def get_atom(self, name: str) -> AtomRecord | None:
        for atom in self.atoms:
            if atom.atom_name == name:
                return atom
        return None


@dataclass(frozen=True)
class ResidueNodeFeatures:
    chain_label: str
    residue_index: int
    rho: float
    tau_flag: float
    ss_type: float
    sasa: float
    residue_id: str = ""
    sse_code: str = ""  # H/E/C populated in MASTER mode for dim_residue persistence

    def as_vector(self) -> np.ndarray:
        return np.array([self.rho, self.tau_flag, self.ss_type, self.sasa], dtype=np.float64)


@dataclass(frozen=True)
class FeatureParityMismatch:
    residue_key: str
    feature: str
    training_value: Any
    inference_value: Any
    rule: str

    def __str__(self) -> str:
        return (
            f"{self.residue_key} {self.feature}: "
            f"training={self.training_value!r} inference={self.inference_value!r} ({self.rule})"
        )


def residue_key(chain: str, index: int) -> str:
    return f"{chain}:{index}"


def compute_dehydron_wrapping_count(
    residue: ResidueRecord,
    all_atoms: Sequence[AtomRecord],
    *,
    wrapping_radius: float = WRAPPING_RADIUS,
) -> float:
    """Dehydron wrapping count — integer semantics, returned as float."""
    n_atom = residue.get_atom("N")
    o_atom = residue.get_atom("O")
    if n_atom is None or o_atom is None:
        return -1.0
    mid = (n_atom.coord + o_atom.coord) / 2.0
    count = 0
    for atom in all_atoms:
        if atom.element != "C":
            continue
        parent = atom.parent_residue_name.strip().upper()
        if parent in POLAR_SIDECHAINS:
            continue
        if atom.atom_name == "C":
            continue
        dist = float(np.linalg.norm(atom.coord - mid))
        if dist >= wrapping_radius:
            continue
        count += 1
    return float(count)


def compute_tau_flag(rho: float, *, tau: float = TAU) -> float:
    if rho < 0:
        return 0.0
    return 1.0 if rho < tau else 0.0


def compute_ss_geometric(ca_coords: np.ndarray) -> np.ndarray:
    """Geometric SS scalar: 0.0 helix-like, 0.5 sheet-like, 1.0 coil default."""
    n = len(ca_coords)
    ss = np.ones(n, dtype=np.float64)
    for i in range(2, n - 2):
        v1 = ca_coords[i] - ca_coords[i - 2]
        v2 = ca_coords[i + 2] - ca_coords[i]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        if d1 < 1e-6 or d2 < 1e-6:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
        if cos_angle < 0.5 and d1 < 7.0:
            ss[i] = 0.0
        elif cos_angle > 0.8:
            ss[i] = 0.5
    return ss


def compute_sasa_proxy(
    ca_coords: np.ndarray,
    *,
    cutoff: float = SASA_PROXY_CUTOFF,
) -> np.ndarray:
    """Training SASA proxy: 1 - normalized Cα neighbor count within cutoff."""
    from scipy.spatial.distance import cdist

    dists = cdist(ca_coords, ca_coords)
    neighbor_counts = ((dists < cutoff) & (dists > 0.1)).sum(axis=1).astype(np.float64)
    max_count = neighbor_counts.max()
    if max_count > 0:
        return 1.0 - (neighbor_counts / max_count)
    return np.full(len(ca_coords), 0.5, dtype=np.float64)


def encode_ss_type_from_sse_code(sse_code: str | None) -> float:
    """Map DSSP H/E/C code to GNN ss_type scalar."""
    key = (sse_code or "C").strip().upper()
    return SSE_CODE_TO_SS_TYPE.get(key, 1.0)


def sse_code_from_ss_type(ss_type: float) -> str:
    """Inverse of encode_ss_type_from_sse_code for parity / distribution checks."""
    for code, val in SSE_CODE_TO_SS_TYPE.items():
        if float(ss_type) == float(val):
            return code
    return "C"


def _residues_to_biotite_atom_array(residues: Sequence[ResidueRecord]):
    """Build biotite AtomArray from governed residue atoms (heavy atoms only).

    annotate_sse requires res_name (and standard hetero/ins_code fields) on each
    atom row; without res_name every residue annotates as blank → coil.
    """
    from biotite.structure import AtomArray

    entries: list[tuple[str, str, np.ndarray, int, str, str]] = []
    for res in residues:
        res_name = (res.residue_name or "UNK").strip().upper()[:3]
        for atom in res.atoms:
            element = (atom.element or "C").upper()
            if element == "H" or atom.atom_name.upper().startswith("H"):
                continue
            entries.append(
                (
                    atom.atom_name,
                    element,
                    atom.coord,
                    res.residue_index,
                    res.chain_label,
                    res_name,
                )
            )

    if not entries:
        return AtomArray(0)

    array = AtomArray(len(entries))
    array.coord = np.array([e[2] for e in entries], dtype=np.float32)
    array.atom_name = np.array([e[0] for e in entries])
    array.element = np.array([e[1] for e in entries])
    array.res_id = np.array([e[3] for e in entries], dtype=np.int32)
    array.chain_id = np.array([e[4] for e in entries])
    array.res_name = np.array([e[5] for e in entries])
    array.ins_code = np.array([""] * len(entries))
    array.hetero = np.zeros(len(entries), dtype=bool)
    return array


def _biotite_sse_to_hec(annotation: str) -> str:
    if annotation == "a":
        return "H"
    if annotation == "b":
        return "E"
    return "C"


def compute_sse_dssp_codes(residues: Sequence[ResidueRecord]) -> list[str]:
    """Secondary structure H/E/C via biotite annotate_sse (P-SEA, Cα-based)."""
    from biotite.structure import annotate_sse, get_residue_starts

    if not residues:
        return []

    array = _residues_to_biotite_atom_array(residues)
    if len(array) == 0:
        return ["C"] * len(residues)

    sse_by_residue = annotate_sse(array)
    starts = get_residue_starts(array)
    if len(starts) != len(residues) or len(sse_by_residue) != len(residues):
        raise ValueError(
            f"SSE residue groups ({len(starts)}) != input residues ({len(residues)}); "
            "atom array grouping does not match ResidueRecord list"
        )

    codes: list[str] = []
    for i, start in enumerate(starts):
        if int(array.res_id[start]) != residues[i].residue_index:
            raise ValueError(
                f"SSE residue order mismatch at index {i}: "
                f"atom array res_id={array.res_id[start]} vs record={residues[i].residue_index}"
            )
        codes.append(_biotite_sse_to_hec(str(sse_by_residue[i])))
    return codes


def sse_code_counts(codes: Sequence[str]) -> dict[str, int]:
    """Count H/E/C sse_code assignments."""
    counts = {"H": 0, "E": 0, "C": 0}
    for code in codes:
        key = (code or "C").strip().upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def assert_sse_informative(
    codes: Sequence[str],
    *,
    min_helix: int = 1,
    min_sheet: int = 1,
    context: str = "",
) -> None:
    """Raise if secondary structure collapsed to a constant (all coil)."""
    counts = sse_code_counts(codes)
    prefix = f"{context}: " if context else ""
    if counts["H"] < min_helix or counts["E"] < min_sheet:
        raise AssertionError(
            f"{prefix}SSE annotation non-informative H={counts['H']} E={counts['E']} C={counts['C']} "
            f"(expected at least {min_helix} helix and {min_sheet} sheet)"
        )


def compute_sasa_freesasa(residues: Sequence[ResidueRecord]) -> np.ndarray:
    """Per-residue SASA in Å² — FreeSASA calcCoord, heavy atoms only."""
    import freesasa

    if not residues:
        return np.array([], dtype=np.float64)

    flat_coords: list[float] = []
    radii: list[float] = []
    atom_to_res_idx: list[int] = []

    for res_i, res in enumerate(residues):
        for atom in res.atoms:
            element = (atom.element or "C").upper()
            if element == "H" or atom.atom_name.upper().startswith("H"):
                continue
            flat_coords.extend(float(c) for c in atom.coord)
            radii.append(FREESASA_ELEMENT_RADII.get(element, FREESASA_DEFAULT_RADIUS))
            atom_to_res_idx.append(res_i)

    per_res = np.zeros(len(residues), dtype=np.float64)
    if not flat_coords:
        return per_res

    result = freesasa.calcCoord(flat_coords, radii)
    for atom_i, res_i in enumerate(atom_to_res_idx):
        per_res[res_i] += float(result.atomArea(atom_i))
    return per_res


def compute_burial_rho_skew(ca_coords: np.ndarray, *, radius: float = 10.0) -> np.ndarray:
    """Legacy GraphBuilder skew — for P_FEATURE_01 negative tests only."""
    n = len(ca_coords)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dists = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        counts[i] = float(np.sum((dists < radius) & (dists > 0.1)))
    return counts


def build_node_features(
    residues: Sequence[ResidueRecord],
    *,
    mode: FeatureMode = FeatureMode.TRAINING_CURRENT,
) -> list[ResidueNodeFeatures]:
    valid: list[ResidueRecord] = []
    for res in residues:
        rho = compute_dehydron_wrapping_count(res, _flatten_atoms(residues))
        if rho < 0 or res.get_atom("CA") is None:
            continue
        valid.append(res)

    if len(valid) < 1:
        return []

    all_atoms = _flatten_atoms(valid)

    if mode == FeatureMode.TRAINING_CURRENT:
        ca_coords = np.array([res.get_atom("CA").coord for res in valid], dtype=np.float64)
        ss_arr = compute_ss_geometric(ca_coords)
        sasa_arr = compute_sasa_proxy(ca_coords)
        sse_codes = [""] * len(valid)
    elif mode == FeatureMode.MASTER:
        sse_codes = compute_sse_dssp_codes(valid)
        ss_arr = np.array([encode_ss_type_from_sse_code(c) for c in sse_codes], dtype=np.float64)
        sasa_arr = compute_sasa_freesasa(valid)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    out: list[ResidueNodeFeatures] = []
    for i, res in enumerate(valid):
        rho = compute_dehydron_wrapping_count(res, all_atoms)
        out.append(
            ResidueNodeFeatures(
                chain_label=res.chain_label,
                residue_index=res.residue_index,
                residue_id=res.residue_id,
                rho=rho,
                tau_flag=compute_tau_flag(rho),
                ss_type=float(ss_arr[i]),
                sasa=float(sasa_arr[i]),
                sse_code=sse_codes[i],
            )
        )
    return out


def build_from_pdb_chain(
    pdb_path: str | Any,
    chain_id: str,
    *,
    mode: FeatureMode = FeatureMode.TRAINING_CURRENT,
) -> list[ResidueNodeFeatures]:
    """Parse a PDB chain and compute node features (inference/training SSOT path)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("feat", str(pdb_path))
    residues_raw = [
        r
        for r in structure.get_residues()
        if r.get_id()[0] == " " and r.parent.id == chain_id
    ]
    records: list[ResidueRecord] = []
    for res in residues_raw:
        res_name = res.get_resname().strip().upper()
        atoms = tuple(
            AtomRecord(
                atom_name=atom.name,
                element=(atom.element or "").upper(),
                coord=np.asarray(atom.coord, dtype=np.float64),
                parent_residue_name=res_name,
            )
            for atom in res.get_atoms()
        )
        records.append(
            ResidueRecord(
                chain_label=chain_id,
                residue_index=int(res.get_id()[1]),
                residue_name=res_name,
                atoms=atoms,
            )
        )
    return build_node_features(records, mode=mode)


@dataclass(frozen=True)
class ResidueFeaturePersistRow:
    """Values to write into dim_residue after MASTER ingest."""

    residue_id: str
    chain_label: str
    residue_index: int
    sse_code: str
    sasa: float


def master_persist_rows(features: Sequence[ResidueNodeFeatures]) -> list[ResidueFeaturePersistRow]:
    """Extract DB persistence rows from MASTER-mode features."""
    rows: list[ResidueFeaturePersistRow] = []
    for feat in features:
        if not feat.sse_code:
            raise ValueError(
                f"residue {feat.chain_label}:{feat.residue_index} missing sse_code in MASTER mode"
            )
        rows.append(
            ResidueFeaturePersistRow(
                residue_id=feat.residue_id,
                chain_label=feat.chain_label,
                residue_index=feat.residue_index,
                sse_code=feat.sse_code,
                sasa=feat.sasa,
            )
        )
    return rows


def compare_feature_parity(
    training: Sequence[ResidueNodeFeatures],
    inference: Sequence[ResidueNodeFeatures],
    *,
    sasa_rtol: float = 1e-5,
    sasa_atol: float = 1e-6,
    ss_rtol: float = 0.0,
    ss_atol: float = 0.0,
) -> list[FeatureParityMismatch]:
    """Compare two feature lists keyed by chain:residue_index."""
    train_map = {residue_key(f.chain_label, f.residue_index): f for f in training}
    infer_map = {residue_key(f.chain_label, f.residue_index): f for f in inference}
    keys = sorted(set(train_map) & set(infer_map))
    mismatches: list[FeatureParityMismatch] = []

    for key in keys:
        t = train_map[key]
        inf = infer_map[key]

        if int(round(t.rho)) != int(round(inf.rho)):
            mismatches.append(
                FeatureParityMismatch(key, "rho", int(round(t.rho)), int(round(inf.rho)), "exact-int")
            )

        if float(t.tau_flag) != float(inf.tau_flag):
            mismatches.append(
                FeatureParityMismatch(key, "tau_flag", t.tau_flag, inf.tau_flag, "exact-bool")
            )

        if not np.isclose(t.ss_type, inf.ss_type, rtol=ss_rtol, atol=ss_atol):
            mismatches.append(
                FeatureParityMismatch(key, "ss_type", t.ss_type, inf.ss_type, "exact-categorical")
            )

        if not np.isclose(t.sasa, inf.sasa, rtol=sasa_rtol, atol=sasa_atol):
            mismatches.append(
                FeatureParityMismatch(
                    key,
                    "sasa",
                    t.sasa,
                    inf.sasa,
                    "same-algorithm-tolerance",
                )
            )

    if len(train_map) != len(infer_map):
        missing = set(train_map) - set(infer_map)
        extra = set(infer_map) - set(train_map)
        if missing:
            mismatches.append(
                FeatureParityMismatch("?", "residue_set", sorted(missing), None, "missing-in-inference")
            )
        if extra:
            mismatches.append(
                FeatureParityMismatch("?", "residue_set", None, sorted(extra), "extra-in-inference")
            )

    return mismatches


def assert_feature_parity(
    training: Sequence[ResidueNodeFeatures],
    inference: Sequence[ResidueNodeFeatures],
) -> None:
    mismatches = compare_feature_parity(training, inference)
    if mismatches:
        lines = "\n".join(f"  - {m}" for m in mismatches[:20])
        extra = len(mismatches) - 20
        suffix = f"\n  ... and {extra} more" if extra > 0 else ""
        raise AssertionError(f"P_FEATURE_01 parity failed ({len(mismatches)} mismatches):\n{lines}{suffix}")


def records_from_db_atom_rows(rows: Sequence[dict[str, Any]]) -> list[ResidueRecord]:
    """Group flat dim_atom query rows into ResidueRecord list."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row["residue_id"])
        if rid not in grouped:
            res_name = (
                row.get("residue_name_3")
                or row.get("residue_name")
                or row.get("res_name")
                or "UNK"
            )
            grouped[rid] = {
                "residue_id": rid,
                "residue_index": int(row["residue_index"]),
                "chain_label": str(row["chain_label"]),
                "residue_name": str(res_name).strip().upper(),
                "atoms": [],
            }
        if row.get("atom_name") is None or row.get("x") is None:
            continue
        parent = grouped[rid]["residue_name"]
        grouped[rid]["atoms"].append(
            AtomRecord(
                atom_name=str(row["atom_name"]),
                element=str(row.get("element") or "").upper(),
                coord=np.array([row["x"], row["y"], row["z"]], dtype=np.float64),
                parent_residue_name=parent,
            )
        )

    records: list[ResidueRecord] = []
    for payload in grouped.values():
        records.append(
            ResidueRecord(
                residue_id=payload["residue_id"],
                chain_label=payload["chain_label"],
                residue_index=payload["residue_index"],
                residue_name=payload["residue_name"],
                atoms=tuple(payload["atoms"]),
            )
        )
    records.sort(key=lambda r: (r.chain_label, r.residue_index))
    return records


def inject_burial_rho_skew(
    features: Sequence[ResidueNodeFeatures],
    ca_coords: np.ndarray,
) -> list[ResidueNodeFeatures]:
    """Deliberate parity break for P_FEATURE_01 negative tests."""
    burial = compute_burial_rho_skew(ca_coords)
    skewed: list[ResidueNodeFeatures] = []
    for i, feat in enumerate(features):
        skewed.append(
            ResidueNodeFeatures(
                chain_label=feat.chain_label,
                residue_index=feat.residue_index,
                residue_id=feat.residue_id,
                rho=float(burial[i]),
                tau_flag=feat.tau_flag,
                ss_type=feat.ss_type,
                sasa=feat.sasa,
            )
        )
    return skewed


def _flatten_atoms(residues: Iterable[ResidueRecord]) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    for res in residues:
        atoms.extend(res.atoms)
    return atoms
