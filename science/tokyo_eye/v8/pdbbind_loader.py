"""PDB-Bind loader helpers (Sprint 10 + 10.1 ligand/R6 re-exports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from science.tokyo_eye.v8.ligand_interface import (
    LIGAND_FEAT_DIM,
    R6_DISTANCE_A,
    LigandAtoms,
    build_r6_edges,
    extract_ligand_hetatm,
    ligand_feature_matrix,
    residue_proxy_coords,
    unique_r6_contacts,
)

DEFAULT_LP_CSV = Path("data/pdbbind/LP_PDBBind_refined_core.csv")


def load_lp_table(csv_path: Path | str = DEFAULT_LP_CSV) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.lower()
    return df


def refined_core_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    refined = df[df["category"].astype(str).str.lower() == "refined"].copy()
    core = df[df["category"].astype(str).str.lower() == "core"].copy()
    if refined.empty or core.empty:
        raise ValueError("CSV must contain refined and core categories")
    return refined, core


def entries_from_split_manifest(
    manifest_path: Path | str,
) -> dict[str, list[dict[str, Any]]]:
    path = Path(manifest_path)
    data = json.loads(path.read_text())
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ("train", "val", "core_test"):
        rows = data.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"manifest missing list for {key}")
        out[key] = rows
    return out


__all__ = [
    "DEFAULT_LP_CSV",
    "LIGAND_FEAT_DIM",
    "LigandAtoms",
    "R6_DISTANCE_A",
    "build_r6_edges",
    "entries_from_split_manifest",
    "extract_ligand_hetatm",
    "ligand_feature_matrix",
    "load_lp_table",
    "refined_core_frames",
    "residue_proxy_coords",
    "unique_r6_contacts",
]
