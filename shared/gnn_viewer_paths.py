"""Torch-free paths for GNN interactive HTML viewer (agent + science)."""

from __future__ import annotations

import os
import re
from pathlib import Path

_STRUCTURE_ID_RE = re.compile(r"^[a-z0-9:_-]+$", re.IGNORECASE)


def viewer_output_dir() -> Path:
    root = Path(
        os.getenv("GNN_VIEWER_OUTPUT_DIR", "./data/local_objects/gnn_viewer")
    ).resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return root


def interactive_viewer_enabled() -> bool:
    return os.getenv("GNN_INTERACTIVE_HTML", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def resolve_viewer_html_path(structure_id: str) -> Path | None:
    """Return interactive HTML for a structure, if present."""
    sid = structure_id.strip().lower()
    if not _STRUCTURE_ID_RE.match(sid):
        return None
    path = viewer_output_dir() / sid / f"{sid}_interactive.html"
    return path if path.is_file() else None


def resolve_disc_html_path(structure_id: str) -> Path | None:
    """Return Poincaré disc interactive HTML for a structure, if present."""
    sid = structure_id.strip().lower()
    if not _STRUCTURE_ID_RE.match(sid):
        return None
    path = viewer_output_dir() / sid / f"{sid}_poincare_disc.html"
    return path if path.is_file() else None


def structure_viewer_paths(structure_id: str) -> dict[str, Path]:
    """Return on-disk viewer artifacts for a structure (empty if missing)."""
    sid = structure_id.strip().lower()
    root = viewer_output_dir() / sid
    out: dict[str, Path] = {}
    for key, name in (
        ("structure", f"{sid}_interactive.html"),
        ("disc", f"{sid}_poincare_disc.html"),
        ("pdb", f"{sid}_gosp_native.pdb"),
    ):
        path = root / name
        if path.is_file():
            out[key] = path
    return out
