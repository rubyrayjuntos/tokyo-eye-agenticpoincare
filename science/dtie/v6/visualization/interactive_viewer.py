"""GNN interactive 3D viewer — enhanced PDB + standalone NGL HTML.

Restores the v2 autonomous viewer pattern for v6 ingest:
  per-residue epistemic → B-factor, cone_depth → occupancy, expert → segID
  cartoon + semi-transparent surface colored by uncertainty (PPI / outer-shell read).

Triggered after ``gnn_inference`` (non-fatal). Files land under
``GNN_VIEWER_OUTPUT_DIR`` (default ``data/local_objects/gnn_viewer``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.common.normalizer_payloads import ProvenanceContext, RunType, SourceType

logger = logging.getLogger(__name__)

from shared.gnn_viewer_paths import interactive_viewer_enabled, viewer_output_dir

_ASSET_TYPE = "gnn_interactive_view"

# dim_residue stores one-letter codes; NGL cartoon needs standard 3-letter names.
_AA1_TO_AA3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "U": "SEC",
    "O": "PYL",
    "X": "UNK",
}

_ATOM_SORT_KEY: dict[str, int] = {
    "N": 0,
    "CA": 1,
    "C": 2,
    "O": 3,
    "OXT": 4,
    "CB": 5,
}


def _residue_name_3letter(name: str | None) -> str:
    raw = (name or "UNK").strip().upper()
    if len(raw) == 3 and raw.isalpha():
        return raw
    if len(raw) == 1:
        return _AA1_TO_AA3.get(raw, "UNK")
    return raw[:3].ljust(3, "X")[:3]


def _format_pdb_atom_name(atom_name: str) -> str:
    """PDB cols 13-16 — locant names (CG1) right-align; elements (CA, N) lead with space."""
    name = atom_name.strip().upper()
    if not name:
        return "    "
    if len(name) == 4:
        return name[:4]
    if len(name) >= 2 and any(ch.isdigit() for ch in name[1:]):
        return f"{name:<4}"[:4]
    return f" {name:<3}"[:4]


def _atom_sort_key(atom_name: str) -> tuple[int, str]:
    key = atom_name.strip().upper()
    return (_ATOM_SORT_KEY.get(key, 99), key)


_STANDARD_AA3 = set(_AA1_TO_AA3.values())


def _is_polymer_residue(residue_name: str, atom_names: set[str]) -> bool:
    """True when residue can participate in a continuous cartoon trace."""
    return residue_name in _STANDARD_AA3 and {"N", "CA", "C"}.issubset(atom_names)


def build_residue_channel_lookup(
    nodes: list[GNNNodeOutput],
) -> dict[tuple[str, int], GNNNodeOutput]:
    """Map (chain_label, residue_index) → GNN node output."""
    lookup: dict[tuple[str, int], GNNNodeOutput] = {}
    for node in nodes:
        lookup[(node.chain_label, int(node.residue_index))] = node
    return lookup


def _scale_channel(values: np.ndarray, *, out_min: float, out_max: float) -> np.ndarray:
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return np.full_like(values, (out_min + out_max) / 2.0, dtype=np.float64)
    return out_min + (values - vmin) * (out_max - out_min) / (vmax - vmin)


def _expert_seg(node: GNNNodeOutput) -> str:
    weights = node.expert_weights
    if weights is None or len(weights) == 0:
        return "E0"
    idx = int(np.argmax(weights))
    return f"E{idx}"


async def write_annotated_pdb(
    structure_id: str,
    db: Any,
    node_lookup: dict[tuple[str, int], GNNNodeOutput],
    output_path: Path,
) -> int:
    """Write full-atom PDB with GNN channels in B-factor / occupancy / segID.

    Returns atom count written.
    """
    rows = await db.fetch_all(
        """
        SELECT a.atom_name, a.element, a.x, a.y, a.z,
               r.residue_index, r.residue_name, c.chain_label
        FROM dim_atom a
        JOIN dim_residue r ON r.residue_id = a.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        ORDER BY c.chain_label, r.residue_index, a.atom_name
        """,
        {"structure_id": structure_id.strip().lower()},
    )
    if not rows:
        return 0

    rows = sorted(
        rows,
        key=lambda row: (
            row["chain_label"] or "A",
            int(row["residue_index"] or 0),
            _atom_sort_key(row["atom_name"] or ""),
        ),
    )

    keys = list(node_lookup.keys())
    epistemic = np.array([node_lookup[k].epistemic_uncertainty for k in keys], dtype=np.float64)
    depth = np.array([node_lookup[k].cone_depth for k in keys], dtype=np.float64)
    epist_scaled = _scale_channel(epistemic, out_min=0.0, out_max=99.0)
    depth_norm = _scale_channel(depth, out_min=0.0, out_max=1.0)
    epist_map = {key: float(epist_scaled[i]) for i, key in enumerate(keys)}
    depth_map = {key: float(depth_norm[i]) for i, key in enumerate(keys)}

    # Classify residues for ATOM (polymer) vs HETATM (ligands/water/incomplete backbone).
    residue_atoms: dict[tuple[str, int], set[str]] = {}
    residue_names: dict[tuple[str, int], str] = {}
    for row in rows:
        chain = (row["chain_label"] or "A").strip()
        res_index = int(row["residue_index"] or 1)
        key = (chain, res_index)
        residue_atoms.setdefault(key, set()).add((row["atom_name"] or "").strip().upper())
        residue_names[key] = _residue_name_3letter(row["residue_name"])

    polymer_keys = {
        key
        for key, atoms in residue_atoms.items()
        if _is_polymer_residue(residue_names[key], atoms)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atom_serial = 0
    prev_chain: str | None = None
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            chain = (row["chain_label"] or "A").strip()
            res_index = int(row["residue_index"] or 1)
            key = (chain, res_index)

            if prev_chain is not None and chain != prev_chain:
                fh.write("TER\n")
            prev_chain = chain

            node = node_lookup.get(key) or node_lookup.get((chain[:1], res_index))
            b_factor = epist_map.get(key, 50.0) if node else 50.0
            occupancy = max(0.35, depth_map.get(key, 0.5) if node else 0.5)

            residue_name = residue_names[key]
            chain_col = chain[0] if chain else "A"
            element = (row["element"] or "C").strip()[:2].upper() or "C"
            atom_name = row["atom_name"] or "X"
            record = "ATOM  " if key in polymer_keys else "HETATM"
            atom_serial += 1

            fh.write(
                f"{record}{atom_serial:5d} {_format_pdb_atom_name(atom_name)} "
                f"{residue_name:>3s} {chain_col:1s}{res_index:4d}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"{occupancy:6.2f}{b_factor:6.2f}          "
                f"{element:>2s}\n"
            )
        fh.write("END\n")
    return atom_serial


def write_offline_annotated_pdb(
    chain_pdb_path: Path,
    node_lookup: dict[tuple[str, int], GNNNodeOutput],
    output_path: Path,
    *,
    chain: str = "A",
) -> int:
    """Annotate a chain-extracted PDB with per-residue ν_epi (B-factor) and depth (occupancy)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(chain_pdb_path.stem, str(chain_pdb_path))
    rows: list[dict[str, Any]] = []
    for res in structure.get_residues():
        if res.get_id()[0] != " ":
            continue
        chain_label = res.get_parent().id.strip() or chain
        res_index = int(res.get_id()[1])
        for atom in res.get_atoms():
            rows.append(
                {
                    "atom_name": atom.get_name(),
                    "element": atom.element,
                    "x": atom.get_coord()[0],
                    "y": atom.get_coord()[1],
                    "z": atom.get_coord()[2],
                    "residue_index": res_index,
                    "residue_name": res.get_resname(),
                    "chain_label": chain_label,
                }
            )
    if not rows:
        return 0

    rows = sorted(
        rows,
        key=lambda row: (
            row["chain_label"] or "A",
            int(row["residue_index"] or 0),
            _atom_sort_key(row["atom_name"] or ""),
        ),
    )

    keys = list(node_lookup.keys())
    epistemic = np.array([node_lookup[k].epistemic_uncertainty for k in keys], dtype=np.float64)
    depth = np.array([node_lookup[k].cone_depth for k in keys], dtype=np.float64)
    epist_scaled = _scale_channel(epistemic, out_min=0.0, out_max=99.0)
    depth_norm = _scale_channel(depth, out_min=0.0, out_max=1.0)
    epist_map = {key: float(epist_scaled[i]) for i, key in enumerate(keys)}
    depth_map = {key: float(depth_norm[i]) for i, key in enumerate(keys)}

    residue_atoms: dict[tuple[str, int], set[str]] = {}
    residue_names: dict[tuple[str, int], str] = {}
    for row in rows:
        chain_label = (row["chain_label"] or "A").strip()
        res_index = int(row["residue_index"] or 1)
        key = (chain_label, res_index)
        residue_atoms.setdefault(key, set()).add((row["atom_name"] or "").strip().upper())
        residue_names[key] = _residue_name_3letter(row["residue_name"])

    polymer_keys = {
        key
        for key, atoms in residue_atoms.items()
        if _is_polymer_residue(residue_names[key], atoms)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atom_serial = 0
    prev_chain: str | None = None
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            chain_label = (row["chain_label"] or "A").strip()
            res_index = int(row["residue_index"] or 1)
            key = (chain_label, res_index)

            if prev_chain is not None and chain_label != prev_chain:
                fh.write("TER\n")
            prev_chain = chain_label

            node = node_lookup.get(key) or node_lookup.get((chain_label[:1], res_index))
            b_factor = epist_map.get(key, 50.0) if node else 50.0
            occupancy = max(0.35, depth_map.get(key, 0.5) if node else 0.5)

            residue_name = residue_names[key]
            chain_col = chain_label[0] if chain_label else "A"
            element = (row["element"] or "C").strip()[:2].upper() or "C"
            atom_name = row["atom_name"] or "X"
            record = "ATOM  " if key in polymer_keys else "HETATM"
            atom_serial += 1

            fh.write(
                f"{record}{atom_serial:5d} {_format_pdb_atom_name(atom_name)} "
                f"{residue_name:>3s} {chain_col:1s}{res_index:4d}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"{occupancy:6.2f}{b_factor:6.2f}          "
                f"{element:>2s}\n"
            )
        fh.write("END\n")
    return atom_serial


def _pharmacophore_ngl_selection(
    sites: list[dict[str, str | int]],
    *,
    chain: str = "A",
) -> str:
    parts = [f"{int(site['resnum'])}:{chain}" for site in sites if site.get("resnum")]
    return " or ".join(parts) if parts else ""


def write_interactive_html(
    *,
    structure_id: str,
    pdb_text: str,
    model_version: str,
    output_path: Path,
    checkpoint_path: str | None = None,
    pharmacophore_sites: list[dict[str, str | int]] | None = None,
    legend_mode: str = "default",
    chain: str = "A",
) -> None:
    """Write standalone NGL viewer HTML (6LDH-style cartoon + uncertainty surface)."""
    sid = structure_id.strip().lower()
    pdb_json = json.dumps(pdb_text)
    title = f"{sid.upper()} GOSP-Native — Uncertainty Viewer"
    checkpoint_note = (
        f" &nbsp;|&nbsp; checkpoint: {Path(checkpoint_path).name}"
        if checkpoint_path
        else ""
    )

    if legend_mode == "possibility_b" and pharmacophore_sites:
        site_labels = ", ".join(
            f"{site['label']} ({site['resnum']})" for site in pharmacophore_sites
        )
        legend_html = f"""
    <b>{sid.upper()}</b> &nbsp;|&nbsp; {model_version}{checkpoint_note}<br/>
    <b>Surface / cartoon coloring:</b> residue-level ν<sub>epi</sub> inherited to all atoms
    (Possibility B — not independent per-shell-point uncertainty).
    <span class="hi">Red</span> = higher ν<sub>epi</sub> &rarr;
    <span class="lo">blue</span> = lower.<br/>
    <b><span class="site">Green spheres</span>:</b> GOSP-validated pharmacophore sites
    ({site_labels}) — identified by structural physics, not by the GNN.<br/>
    <b>Claim:</b> validated sites sit in the transitional uncertainty band
    (neither buried-blue nor exposed-red); the model characterizes their wrapping
    environment, it did not discover the sites.
"""
        pharma_sele = _pharmacophore_ngl_selection(pharmacophore_sites, chain=chain)
        pharma_js = f"""
        comp.addRepresentation("ball+stick", {{
          sele: "{pharma_sele}",
          colorValue: 0x2ecc71,
          radiusScale: 2.5,
          aspectRatio: 1.5,
          opacity: 1.0
        }});"""
    else:
        legend_html = f"""
    <b>{sid.upper()}</b> &nbsp;|&nbsp;
    {model_version}{checkpoint_note} &nbsp;|&nbsp;
    B-factor = Epistemic uncertainty
    (<span class="hi">red = high gap / training uncertainty</span>
     &rarr; <span class="lo">blue = well-trained region</span>)
    &nbsp;|&nbsp; Occupancy = Cone depth &nbsp;|&nbsp;
    SegID = Expert assignment (E0-E3)
"""
        pharma_js = ""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://unpkg.com/ngl@2.1.0/dist/ngl.js"></script>
  <style>
    body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; }}
    #viewport {{ width:100vw; height:82vh; }}
    #legend {{ padding:8px 14px; font-size:12px; line-height:1.7; background:#1a1a1a; }}
    span.hi {{ color:#f77; }} span.lo {{ color:#77f; }} span.site {{ color:#2ecc71; }}
  </style>
</head>
<body>
  <div id="viewport"></div>
  <div id="legend">{legend_html}
  </div>
  <script>
    var PDB_DATA = {pdb_json};

    document.addEventListener("DOMContentLoaded", function() {{
      var stage = new NGL.Stage("viewport", {{backgroundColor: "#111111"}});
      var blob = new Blob([PDB_DATA], {{type: "text/plain"}});
      stage.loadFile(blob, {{ext: "pdb", defaultRepresentation: false}}).then(function(comp) {{
        var colorOpts = {{
          colorScheme: "bfactor",
          colorScale: "RdYlBu",
          colorReverse: true
        }};
        comp.addRepresentation("surface", Object.assign({{
          sele: "polymer",
          opacity: 0.18,
          side: "front",
          smooth: 2
        }}, colorOpts));
        comp.addRepresentation("cartoon", Object.assign({{
          sele: "polymer",
          opacity: 1.0,
          smoothSheet: true,
          quality: "high"
        }}, colorOpts));
        comp.addRepresentation("ball+stick", Object.assign({{
          sele: "hetero or water or ion",
          opacity: 0.7
        }}, colorOpts));{pharma_js}
        comp.autoView();
      }});

      document.addEventListener("contextmenu", function(e) {{
        e.preventDefault();
        stage.autoView();
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _asset_id(structure_id: str, run_id: str) -> str:
    digest = hashlib.sha256(f"{structure_id}:{run_id}:gnn_interactive_view".encode()).hexdigest()
    return f"gnn_viewer_{digest[:24]}"


async def generate_gnn_interactive_viewer(
    db: Any,
    *,
    gnn_result: GNNInferenceResult,
    run_id: str,
    caller_identity: str = "gnn_interactive_viewer",
    pipeline_name: str = "gnn_inference",
    code_version: str | None = None,
) -> dict[str, Any]:
    """Build enhanced PDB + interactive HTML and register governed asset."""
    if not gnn_result.nodes:
        raise ValueError("GNN result has no nodes")

    structure_id = gnn_result.structure_id.strip().lower()
    out_dir = viewer_output_dir() / structure_id
    pdb_path = out_dir / f"{structure_id}_gosp_native.pdb"
    html_path = out_dir / f"{structure_id}_interactive.html"

    node_lookup = build_residue_channel_lookup(gnn_result.nodes)
    n_atoms = await write_annotated_pdb(structure_id, db, node_lookup, pdb_path)
    if n_atoms == 0:
        raise ValueError(f"No atoms in dim_atom for structure {structure_id}")

    pdb_text = pdb_path.read_text(encoding="utf-8")
    write_interactive_html(
        structure_id=structure_id,
        pdb_text=pdb_text,
        model_version=gnn_result.model_version,
        output_path=html_path,
        checkpoint_path=gnn_result.checkpoint_path,
    )

    asset_id = _asset_id(structure_id, run_id)
    prov = ProvenanceContext(
        run_id=run_id,
        structure_id=structure_id,
        model_version=gnn_result.model_version,
        pipeline_name=pipeline_name,
        run_type=RunType.INFERENCE,
        source_type=SourceType.PROBABILISTIC,
        checkpoint_uri=gnn_result.checkpoint_path,
        code_version=code_version,
    )

    from data.normalizer.core import Normalizer

    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    try:
        await normalizer.register_file_asset(
            asset_id=asset_id,
            asset_type=_ASSET_TYPE,
            prov=prov,
            storage_uri=str(html_path),
            metadata={
                "pdb_path": str(pdb_path),
                "n_atoms": n_atoms,
                "n_residues": len(gnn_result.nodes),
                "viewer_format": "ngl_html_v1",
            },
        )
    except Exception as exc:
        logger.warning("GNN viewer governed_asset registration failed (non-fatal): %s", exc)

    logger.info(
        "GNN interactive viewer written structure=%s html=%s atoms=%d residues=%d",
        structure_id,
        html_path,
        n_atoms,
        len(gnn_result.nodes),
    )

    return {
        "asset_id": asset_id,
        "html_path": str(html_path),
        "pdb_path": str(pdb_path),
        "n_atoms": n_atoms,
        "n_residues": len(gnn_result.nodes),
        "viewer_url": f"/api/structures/{structure_id}/gnn-viewer",
    }
