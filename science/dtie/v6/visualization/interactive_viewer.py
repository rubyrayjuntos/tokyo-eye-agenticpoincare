"""GNN interactive 3D viewer — enhanced PDB + standalone NGL HTML.

Restores the v2 autonomous viewer pattern for v6 ingest:
  per-residue investigation score (high aleatoric, low epistemic) → B-factor default
  cone_depth → occupancy; toggles for aleatoric / epistemic on structure + disc HTML
  cartoon + semi-transparent surface colored by selected metric (outer-shell read).

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


def disc_xy_from_model_output(model_output: dict[str, Any]) -> Any:
    """Pre-routing 2D disc coords — post-routing collapses to a streak."""
    pre = model_output.get("hyp_projections_2d_pre")
    if pre is not None:
        return pre
    return model_output["hyp_projections_2d"]


def _scale_channel(values: np.ndarray, *, out_min: float, out_max: float) -> np.ndarray:
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return np.full_like(values, (out_min + out_max) / 2.0, dtype=np.float64)
    return out_min + (values - vmin) * (out_max - out_min) / (vmax - vmin)


def _node_aleatoric(node: GNNNodeOutput) -> float:
    if node.aleatoric_uncertainty is not None:
        return float(node.aleatoric_uncertainty)
    return 0.0


def investigation_scores(epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
    """Per-residue score: high aleatoric + low epistemic → investigation priority."""
    epi = np.asarray(epistemic, dtype=np.float64)
    ale = np.asarray(aleatoric, dtype=np.float64)
    epi_n = (epi - epi.min()) / (epi.max() - epi.min() + 1e-8)
    ale_n = (ale - ale.min()) / (ale.max() - ale.min() + 1e-8)
    return ale_n * (1.0 - epi_n)


def _metric_maps_from_nodes(
    node_lookup: dict[tuple[str, int], GNNNodeOutput],
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    keys = list(node_lookup.keys())
    epistemic = np.array([node_lookup[k].epistemic_uncertainty for k in keys], dtype=np.float64)
    aleatoric = np.array([_node_aleatoric(node_lookup[k]) for k in keys], dtype=np.float64)
    investigation = investigation_scores(epistemic, aleatoric)
    epist_scaled = _scale_channel(epistemic, out_min=0.0, out_max=99.0)
    ale_scaled = _scale_channel(aleatoric, out_min=0.0, out_max=99.0)
    inv_scaled = _scale_channel(investigation, out_min=0.0, out_max=99.0)
    epist_map = {key: float(epist_scaled[i]) for i, key in enumerate(keys)}
    ale_map = {key: float(ale_scaled[i]) for i, key in enumerate(keys)}
    inv_map = {key: float(inv_scaled[i]) for i, key in enumerate(keys)}
    return epist_map, ale_map, inv_map


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
    epist_map, ale_map, inv_map = _metric_maps_from_nodes(node_lookup)
    depth_norm = _scale_channel(depth, out_min=0.0, out_max=1.0)
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
            b_factor = inv_map.get(key, 50.0) if node else 50.0
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
    epist_map, ale_map, inv_map = _metric_maps_from_nodes(node_lookup)
    depth_norm = _scale_channel(depth, out_min=0.0, out_max=1.0)
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
            b_factor = inv_map.get(key, 50.0) if node else 50.0
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
    residue_metrics: list[dict[str, Any]] | None = None,
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
    {model_version}{checkpoint_note}<br/>
    <b>ν<sub>epi</sub></b> = model training gap (red = under-trained) &nbsp;|&nbsp;
    <b>ν<sub>ale</sub></b> = structural ambiguity (red = information-poor local geometry)<br/>
    <b>Investigation</b> = high ν<sub>ale</sub> + low ν<sub>epi</sub>
    (<span class="hi">red = priority sites</span> — model is confident the structure is ambiguous here)
    &nbsp;|&nbsp; Occupancy = Cone depth
"""
        pharma_js = ""

    metrics_json = json.dumps(residue_metrics or [])
    metrics_toolbar = ""
    metrics_js = ""
    if residue_metrics:
        metrics_toolbar = """
    <label style="margin-left:12px">Color by
      <select id="structure-metric">
        <option value="investigation" selected>Investigation (high ale, low epi)</option>
        <option value="aleatoric">Aleatoric uncertainty</option>
        <option value="epistemic">Epistemic uncertainty</option>
        <option value="cone_depth">Cone depth</option>
        <option value="expert">Route expert (E0–E3)</option>
      </select>
    </label>"""
        metrics_js = """
        var RESIDUE_METRICS = __METRICS_JSON__;
        var metricLookup = {};
        RESIDUE_METRICS.forEach(function(row) { metricLookup[row.key] = row; });

        function metricRange(key) {
          var vals = RESIDUE_METRICS.map(function(r) { return r[key]; });
          return { min: Math.min.apply(null, vals), max: Math.max.apply(null, vals) };
        }

        function scaleMetric(value, key) {
          var rr = metricRange(key);
          if (rr.max <= rr.min) return 50.0;
          return 99.0 * (value - rr.min) / (rr.max - rr.min);
        }

        function bfactorForMetric(row, key) {
          if (key === "expert") {
            var bands = [12, 38, 62, 88];
            return bands[row.expert] || 50;
          }
          return scaleMetric(row[key], key);
        }

        function applyStructureMetric(key) {
          comp.structure.eachAtom(function(ap) {
            if (!ap.isProtein()) return;
            var rk = ap.chainname + ":" + ap.resno;
            var row = metricLookup[rk];
            if (!row) return;
            ap.bfactor = bfactorForMetric(row, key);
          });
          comp.updateRepresentations({ what: "color" });
        }

        document.getElementById("structure-metric").addEventListener("change", function(ev) {
          applyStructureMetric(ev.target.value);
        });"""

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
  <div id="legend">{legend_html}{metrics_toolbar}
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
        {metrics_js.replace("__METRICS_JSON__", metrics_json)}
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


def disc_payload_from_nodes(nodes: list[GNNNodeOutput]) -> list[dict[str, Any]]:
    """Serialize per-residue disc coordinates + metrics for standalone HTML."""
    points: list[dict[str, Any]] = []
    for node in nodes:
        hyp = node.hyp_projections
        if hyp is None or len(hyp) < 2:
            continue
        expert = 0
        if node.expert_weights is not None and len(node.expert_weights):
            expert = int(np.argmax(node.expert_weights))
        sasa = (
            float(node.input_features[3])
            if node.input_features is not None and len(node.input_features) > 3
            else 0.0
        )
        points.append(
            {
                "label": f"{node.chain_label}:{node.residue_index}",
                "x": float(hyp[0]),
                "y": float(hyp[1]),
                "r": float(np.hypot(hyp[0], hyp[1])),
                "epistemic": float(node.epistemic_uncertainty),
                "aleatoric": _node_aleatoric(node),
                "cone_depth": float(node.cone_depth),
                "sasa": sasa,
                "expert": expert,
            }
        )
    if points:
        epi = np.array([p["epistemic"] for p in points], dtype=np.float64)
        ale = np.array([p["aleatoric"] for p in points], dtype=np.float64)
        inv = investigation_scores(epi, ale)
        for i, point in enumerate(points):
            point["investigation"] = float(inv[i])
    return points


def residue_metrics_payload(nodes: list[GNNNodeOutput]) -> list[dict[str, Any]]:
    """Per-residue metrics for NGL structure viewer toggles (chain:resnum keys)."""
    payload: list[dict[str, Any]] = []
    if not nodes:
        return payload
    epi = np.array([n.epistemic_uncertainty for n in nodes], dtype=np.float64)
    ale = np.array([_node_aleatoric(n) for n in nodes], dtype=np.float64)
    inv = investigation_scores(epi, ale)
    for i, node in enumerate(nodes):
        expert = 0
        if node.expert_weights is not None and len(node.expert_weights):
            expert = int(np.argmax(node.expert_weights))
        payload.append(
            {
                "key": f"{node.chain_label}:{node.residue_index}",
                "epistemic": float(epi[i]),
                "aleatoric": float(ale[i]),
                "investigation": float(inv[i]),
                "cone_depth": float(node.cone_depth),
                "expert": expert,
            }
        )
    return payload


def write_poincare_disc_html(
    *,
    structure_id: str,
    points: list[dict[str, Any]],
    model_version: str,
    output_path: Path,
    checkpoint_path: str | None = None,
    curvature: float | None = None,
) -> None:
    """Write standalone interactive Poincaré disc (canvas scatter, metric toggle)."""
    sid = structure_id.strip().lower()
    title = f"{sid.upper()} — Poincaré Disc"
    checkpoint_note = (
        f" &nbsp;|&nbsp; checkpoint: {Path(checkpoint_path).name}"
        if checkpoint_path
        else ""
    )
    curv_note = f" &nbsp;|&nbsp; κ={curvature:.3f}" if curvature is not None else ""
    points_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; }}
    #toolbar {{ padding:8px 14px; background:#1a1a1a; font-size:12px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    #canvas {{ display:block; width:100vw; height:calc(100vh - 72px); cursor:crosshair; }}
    select, label {{ font-size:12px; }}
    #tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,0.85); border:1px solid #444; padding:4px 8px; font-size:11px; display:none; z-index:9; white-space:pre-line; }}
    span.hi {{ color:#f77; }} span.lo {{ color:#77f; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <b>{sid.upper()}</b> &nbsp;|&nbsp; {model_version}{checkpoint_note}{curv_note}
    &nbsp;|&nbsp; <b>pre-routing x<sub>hyp</sub></b> (not post-routing streak)
    <label>Color by
      <select id="metric">
        <option value="investigation" selected>Investigation (high ale, low epi)</option>
        <option value="aleatoric">Aleatoric uncertainty</option>
        <option value="epistemic">Epistemic uncertainty</option>
        <option value="cone_depth">Cone depth</option>
        <option value="expert">Route expert (E0–E3)</option>
        <option value="sasa">SASA proxy</option>
        <option value="r">Disc radius |z|</option>
      </select>
    </label>
    <span id="stats"></span>
  </div>
  <canvas id="canvas"></canvas>
  <div id="tip"></div>
  <script>
    var POINTS = {points_json};

    function colorScale(t) {{
      t = Math.max(0, Math.min(1, t));
      var r = t < 0.5 ? 255 : Math.round(255 - (t - 0.5) * 2 * 255);
      var b = t < 0.5 ? Math.round(t * 2 * 255) : 255;
      var g = Math.round(80 + 80 * (1 - Math.abs(t - 0.5) * 2));
      return "rgb(" + r + "," + g + "," + b + ")";
    }}

    function metricValue(p, key) {{
      if (key === "expert") return p.expert / 3.0;
      return p[key];
    }}

    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var tip = document.getElementById("tip");
    var metricSel = document.getElementById("metric");
    var hover = -1;

    function resize() {{
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight - 72;
      draw();
    }}

    function toScreen(x, y) {{
      var pad = 40;
      var size = Math.min(canvas.width, canvas.height) - pad * 2;
      var cx = canvas.width / 2;
      var cy = canvas.height / 2;
      return [cx + x * size / 2, cy - y * size / 2];
    }}

    function draw() {{
      var key = metricSel.value;
      var vals = POINTS.map(function(p) {{ return metricValue(p, key); }});
      var vmin = Math.min.apply(null, vals);
      var vmax = Math.max.apply(null, vals);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      var pad = 40;
      var size = Math.min(canvas.width, canvas.height) - pad * 2;
      var cx = canvas.width / 2;
      var cy = canvas.height / 2;
      ctx.strokeStyle = "#666";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, size / 2, 0, Math.PI * 2);
      ctx.stroke();
      for (var i = 0; i < POINTS.length; i++) {{
        var p = POINTS[i];
        var t = vmax > vmin ? (metricValue(p, key) - vmin) / (vmax - vmin) : 0.5;
        var sc = toScreen(p.x, p.y);
        ctx.beginPath();
        ctx.fillStyle = colorScale(t);
        ctx.arc(sc[0], sc[1], i === hover ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
      }}
      document.getElementById("stats").textContent =
        POINTS.length + " residues | " + key + " [" + vmin.toFixed(3) + ", " + vmax.toFixed(3) + "]";
    }}

    canvas.addEventListener("mousemove", function(ev) {{
      var rect = canvas.getBoundingClientRect();
      var mx = ev.clientX - rect.left;
      var my = ev.clientY - rect.top;
      hover = -1;
      for (var i = 0; i < POINTS.length; i++) {{
        var sc = toScreen(POINTS[i].x, POINTS[i].y);
        var dx = sc[0] - mx, dy = sc[1] - my;
        if (dx * dx + dy * dy <= 100) {{ hover = i; break; }}
      }}
      if (hover >= 0) {{
        var p = POINTS[hover];
        tip.style.display = "block";
        tip.style.left = (ev.clientX + 12) + "px";
        tip.style.top = (ev.clientY + 12) + "px";
        tip.textContent = p.label + "\\nr=" + p.r.toFixed(3) +
          " depth=" + p.cone_depth.toFixed(2) +
          " nu_epi=" + p.epistemic.toFixed(3) +
          " nu_ale=" + (p.aleatoric || 0).toFixed(3) +
          " inv=" + (p.investigation || 0).toFixed(3) +
          " E" + p.expert;
      }} else {{
        tip.style.display = "none";
      }}
      draw();
    }});

    metricSel.addEventListener("change", draw);
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def write_structure_viewers(
    *,
    structure_id: str,
    pdb_text: str,
    nodes: list[GNNNodeOutput],
    model_version: str,
    out_dir: Path,
    checkpoint_path: str | None = None,
    curvature: float | None = None,
) -> dict[str, str]:
    """Write both interactive HTML outputs (3D + Poincaré disc) for one structure."""
    sid = structure_id.strip().lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_html = out_dir / f"{sid}_interactive.html"
    disc_html = out_dir / f"{sid}_poincare_disc.html"
    write_interactive_html(
        structure_id=sid,
        pdb_text=pdb_text,
        model_version=model_version,
        output_path=structure_html,
        checkpoint_path=checkpoint_path,
        residue_metrics=residue_metrics_payload(nodes),
    )
    write_poincare_disc_html(
        structure_id=sid,
        points=disc_payload_from_nodes(nodes),
        model_version=model_version,
        output_path=disc_html,
        checkpoint_path=checkpoint_path,
        curvature=curvature,
    )
    return {
        "structure_html": str(structure_html),
        "disc_html": str(disc_html),
    }


def nodes_from_training_inference(
    prot: dict[str, Any],
    model_output: dict[str, Any],
) -> list[GNNNodeOutput]:
    """Build GNNNodeOutput list from a training-graph forward pass."""
    import torch

    data_x = prot["data"].x
    residue_ids = prot["residue_ids"]
    nodes: list[GNNNodeOutput] = []
    raw_depth = model_output["radial_features"].squeeze(-1).detach()
    depth_max = raw_depth.max() + 1e-8
    normalized_depth = (raw_depth / depth_max) * 8.0
    cone_width = torch.exp(-normalized_depth)
    for i, rid in enumerate(residue_ids):
        parts = str(rid).split(":")
        chain = parts[0] if parts else str(prot.get("chain", "A"))
        res_index = int(parts[1]) if len(parts) > 1 else i + 1
        nodes.append(
            GNNNodeOutput(
                residue_index=res_index,
                chain_label=chain,
                input_features=data_x[i].detach().cpu().numpy(),
                projections=model_output["projections"][i].detach().cpu().numpy(),
                cone_depth=float(normalized_depth[i]),
                cone_width=float(cone_width[i]),
                epistemic_uncertainty=float(model_output["uncertainty"]["epistemic"][i].detach()),
                aleatoric_uncertainty=float(model_output["uncertainty"]["aleatoric"][i].detach()),
                total_uncertainty=float(model_output["uncertainty"]["total"][i].detach()),
                hyp_projections=disc_xy_from_model_output(model_output)[i].detach().cpu().numpy(),
                expert_weights=model_output["expert_weights"][i].detach().cpu().numpy(),
            )
        )
    return nodes


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
    disc_html_path = out_dir / f"{structure_id}_poincare_disc.html"

    node_lookup = build_residue_channel_lookup(gnn_result.nodes)
    n_atoms = await write_annotated_pdb(structure_id, db, node_lookup, pdb_path)
    if n_atoms == 0:
        raise ValueError(f"No atoms in dim_atom for structure {structure_id}")

    pdb_text = pdb_path.read_text(encoding="utf-8")
    viewer_paths = write_structure_viewers(
        structure_id=structure_id,
        pdb_text=pdb_text,
        nodes=gnn_result.nodes,
        model_version=gnn_result.model_version,
        out_dir=out_dir,
        checkpoint_path=gnn_result.checkpoint_path,
        curvature=gnn_result.curvature,
    )
    html_path = Path(viewer_paths["structure_html"])
    disc_html_path = Path(viewer_paths["disc_html"])

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
                "disc_html_path": str(disc_html_path),
                "n_atoms": n_atoms,
                "n_residues": len(gnn_result.nodes),
                "viewer_format": "ngl_html_v1",
                "disc_viewer_format": "poincare_canvas_v1",
            },
        )
    except Exception as exc:
        logger.warning("GNN viewer governed_asset registration failed (non-fatal): %s", exc)

    logger.info(
        "GNN interactive viewer written structure=%s html=%s disc=%s atoms=%d residues=%d",
        structure_id,
        html_path,
        disc_html_path,
        n_atoms,
        len(gnn_result.nodes),
    )

    return {
        "asset_id": asset_id,
        "html_path": str(html_path),
        "disc_html_path": str(disc_html_path),
        "pdb_path": str(pdb_path),
        "n_atoms": n_atoms,
        "n_residues": len(gnn_result.nodes),
        "viewer_url": f"/api/structures/{structure_id}/gnn-viewer",
        "disc_viewer_url": f"/api/structures/{structure_id}/gnn-viewer/disc",
    }
