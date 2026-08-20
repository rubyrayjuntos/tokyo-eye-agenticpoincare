"""GNN interactive 3D viewer — enhanced PDB + standalone NGL HTML.

Restores the v2 autonomous viewer pattern for v6 ingest:
  default color = physics Investigation (ρ/τ underwrap) — checkpoint-stable
  evidential Investigation (ale×(1−epi)) kept as an explicitly experimental overlay
  cone_depth → occupancy; toggles for aleatoric / epistemic on structure + disc HTML
  cartoon + semi-transparent surface colored by selected metric (outer-shell read).

Disc canvas ``colorScale`` must match NGL ``RdYlBu`` + ``colorReverse: true``
(low=blue, high=red). See ``docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md``.

Triggered after ``gnn_inference`` (non-fatal). Files land under
``GNN_VIEWER_OUTPUT_DIR`` (default ``data/local_objects/gnn_viewer``).

Artifacts per structure:
  - ``{sid}_interactive.html`` — 3D NGL viewer
  - ``{sid}_poincare_disc.html`` — disc canvas viewer
  - ``{sid}_split_viewer.html`` — split-screen synced 3D + disc
  - ``{sid}_shell_signal_gate.json`` — τ-rim uncertainty spread gate
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

# Shared by disc / split HTML — must stay aligned with NGL RdYlBu + colorReverse.
_DISC_COLOR_SCALE_JS = """
    function colorScale(t) {
      // Match NGL RdYlBu + colorReverse:true — low=blue, high=red.
      t = Math.max(0, Math.min(1, t));
      var r = t < 0.5 ? Math.round(t * 2 * 255) : 255;
      var b = t < 0.5 ? 255 : Math.round(255 - (t - 0.5) * 2 * 255);
      var g = Math.round(80 + 80 * (1 - Math.abs(t - 0.5) * 2));
      return "rgb(" + r + "," + g + "," + b + ")";
    }"""

# NGL caches bfactor colors; repr.update alone often no-ops after AtomProxy writes.
# Viewers rebuild surface/cartoon (see refreshStructureColors in generated HTML).

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
    """2D disc coordinates for viewers and governance overlays.

    When structural SSOT is frozen (ingest / diagnostics), use ``hyp_projections_2d``
    directly. Otherwise prefer pre-routing coords — post-routing can collapse to a streak.
    """
    audit = model_output.get("audit_trail") or {}
    if audit.get("structural_disc_frozen") or audit.get("disc_projection_source") == (
        "structural_ssot_frozen"
    ):
        return model_output["hyp_projections_2d"]
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
    """Evidential overlay: high aleatoric + low epistemic.

    Not a trusted product default — heads are near-flat and ρ-correlated (G5b;
    ``docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md``). Prefer
    ``physics_investigation_scores`` for user-facing Investigation coloring.
    """
    epi = np.asarray(epistemic, dtype=np.float64)
    ale = np.asarray(aleatoric, dtype=np.float64)
    epi_n = (epi - epi.min()) / (epi.max() - epi.min() + 1e-8)
    ale_n = (ale - ale.min()) / (ale.max() - ale.min() + 1e-8)
    return ale_n * (1.0 - epi_n)


def physics_investigation_scores(rho: np.ndarray, tau_flag: np.ndarray) -> np.ndarray:
    """Trusted Investigation default: underwrap priority from frozen physics.

    High = dehydron-flagged (τ=1) and/or low wrapping count ρ. Independent of
    evidential heads and checkpoint seed.
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    tau = np.asarray(tau_flag, dtype=np.float64)
    if rho_arr.size == 0:
        return rho_arr
    rho_n = (rho_arr - rho_arr.min()) / (rho_arr.max() - rho_arr.min() + 1e-8)
    underwrap = 1.0 - rho_n
    return tau * underwrap + (1.0 - tau) * 0.25 * underwrap


def _node_rho(node: GNNNodeOutput) -> float:
    feats = node.input_features
    if feats is not None and len(feats) > 0:
        return float(feats[0])
    return 0.0


def _node_tau_flag(node: GNNNodeOutput) -> float:
    feats = node.input_features
    if feats is not None and len(feats) > 1:
        return float(feats[1])
    return 0.0


def _metric_maps_from_nodes(
    node_lookup: dict[tuple[str, int], GNNNodeOutput],
) -> tuple[
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
]:
    keys = list(node_lookup.keys())
    epistemic = np.array([node_lookup[k].epistemic_uncertainty for k in keys], dtype=np.float64)
    aleatoric = np.array([_node_aleatoric(node_lookup[k]) for k in keys], dtype=np.float64)
    rho = np.array([_node_rho(node_lookup[k]) for k in keys], dtype=np.float64)
    tau = np.array([_node_tau_flag(node_lookup[k]) for k in keys], dtype=np.float64)
    investigation = investigation_scores(epistemic, aleatoric)
    physics_inv = physics_investigation_scores(rho, tau)
    epist_scaled = _scale_channel(epistemic, out_min=0.0, out_max=99.0)
    ale_scaled = _scale_channel(aleatoric, out_min=0.0, out_max=99.0)
    inv_scaled = _scale_channel(investigation, out_min=0.0, out_max=99.0)
    phys_scaled = _scale_channel(physics_inv, out_min=0.0, out_max=99.0)
    epist_map = {key: float(epist_scaled[i]) for i, key in enumerate(keys)}
    ale_map = {key: float(ale_scaled[i]) for i, key in enumerate(keys)}
    inv_map = {key: float(inv_scaled[i]) for i, key in enumerate(keys)}
    phys_map = {key: float(phys_scaled[i]) for i, key in enumerate(keys)}
    return epist_map, ale_map, inv_map, phys_map


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
    _epist_map, _ale_map, _inv_map, phys_map = _metric_maps_from_nodes(node_lookup)
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
            b_factor = phys_map.get(key, 50.0) if node else 50.0
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
    _epist_map, _ale_map, _inv_map, phys_map = _metric_maps_from_nodes(node_lookup)
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
            b_factor = phys_map.get(key, 50.0) if node else 50.0
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
    <b>Investigation</b> = ρ/τ underwrap priority
    (<span class="hi">red = dehydron / under-wrapped</span> — frozen physics layer)
    &nbsp;|&nbsp; Evidential overlay available in Color by (experimental)
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
        <option value="physics_investigation" selected>Investigation (ρ/τ physics)</option>
        <option value="rho">Dehydron ρ</option>
        <option value="tau">τ flag</option>
        <option value="cone_depth">Cone depth</option>
        <option value="investigation">Investigation (evidential · experimental)</option>
        <option value="aleatoric">Aleatoric uncertainty</option>
        <option value="epistemic">Epistemic uncertainty</option>
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

        function writeAtomBfactor(ap, value) {
          ap.bfactor = value;
          var store = comp.structure.atomStore;
          if (store && store.bfactor) store.bfactor[ap.index] = value;
        }

        function refreshStructureColors() {
          // Rebuild colored reps so NGL re-reads atomStore.bfactor (update alone is a no-op).
          var colorOpts = { colorScheme: "bfactor", colorScale: "RdYlBu", colorReverse: true };
          if (surfaceRep) { comp.removeRepresentation(surfaceRep); surfaceRep = null; }
          if (cartoonRep) { comp.removeRepresentation(cartoonRep); cartoonRep = null; }
          if (heteroRep) { comp.removeRepresentation(heteroRep); heteroRep = null; }
          surfaceRep = comp.addRepresentation("surface", Object.assign({
            sele: "polymer", opacity: 0.18, side: "front", smooth: 2
          }, colorOpts));
          cartoonRep = comp.addRepresentation("cartoon", Object.assign({
            sele: "polymer", opacity: 1.0, smoothSheet: true, quality: "high"
          }, colorOpts));
          heteroRep = comp.addRepresentation("ball+stick", Object.assign({
            sele: "hetero or water or ion", opacity: 0.7
          }, colorOpts));
        }

        function applyStructureMetric(key) {
          comp.structure.eachAtom(function(ap) {
            if (!ap.isProtein()) return;
            var rk = ap.chainname + ":" + ap.resno;
            var row = metricLookup[rk];
            if (!row) return;
            writeAtomBfactor(ap, bfactorForMetric(row, key));
          });
          refreshStructureColors();
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
        var surfaceRep = comp.addRepresentation("surface", Object.assign({{
          sele: "polymer",
          opacity: 0.18,
          side: "front",
          smooth: 2
        }}, colorOpts));
        var cartoonRep = comp.addRepresentation("cartoon", Object.assign({{
          sele: "polymer",
          opacity: 1.0,
          smoothSheet: true,
          quality: "high"
        }}, colorOpts));
        var heteroRep = comp.addRepresentation("ball+stick", Object.assign({{
          sele: "hetero or water or ion",
          opacity: 0.7
        }}, colorOpts));{pharma_js}
        {metrics_js.replace("__METRICS_JSON__", metrics_json)}
        if (typeof applyStructureMetric === "function") {{
          applyStructureMetric("physics_investigation");
        }}
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


def disc_payload_from_nodes(
    nodes: list[GNNNodeOutput],
    *,
    curvature: float | None = None,
) -> list[dict[str, Any]]:
    """Serialize per-residue disc coordinates + metrics for standalone HTML."""
    from science.dtie.common.hyperbolic_lorentz_ops import hyperbolic_distance_from_origin

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
        euclid_r = float(np.hypot(hyp[0], hyp[1]))
        hyp_r = euclid_r
        if curvature is not None and curvature > 0:
            hyp_r = float(hyperbolic_distance_from_origin(np.asarray(hyp[:2]), curvature))
        points.append(
            {
                "label": f"{node.chain_label}:{node.residue_index}",
                "x": float(hyp[0]),
                "y": float(hyp[1]),
                "r": euclid_r,
                "hyperbolic_r": hyp_r,
                "epistemic": float(node.epistemic_uncertainty),
                "aleatoric": _node_aleatoric(node),
                "cone_depth": float(node.cone_depth),
                "rho": _node_rho(node),
                "tau": _node_tau_flag(node),
                "sasa": sasa,
                "expert": expert,
            }
        )
    if points:
        epi = np.array([p["epistemic"] for p in points], dtype=np.float64)
        ale = np.array([p["aleatoric"] for p in points], dtype=np.float64)
        rho = np.array([p["rho"] for p in points], dtype=np.float64)
        tau = np.array([p["tau"] for p in points], dtype=np.float64)
        inv = investigation_scores(epi, ale)
        phys = physics_investigation_scores(rho, tau)
        for i, point in enumerate(points):
            point["investigation"] = float(inv[i])
            point["physics_investigation"] = float(phys[i])
    return points


def residue_metrics_payload(nodes: list[GNNNodeOutput]) -> list[dict[str, Any]]:
    """Per-residue metrics for NGL structure viewer toggles (chain:resnum keys)."""
    payload: list[dict[str, Any]] = []
    if not nodes:
        return payload
    epi = np.array([n.epistemic_uncertainty for n in nodes], dtype=np.float64)
    ale = np.array([_node_aleatoric(n) for n in nodes], dtype=np.float64)
    rho = np.array([_node_rho(n) for n in nodes], dtype=np.float64)
    tau = np.array([_node_tau_flag(n) for n in nodes], dtype=np.float64)
    inv = investigation_scores(epi, ale)
    phys = physics_investigation_scores(rho, tau)
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
                "physics_investigation": float(phys[i]),
                "rho": float(rho[i]),
                "tau": float(tau[i]),
                "cone_depth": float(node.cone_depth),
                "expert": expert,
            }
        )
    return payload


def _disc_boundary_radius(curvature: float | None) -> float:
    """Geoopt ball radius 1/sqrt(κ) for viewer boundary circle."""
    if curvature is None or curvature <= 0:
        return 1.0
    return 1.0 / float(np.sqrt(curvature))


def write_poincare_disc_html(
    *,
    structure_id: str,
    points: list[dict[str, Any]],
    model_version: str,
    output_path: Path,
    checkpoint_path: str | None = None,
    curvature: float | None = None,
    disc_layout_label: str = "pre-routing x_hyp",
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
    disc_boundary = _disc_boundary_radius(curvature)

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
    &nbsp;|&nbsp; <b>{disc_layout_label}</b>
    <label>Color by
      <select id="metric">
        <option value="physics_investigation" selected>Investigation (ρ/τ physics)</option>
        <option value="rho">Dehydron ρ</option>
        <option value="tau">τ flag</option>
        <option value="cone_depth">Cone depth</option>
        <option value="investigation">Investigation (evidential · experimental)</option>
        <option value="aleatoric">Aleatoric uncertainty</option>
        <option value="epistemic">Epistemic uncertainty</option>
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
    var DISC_BOUNDARY = {disc_boundary};
{_DISC_COLOR_SCALE_JS}

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
      var scale = (size / 2) / DISC_BOUNDARY;
      return [cx + x * scale, cy - y * scale];
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
          " d_H=" + (p.hyperbolic_r != null ? p.hyperbolic_r.toFixed(3) : p.r.toFixed(3)) +
          " depth=" + p.cone_depth.toFixed(2) +
          " rho=" + (p.rho != null ? p.rho.toFixed(1) : "?") +
          " tau=" + (p.tau != null ? p.tau.toFixed(0) : "?") +
          " phys=" + (p.physics_investigation || 0).toFixed(3) +
          " inv_e=" + (p.investigation || 0).toFixed(3) +
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


def write_split_screen_viewer_html(
    *,
    structure_id: str,
    pdb_text: str,
    points: list[dict[str, Any]],
    residue_metrics: list[dict[str, Any]],
    model_version: str,
    output_path: Path,
    checkpoint_path: str | None = None,
    curvature: float | None = None,
    disc_layout_label: str = "structural SSOT",
) -> None:
    """Split-screen 3D structure (NGL) + Poincaré disc with synced hover, click, and metrics."""
    sid = structure_id.strip().lower()
    pdb_json = json.dumps(pdb_text)
    points_json = json.dumps(points)
    metrics_json = json.dumps(residue_metrics)
    title = f"{sid.upper()} — Structure + Poincaré Disc"
    checkpoint_note = (
        f" &nbsp;|&nbsp; checkpoint: {Path(checkpoint_path).name}"
        if checkpoint_path
        else ""
    )
    curv_note = f" &nbsp;|&nbsp; κ={curvature:.3f}" if curvature is not None else ""
    disc_boundary = _disc_boundary_radius(curvature)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://unpkg.com/ngl@2.1.0/dist/ngl.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; height:100vh; display:flex; flex-direction:column; }}
    #main {{ flex:1; display:flex; min-height:0; }}
    #panel-3d, #panel-disc {{ flex:1; min-width:0; position:relative; border-right:1px solid #333; }}
    #panel-disc {{ border-right:none; }}
    #viewport-3d {{ width:100%; height:100%; }}
    #canvas-disc {{ display:block; width:100%; height:100%; cursor:grab; }}
    #canvas-disc.dragging {{ cursor:grabbing; }}
    .panel-label {{ position:absolute; top:8px; left:10px; font-size:11px; color:#aaa; z-index:2; pointer-events:none; }}
    #toolbar {{ padding:8px 14px; font-size:12px; line-height:1.6; background:#1a1a1a; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
    #detail {{ margin-left:auto; font-size:11px; color:#ccc; max-width:42vw; text-align:right; }}
    span.hi {{ color:#f77; }} span.lo {{ color:#77f; }}
    #tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,0.9); border:1px solid #555; padding:6px 10px; font-size:11px; display:none; z-index:99; white-space:pre-line; }}
  </style>
</head>
<body>
  <div id="main">
    <div id="panel-3d">
      <div class="panel-label">3D structure — drag rotate, scroll zoom, right-drag pan</div>
      <div id="viewport-3d"></div>
    </div>
    <div id="panel-disc">
      <div class="panel-label">Poincaré disc — drag pan, scroll zoom</div>
      <canvas id="canvas-disc"></canvas>
    </div>
  </div>
  <div id="toolbar">
    <b>{sid.upper()}</b> &nbsp;|&nbsp; {model_version}{checkpoint_note}{curv_note}
    &nbsp;|&nbsp; <b>{disc_layout_label}</b>
    <label>Color by
      <select id="metric">
        <option value="physics_investigation" selected>Investigation (ρ/τ physics)</option>
        <option value="rho">Dehydron ρ</option>
        <option value="tau">τ flag</option>
        <option value="cone_depth">Cone depth</option>
        <option value="investigation">Investigation (evidential · experimental)</option>
        <option value="aleatoric">Aleatoric</option>
        <option value="epistemic">Epistemic</option>
        <option value="expert">Expert route</option>
        <option value="r">Disc radius</option>
      </select>
    </label>
    <span>Occupancy = depth on 3D surface</span>
    <span id="detail">Hover or click a residue</span>
  </div>
  <div id="tip"></div>
  <script>
    var PDB_DATA = {pdb_json};
    var POINTS = {points_json};
    var RESIDUE_METRICS = {metrics_json};
    var DISC_BOUNDARY = {disc_boundary};

    var pointByKey = {{}};
    POINTS.forEach(function(p) {{ pointByKey[p.label] = p; }});

    var metricLookup = {{}};
    RESIDUE_METRICS.forEach(function(r) {{ metricLookup[r.key] = r; }});

    // Unified per-residue row (metrics + disc fields) for 3D coloring
    var unifiedRow = {{}};
    RESIDUE_METRICS.forEach(function(r) {{
      unifiedRow[r.key] = Object.assign({{}}, r);
    }});
    POINTS.forEach(function(p) {{
      unifiedRow[p.label] = Object.assign(unifiedRow[p.label] || {{}}, p);
    }});

    function residueKey(chain, resno) {{
      var ch = (chain || "A").trim();
      if (!ch) ch = "A";
      return ch + ":" + resno;
    }}

    function nglSelection(key) {{
      var parts = key.split(":");
      if (parts.length < 2) return key;
      return parts[1] + ":" + parts[0];
    }}

    var selectedKey = null;
    var hoverKey = null;
    var metricKey = "physics_investigation";

    // ── Disc view transform (pan / zoom) ─────────────────────────────
    var canvas = document.getElementById("canvas-disc");
    var ctx = canvas.getContext("2d");
    var viewScale = 1.0;
    var viewOffX = 0, viewOffY = 0;
    var dragging = false, dragX = 0, dragY = 0;

    function discLayout() {{
      var pad = 36;
      var size = Math.min(canvas.width, canvas.height) - pad * 2;
      return {{ pad: pad, size: size, cx: canvas.width / 2 + viewOffX, cy: canvas.height / 2 + viewOffY }};
    }}

    function toScreen(x, y) {{
      var L = discLayout();
      var scale = (L.size * viewScale / 2) / DISC_BOUNDARY;
      return [L.cx + x * scale, L.cy - y * scale];
    }}
{_DISC_COLOR_SCALE_JS}

    function metricValue(row, key) {{
      if (key === "expert") return (row.expert || 0) / 3.0;
      if (key === "r") return row.r || 0;
      return row[key];
    }}

    function metricRange(key) {{
      var vals = [];
      for (var k in unifiedRow) {{
        if (Object.prototype.hasOwnProperty.call(unifiedRow, k)) {{
          vals.push(metricValue(unifiedRow[k], key));
        }}
      }}
      if (!vals.length) return {{ min: 0, max: 1 }};
      return {{ min: Math.min.apply(null, vals), max: Math.max.apply(null, vals) }};
    }}

    function scaleMetric(value, key) {{
      var rr = metricRange(key);
      if (rr.max <= rr.min) return 50.0;
      return 99.0 * (value - rr.min) / (rr.max - rr.min);
    }}

    function bfactorForMetric(row, key) {{
      if (key === "expert") {{
        var bands = [12, 38, 62, 88];
        return bands[row.expert] || 50;
      }}
      return scaleMetric(metricValue(row, key), key);
    }}

    function formatDetail(key) {{
      var p = pointByKey[key];
      if (!p) return key;
      return key + " | r=" + p.r.toFixed(3) + " d_H=" + (p.hyperbolic_r != null ? p.hyperbolic_r.toFixed(3) : p.r.toFixed(3)) +
        " depth=" + p.cone_depth.toFixed(2) +
        " rho=" + (p.rho != null ? Number(p.rho).toFixed(1) : "?") +
        " tau=" + (p.tau != null ? Number(p.tau).toFixed(0) : "?") +
        " phys=" + (p.physics_investigation || 0).toFixed(3) +
        " inv_e=" + (p.investigation || 0).toFixed(3) + " E" + p.expert;
    }}

    function updateStructureHighlight() {{
      if (!highlightRep) return;
      var key = selectedKey || hoverKey;
      if (!key) {{
        highlightRep.setSelection("none");
        return;
      }}
      highlightRep.setSelection(nglSelection(key));
    }}

    function refreshStructureColors() {{
      if (!comp) return;
      // Rebuild polymer reps so NGL re-reads atomStore.bfactor (update alone is a no-op).
      var colorOpts = {{
        colorScheme: "bfactor",
        colorScale: "RdYlBu",
        colorReverse: true
      }};
      if (surfaceRep) {{ comp.removeRepresentation(surfaceRep); surfaceRep = null; }}
      if (cartoonRep) {{ comp.removeRepresentation(cartoonRep); cartoonRep = null; }}
      surfaceRep = comp.addRepresentation("surface", Object.assign({{
        sele: "polymer",
        opacity: 0.18,
        side: "front",
        smooth: 2,
        pickable: false
      }}, colorOpts));
      cartoonRep = comp.addRepresentation("cartoon", Object.assign({{
        sele: "polymer",
        opacity: 1.0,
        smoothSheet: true,
        quality: "high",
        pickable: true
      }}, colorOpts));
    }}

    function writeAtomBfactor(ap, value) {{
      ap.bfactor = value;
      var store = comp.structure.atomStore;
      if (store && store.bfactor) store.bfactor[ap.index] = value;
    }}

    function writeAtomOccupancy(ap, value) {{
      ap.occupancy = value;
      var store = comp.structure.atomStore;
      if (store && store.occupancy) store.occupancy[ap.index] = value;
    }}

    function applyStructureMetric(key) {{
      if (!comp) return;
      var depthRR = metricRange("cone_depth");
      comp.structure.eachAtom(function(ap) {{
        if (!ap.isProtein()) return;
        var rk = residueKey(ap.chainname, ap.resno);
        var row = unifiedRow[rk];
        if (!row) return;
        writeAtomBfactor(ap, bfactorForMetric(row, key));
        if (row.cone_depth != null) {{
          var occ = depthRR.max > depthRR.min
            ? 0.35 + 0.65 * (row.cone_depth - depthRR.min) / (depthRR.max - depthRR.min)
            : 0.5;
          writeAtomOccupancy(ap, occ);
        }}
      }});
      refreshStructureColors();
    }}

    function setSelected(key) {{
      selectedKey = key;
      document.getElementById("detail").textContent = key ? formatDetail(key) : "Hover or click a residue";
      drawDisc();
      updateStructureHighlight();
    }}

    function drawDisc() {{
      var key = metricKey;
      var vals = POINTS.map(function(p) {{ return metricValue(p, key); }});
      var vmin = Math.min.apply(null, vals);
      var vmax = Math.max.apply(null, vals);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      var L = discLayout();
      ctx.strokeStyle = "#666";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(L.cx, L.cy, L.size * viewScale / 2, 0, Math.PI * 2);
      ctx.stroke();
      for (var i = 0; i < POINTS.length; i++) {{
        var p = POINTS[i];
        var t = vmax > vmin ? (metricValue(p, key) - vmin) / (vmax - vmin) : 0.5;
        var sc = toScreen(p.x, p.y);
        var isSel = p.label === selectedKey;
        var isHov = p.label === hoverKey;
        var rad = isSel ? 7 : (isHov ? 5.5 : 3.5);
        ctx.beginPath();
        ctx.fillStyle = colorScale(t);
        ctx.arc(sc[0], sc[1], rad, 0, Math.PI * 2);
        ctx.fill();
        if (isSel || isHov) {{
          ctx.strokeStyle = isSel ? "#fff" : "#aaa";
          ctx.lineWidth = isSel ? 2 : 1.5;
          ctx.stroke();
        }}
      }}
    }}

    function hitTestDisc(mx, my) {{
      var best = -1, bestD = 120;
      for (var i = 0; i < POINTS.length; i++) {{
        var sc = toScreen(POINTS[i].x, POINTS[i].y);
        var dx = sc[0] - mx, dy = sc[1] - my;
        var d2 = dx * dx + dy * dy;
        if (d2 < bestD) {{ bestD = d2; best = i; }}
      }}
      return best;
    }}

    function resizeDisc() {{
      var panel = document.getElementById("panel-disc");
      canvas.width = panel.clientWidth;
      canvas.height = panel.clientHeight;
      drawDisc();
    }}

    canvas.addEventListener("wheel", function(ev) {{
      ev.preventDefault();
      var factor = ev.deltaY < 0 ? 1.08 : 0.92;
      viewScale = Math.max(0.4, Math.min(6.0, viewScale * factor));
      drawDisc();
    }}, {{ passive: false }});

    canvas.addEventListener("mousedown", function(ev) {{
      dragging = true;
      dragX = ev.clientX; dragY = ev.clientY;
      canvas.classList.add("dragging");
    }});
    window.addEventListener("mouseup", function() {{
      dragging = false;
      canvas.classList.remove("dragging");
    }});
    canvas.addEventListener("mousemove", function(ev) {{
      if (dragging) {{
        viewOffX += ev.clientX - dragX;
        viewOffY += ev.clientY - dragY;
        dragX = ev.clientX; dragY = ev.clientY;
        drawDisc();
        return;
      }}
      var rect = canvas.getBoundingClientRect();
      var idx = hitTestDisc(ev.clientX - rect.left, ev.clientY - rect.top);
      var tip = document.getElementById("tip");
      if (idx >= 0) {{
        hoverKey = POINTS[idx].label;
        tip.style.display = "block";
        tip.style.left = (ev.clientX + 12) + "px";
        tip.style.top = (ev.clientY + 12) + "px";
        tip.textContent = formatDetail(hoverKey);
        updateStructureHighlight();
      }} else {{
        hoverKey = null;
        tip.style.display = "none";
        updateStructureHighlight();
      }}
      drawDisc();
    }});
    canvas.addEventListener("click", function(ev) {{
      var rect = canvas.getBoundingClientRect();
      var idx = hitTestDisc(ev.clientX - rect.left, ev.clientY - rect.top);
      setSelected(idx >= 0 ? POINTS[idx].label : null);
    }});

    // ── NGL 3D ───────────────────────────────────────────────────────
    var stage, comp, surfaceRep, cartoonRep, highlightRep;
    var lastMouseX = 0, lastMouseY = 0;
    document.addEventListener("mousemove", function(ev) {{
      lastMouseX = ev.clientX;
      lastMouseY = ev.clientY;
    }});

    function initNglStage() {{
      stage = new NGL.Stage("viewport-3d", {{ backgroundColor: "#111111" }});
      stage.setParameters({{ hoverTimeout: 0 }});
      var blob = new Blob([PDB_DATA], {{ type: "text/plain" }});
      stage.loadFile(blob, {{ ext: "pdb", defaultRepresentation: false }}).then(function(c) {{
        comp = c;
        var colorOpts = {{
          colorScheme: "bfactor",
          colorScale: "RdYlBu",
          colorReverse: true
        }};
        surfaceRep = comp.addRepresentation("surface", Object.assign({{
          sele: "polymer",
          opacity: 0.18,
          side: "front",
          smooth: 2,
          pickable: false
        }}, colorOpts));
        cartoonRep = comp.addRepresentation("cartoon", Object.assign({{
          sele: "polymer",
          opacity: 1.0,
          smoothSheet: true,
          quality: "high",
          pickable: true
        }}, colorOpts));
        highlightRep = comp.addRepresentation("licorice", {{
          sele: "none",
          colorValue: 0xffcc44,
          opacity: 1.0,
          radiusScale: 1.4,
          pickable: false
        }});
        applyStructureMetric(metricKey);
        comp.autoView();

        stage.signals.clicked.add(function(pickingProxy) {{
          if (!pickingProxy || !pickingProxy.atom) return;
          var ap = pickingProxy.atom;
          if (!ap.isProtein()) return;
          var key = residueKey(ap.chainname, ap.resno);
          if (!unifiedRow[key]) return;
          setSelected(selectedKey === key ? null : key);
        }});

        stage.signals.hovered.add(function(pickingProxy) {{
          if (dragging) return;
          var tip = document.getElementById("tip");
          if (!pickingProxy || !pickingProxy.atom || !pickingProxy.atom.isProtein()) {{
            if (!selectedKey) hoverKey = null;
            tip.style.display = "none";
            updateStructureHighlight();
            drawDisc();
            return;
          }}
          var ap = pickingProxy.atom;
          var key = residueKey(ap.chainname, ap.resno);
          if (!unifiedRow[key]) return;
          hoverKey = key;
          tip.style.display = "block";
          var mx = (pickingProxy.mouse && pickingProxy.mouse.position)
            ? pickingProxy.mouse.position.x : lastMouseX;
          var my = (pickingProxy.mouse && pickingProxy.mouse.position)
            ? pickingProxy.mouse.position.y : lastMouseY;
          tip.style.left = (mx + 12) + "px";
          tip.style.top = (my + 12) + "px";
          tip.textContent = formatDetail(hoverKey);
          updateStructureHighlight();
          drawDisc();
        }});
      }});
    }}

    function bootSplitViewer() {{
      initNglStage();
      resizeDisc();
      window.addEventListener("resize", function() {{
        if (stage) stage.handleResize();
        resizeDisc();
      }});
    }}

    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", bootSplitViewer);
    }} else {{
      bootSplitViewer();
    }}

    document.getElementById("metric").addEventListener("change", function(ev) {{
      metricKey = ev.target.value;
      applyStructureMetric(metricKey);
      drawDisc();
    }});
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
    disc_layout_label: str = "pre-routing x_hyp",
) -> dict[str, str]:
    """Write 3D, Poincaré disc, and split-screen interactive HTML for one structure."""
    sid = structure_id.strip().lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_html = out_dir / f"{sid}_interactive.html"
    disc_html = out_dir / f"{sid}_poincare_disc.html"
    split_html = out_dir / f"{sid}_split_viewer.html"
    write_interactive_html(
        structure_id=sid,
        pdb_text=pdb_text,
        model_version=model_version,
        output_path=structure_html,
        checkpoint_path=checkpoint_path,
        residue_metrics=residue_metrics_payload(nodes),
    )
    points = disc_payload_from_nodes(nodes, curvature=curvature)
    write_poincare_disc_html(
        structure_id=sid,
        points=points,
        model_version=model_version,
        output_path=disc_html,
        checkpoint_path=checkpoint_path,
        curvature=curvature,
        disc_layout_label=disc_layout_label,
    )
    write_split_screen_viewer_html(
        structure_id=sid,
        pdb_text=pdb_text,
        points=points,
        residue_metrics=residue_metrics_payload(nodes),
        model_version=model_version,
        output_path=split_html,
        checkpoint_path=checkpoint_path,
        curvature=curvature,
        disc_layout_label=disc_layout_label,
    )
    return {
        "structure_html": str(structure_html),
        "disc_html": str(disc_html),
        "split_html": str(split_html),
    }


def nodes_from_training_inference(
    prot: dict[str, Any],
    model_output: dict[str, Any],
    *,
    data: Any | None = None,
) -> list[GNNNodeOutput]:
    """Build GNNNodeOutput list from a training-graph forward pass."""
    import torch

    graph_data = data if data is not None else prot["data"]
    data_x = graph_data.x
    residue_ids = prot["residue_ids"]
    nodes: list[GNNNodeOutput] = []
    if bool(getattr(graph_data, "structural_z_disc_frozen", False)) and hasattr(
        graph_data, "structural_cone_depth"
    ):
        cone_depth_tensor = graph_data.structural_cone_depth.detach().float()
    else:
        raw_depth = model_output["radial_features"].squeeze(-1).detach()
        depth_max = raw_depth.max() + 1e-8
        cone_depth_tensor = (raw_depth / depth_max) * 8.0
    cone_width = torch.exp(-cone_depth_tensor)
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
                cone_depth=float(cone_depth_tensor[i]),
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
    split_html_path = out_dir / f"{structure_id}_split_viewer.html"

    node_lookup = build_residue_channel_lookup(gnn_result.nodes)
    n_atoms = await write_annotated_pdb(structure_id, db, node_lookup, pdb_path)
    if n_atoms == 0:
        raise ValueError(f"No atoms in dim_atom for structure {structure_id}")

    pdb_text = pdb_path.read_text(encoding="utf-8")
    meta = gnn_result.metadata or {}
    disc_layout_label = "structural SSOT (ρ, τ, Cα)"
    if meta.get("structural_disc_frozen"):
        layout = meta.get("structural_disc_layout") or "structural_tau_rho_pca_expmap0"
        disc_layout_label = f"structural SSOT — {layout}"
    elif meta.get("hyp_projections_2d_source"):
        disc_layout_label = str(meta["hyp_projections_2d_source"])

    viewer_paths = write_structure_viewers(
        structure_id=structure_id,
        pdb_text=pdb_text,
        nodes=gnn_result.nodes,
        model_version=gnn_result.model_version,
        out_dir=out_dir,
        checkpoint_path=gnn_result.checkpoint_path,
        curvature=gnn_result.curvature,
        disc_layout_label=disc_layout_label,
    )
    html_path = Path(viewer_paths["structure_html"])
    disc_html_path = Path(viewer_paths["disc_html"])
    split_html_path = Path(viewer_paths["split_html"])

    structural_frozen = bool(meta.get("structural_disc_frozen"))
    from science.dtie.v66.visualization.shell_signal_gate import (
        evaluate_shell_signal_ssot,
        write_shell_gate_report,
    )

    gate_verdict = evaluate_shell_signal_ssot(
        gnn_result.nodes,
        structure_id=structure_id,
        structural_disc_frozen=structural_frozen,
    )
    gate_path = out_dir / f"{structure_id}_shell_signal_gate.json"
    write_shell_gate_report(gate_verdict, gate_path)
    if not gate_verdict.passed:
        logger.warning(
            "Shell signal gate FAILED for %s: %s",
            structure_id,
            gate_verdict.summary,
        )
    elif gate_verdict.warnings:
        for w in gate_verdict.warnings:
            logger.warning("Shell signal gate note (%s): %s", structure_id, w)
    else:
        logger.info("Shell signal gate passed for %s", structure_id)

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
                "split_html_path": str(split_html_path),
                "shell_gate_path": str(gate_path),
                "shell_signal_gate_passed": gate_verdict.passed,
                "n_atoms": n_atoms,
                "n_residues": len(gnn_result.nodes),
                "viewer_format": "ngl_html_v1",
                "disc_viewer_format": "poincare_canvas_v1",
                "split_viewer_format": "split_sync_v1",
            },
        )
    except Exception as exc:
        logger.warning("GNN viewer governed_asset registration failed (non-fatal): %s", exc)

    logger.info(
        "GNN interactive viewer written structure=%s html=%s disc=%s split=%s atoms=%d residues=%d gate=%s",
        structure_id,
        html_path,
        disc_html_path,
        split_html_path,
        n_atoms,
        len(gnn_result.nodes),
        "pass" if gate_verdict.passed else "FAIL",
    )

    return {
        "asset_id": asset_id,
        "html_path": str(html_path),
        "disc_html_path": str(disc_html_path),
        "split_html_path": str(split_html_path),
        "shell_gate_path": str(gate_path),
        "shell_signal_gate": gate_verdict.to_dict(),
        "pdb_path": str(pdb_path),
        "n_atoms": n_atoms,
        "n_residues": len(gnn_result.nodes),
        "viewer_url": f"/api/structures/{structure_id}/gnn-viewer",
        "disc_viewer_url": f"/api/structures/{structure_id}/gnn-viewer/disc",
        "split_viewer_url": f"/api/structures/{structure_id}/gnn-viewer/split",
    }
