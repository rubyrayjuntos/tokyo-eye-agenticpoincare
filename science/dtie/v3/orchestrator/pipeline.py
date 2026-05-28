# Migrated from: SRC_DEM/DTIE_GNN_ORCHESTRATION/dtie_pipeline.py on 2026-05-27
"""
dtie_pipeline.py
================
Eidetix Bio — Dynamic Topology Inference Engine v3.0
Full pipeline orchestrator: GDP/GTP PDB inputs → state-selective ranked drug candidates.

Usage
-----
    python dtie_pipeline.py \
        --gdp    data/egfr_gdp.pdb \
        --gtp    data/egfr_gtp.pdb \
        --protein data/egfr_gdp.pdb \
        --library data/screening_library.sdf \
        --checkpoint models/gosp_gnn.ckpt \
        --output results/egfr_run_01 \
        --test-glu697

Pipeline Phases
---------------
    Phase 1   — Hyperbolic State Ingestion & Witness Selection
    Phase 2   — Differential Vulnerability Scan (GDP vs GTP)
    Phase 3   — Witness Complex & Persistence Extraction
    Phase 3.5 — Topological Lift: persistence bars → 3D Euclidean coordinates
    Phase 4   — Effective Resistance & Allosteric Conductance Mapping
    Phase 5   — Pharmacophore Generation (Fpocket + VdW grid)
    Phase 6a  — Virtual Screening (RDKit pharmacophore pre-filter)
    Phase 6b  — Binding Affinity Estimation (AutoDock Vina)
    Phase 6c  — ADMET Filter (Lipinski, hERG, CYP3A4, Ames)
    Phase 6d  — State Selectivity Verification (GDP vs GTP docking ratio)

Scientific mandate
------------------
    - TAU = 13.0 enforced throughout (corrected from v2.0's erroneous 19.0)
    - No result incorporated unless validated with canonical GOSP formulas
    - Curvature c is topologically invariant over 1.0–1.5 for tested targets;
      supply --checkpoint to use GNN-predicted c when training data are available

Confidential — Eidetix Bio | March 2026
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import sys
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_FRAGMENT_LIBRARY = (
    Path(__file__).resolve().parent
    / "data"
    / "libraries"
    / "enamine_fragment_proxy.sdf"
)

"""
dtie_pipeline.py
================
Eidetix Bio — Dynamic Topology Inference Engine v3.0
Full pipeline orchestrator: GDP/GTP PDB inputs → state-selective ranked drug candidates.

Usage
-----
    python dtie_pipeline.py \
        --gdp    data/egfr_gdp.pdb \
        --gtp    data/egfr_gtp.pdb \
        --protein data/egfr_gdp.pdb \
        --library data/screening_library.sdf \
        --checkpoint models/gosp_gnn.ckpt \
        --output results/egfr_run_01 \
        --test-glu697

Pipeline Phases
---------------
    Phase 1   — Hyperbolic State Ingestion & Witness Selection
    Phase 2   — Differential Vulnerability Scan (GDP vs GTP)
    Phase 3   — Witness Complex & Persistence Extraction
    Phase 3.5 — Topological Lift: persistence bars → 3D Euclidean coordinates
    Phase 4   — Effective Resistance & Allosteric Conductance Mapping
    Phase 5   — Pharmacophore Generation (Fpocket + VdW grid)
    Phase 6a  — Virtual Screening (RDKit pharmacophore pre-filter)
    Phase 6b  — Binding Affinity Estimation (AutoDock Vina)
    Phase 6c  — ADMET Filter (Lipinski, hERG, CYP3A4, Ames)
    Phase 6d  — State Selectivity Verification (GDP vs GTP docking ratio)

Scientific mandate
------------------
    - TAU = 13.0 enforced throughout (corrected from v2.0's erroneous 19.0)
    - No result incorporated unless validated with canonical GOSP formulas
    - Curvature c is topologically invariant over 1.0–1.5 for tested targets;
      supply --checkpoint to use GNN-predicted c when training data are available

Confidential — Eidetix Bio | March 2026
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import sys
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

import numpy as np

try:
    from . import utils
    from .contracts import (
        Phase1Input,
        Phase1Output,
        Phase2Input,
        Phase2Output,
        Phase3Input,
        Phase3Output,
        Phase35Input,
        Phase35Output,
        Phase4Input,
        Phase5Input,
        Phase6aInput,
        Phase6bInput,
        Phase6cInput,
        Phase6dInput,
        LiftedSite,
        Doorway,
    )
    from .ingestion import ingest_pdb
    from .gnn_runner import run_gnn
except ImportError:
    import utils
    from contracts import (
        Phase1Input,
        Phase1Output,
        Phase2Input,
        Phase2Output,
        Phase3Input,
        Phase3Output,
        Phase35Input,
        Phase35Output,
        Phase4Input,
        Phase5Input,
        Phase6aInput,
        Phase6bInput,
        Phase6cInput,
        Phase6dInput,
        LiftedSite,
        Doorway,
    )
    from ingestion import ingest_pdb
    from gnn_runner import run_gnn

DEFAULT_FRAGMENT_LIBRARY = (
    Path(__file__).resolve().parent
    / "data"
    / "libraries"
    / "enamine_fragment_proxy.sdf"
)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "robust_experts.pt"

# ── Phase imports ─────────────────────────────────────────────────────────────
# Each phase module is imported inside the runner functions so that import
# errors are caught per-phase with actionable messages rather than a silent
# top-level crash.


def _import_phase(module_name: str, phase_label: str):
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import {phase_label} module '{module_name}': {e}\n"
            f"Ensure all dependencies are installed and the module is in PYTHONPATH."
        ) from e


def _result_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


# ── Checkpoint loader ─────────────────────────────────────────────────────────


def load_calibrated_c(checkpoint_path: Optional[str], logger: logging.Logger) -> float:
    """
    DEPRECATED: Use gnn_output.npz curvature_c from gnn_runner.py instead.
    This function is superseded by the ingestion → GNN runner preamble which
    extracts curvature_c directly from the checkpoint via softplus(log_c) + 1e-4
    and writes it to gnn_output.npz.

    Load curvature parameter c from GOSP-GNN training checkpoint.
    Raises if checkpoint exists but c is missing — miscalibrated c corrupts
    all downstream geodesic distances and persistence barcodes.
    """
    if checkpoint_path is None:
        logger.warning(
            "No checkpoint provided — using CALIBRATED_C = 1.0. "
            "This must be replaced with your GOSP-GNN training value before production runs."
        )
        return 1.0

    try:
        import torch
        import torch.nn.functional as F

        meta = torch.load(checkpoint_path, map_location="cpu")
        c = meta.get("curvature_c") if isinstance(meta, dict) else None
        if c is None and isinstance(meta, dict):
            state = meta.get("model_state_dict")
            if isinstance(state, dict):
                log_c_key = next((k for k in state.keys() if k.endswith("log_c") or k == "log_c"), None)
                if log_c_key is not None:
                    raw_log_c = state[log_c_key]
                    if hasattr(raw_log_c, "detach"):
                        raw_log_c = raw_log_c.detach().cpu().reshape(-1)[0]
                    c = F.softplus(torch.tensor(float(raw_log_c))).item() + 1e-4
                    logger.info(
                        f"Derived curvature c={c:.6f} from checkpoint parameter '{log_c_key}'"
                    )

        if c is None:
            raise ValueError("Checkpoint missing required curvature_c/log_c signal")

        logger.info(f"Loaded calibrated curvature c={c} from checkpoint")
        return float(c)
    except Exception as e:
        logger.warning(
            f"Failed to load checkpoint curvature_c ({e}); falling back to CALIBRATED_C = 1.0"
        )
        return 1.0


# ── Structure loader ──────────────────────────────────────────────────────────


def load_structure_ensemble(pdb_path: str, label: str, logger: logging.Logger):
    """
    Load a PDB file as a BioPython Structure object wrapped in a lightweight
    ensemble container that exposes the interface expected by Phase 1/2.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    logger.info(f"Loading {label} structure: {pdb_path}")
    suffix = Path(pdb_path).suffix.lower()
    parser = (
        MMCIFParser(QUIET=True)
        if suffix in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure(label, pdb_path)

    class EnsembleWrapper:
        """Thin wrapper giving Phase 1/2 the interface they expect."""

        def __init__(self, structure):
            self._structure = structure
            # Filter to protein residues only (hetflag ' ') — excludes HOH and ligands
            all_res = list(structure.get_residues())
            self._residues = [r for r in all_res if r.get_id()[0] == " "]
            self._atoms = [a for r in self._residues for a in r.get_atoms()]

        def get_coordinates(self) -> np.ndarray:
            return np.array([a.get_coord() for a in self._atoms])

        def get_all_atoms(self):
            return iter(self._atoms)

        @property
        def residues(self):
            return self._residues

        def get_residue_by_id(self, res_id):
            for r in self._residues:
                if r.get_id() == res_id or str(r.get_id()) == str(res_id):
                    return r
            return None

        def get_residue_id_for_atom(self, atom_idx: int):
            if atom_idx < len(self._atoms):
                return self._atoms[atom_idx].get_parent().get_id()
            return "UNKNOWN"

    return EnsembleWrapper(structure)


def load_protein_atoms(pdb_path: str, logger: logging.Logger) -> list:
    from Bio.PDB import MMCIFParser, PDBParser

    suffix = Path(pdb_path).suffix.lower()
    parser = (
        MMCIFParser(QUIET=True)
        if suffix in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("protein", pdb_path)
    # Protein residues only (hetflag ' ') — excludes water and crystallographic ligands,
    # consistent with EnsembleWrapper. Including HETATM atoms contaminates VdW grid
    # and pocket feature inference in Phase 5.
    atoms = [
        a
        for r in structure.get_residues()
        if r.get_id()[0] == " "
        for a in r.get_atoms()
    ]
    logger.info(f"Loaded {len(atoms)} protein atoms from {pdb_path}")
    return atoms


# ── GNN loader ────────────────────────────────────────────────────────────────


def load_evidential_gnn(
    checkpoint_path: Optional[str],
    logger: logging.Logger,
    device: str = "cpu",
    gdp_pdb_path: Optional[str] = None,
    gdp_structure=None,
):
    """Load the GOSPConeMapper evidential GNN via EvidentialGNNAdapter."""
    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint not found: {checkpoint_path}. Falling back to adapter without checkpoint."
        )
        checkpoint_path = None

    # Load real GOSPConeMapper via adapter
    try:
        candidate_paths = [
            Path(__file__).resolve().parent / "gosp_adapter.py",
            Path(__file__).resolve().parent
            / "backend"
            / "gnn-model"
            / "gosp_adapter.py",
        ]

        last_error = None
        for idx, adapter_path in enumerate(candidate_paths):
            if not adapter_path.exists():
                continue

            spec = importlib.util.spec_from_file_location(
                f"gosp_adapter_{idx}", adapter_path
            )
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            load_fn = getattr(module, "load_evidential_gnn", None)
            if load_fn is not None:
                try:
                    adapter = load_fn(
                        checkpoint_path=checkpoint_path,
                        device=device,
                        pdb_path=gdp_pdb_path,
                        structure=gdp_structure,
                    )
                    logger.info(
                        f"EvidentialGNNAdapter loaded from checkpoint: {checkpoint_path} "
                        f"(adapter={adapter_path})"
                    )
                    return adapter
                except Exception as load_exc:
                    last_error = load_exc

            # Fallback path: some adapters expose only the class constructor
            # and can run without checkpoint weights.
            adapter_cls = getattr(module, "EvidentialGNNAdapter", None)
            if adapter_cls is not None:
                try:
                    adapter = adapter_cls(checkpoint_path=None, device=device)
                    logger.warning(
                        f"No usable checkpoint-backed adapter in {adapter_path}; "
                        "using fallback EvidentialGNNAdapter without checkpoint."
                    )
                    return adapter
                except Exception as class_exc:
                    last_error = class_exc

            if load_fn is None and adapter_cls is None:
                last_error = AttributeError(
                    f"{adapter_path} does not export load_evidential_gnn or EvidentialGNNAdapter"
                )

        if last_error is not None:
            raise last_error
        raise FileNotFoundError("No usable gosp_adapter.py found")

    except Exception as exc:
        logger.error(
            f"Failed to load EvidentialGNNAdapter ({exc}). "
            "Aborting run to avoid mixed-model semantics."
        )
        raise


# ── Hyperbolic graph builder ──────────────────────────────────────────────────


def build_hyperbolic_graph(phase1_result: Phase1Output, logger: logging.Logger):
    """
    Construct a NetworkX graph from hyperbolic landmark coordinates.
    Nodes carry 'xyz' attribute for Phase 4 KD-tree lookup.
    Edges carry 'dist_H' (hyperbolic geodesic distance) for conductance weights.
    """
    import networkx as nx
    from scipy.spatial import KDTree

    landmarks = phase1_result.landmarks  # shape (n_landmarks, D)
    witnesses = phase1_result.witnesses  # shape (N, 3) — Euclidean

    # Reconstruct Euclidean centroids per landmark for xyz annotation
    lm_indices = phase1_result.landmark_indices
    landmark_xyz = {}
    for k in np.unique(lm_indices):
        mask = lm_indices == k
        landmark_xyz[k] = witnesses[mask].mean(axis=0)

    G = nx.Graph()
    n = len(landmarks)

    # Add nodes
    for i in range(n):
        G.add_node(i, xyz=landmark_xyz.get(i, np.zeros(3)))

    # Add edges: connect each landmark to its K nearest neighbours
    K = min(8, n - 1)
    tree = KDTree(landmarks)
    for i in range(n):
        dists, idxs = tree.query(landmarks[i], k=K + 1)
        for dist, j in zip(dists[1:], idxs[1:]):  # skip self
            if not G.has_edge(i, j):
                G.add_edge(i, j, dist_H=float(dist))

    logger.info(
        f"Built hyperbolic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G, landmark_xyz


# ── GLU697 validation ─────────────────────────────────────────────────────────


def test_glu697_validation(
    phase2_result: Phase2Output,
    phase4_result: List[Dict],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """
    GLU697 (rho=8, P-loop) validation — highest-priority open EGFR benchmark.
    Checks whether GLU697 appears in the doorway list and, if so, what its
    allosteric coupling strength to known effector sites is.

    This is a preregistered validation target. Do not modify the detection
    logic without updating the Validation Target Selection Protocol.
    """
    logger.info("=== GLU697 Validation ===")
    report = {
        "target": "GLU697",
        "expected_rho": 8,
        "location": "P-loop",
        "status": "NOT_FOUND",
        "details": {},
    }

    doorways = phase2_result.doorways
    glu697_hits = [
        d for d in doorways if "697" in str(d.id) or "GLU697" in str(d.id).upper()
    ]

    if not glu697_hits:
        logger.warning("GLU697 NOT detected in state-selective doorway list.")
        report["status"] = "NOT_FOUND"
        report["details"]["doorway_ids"] = [d.id for d in doorways[:10]]
    else:
        site = glu697_hits[0]
        rho = site.rho
        report["status"] = "FOUND"
        report["details"]["rho"] = rho
        report["details"]["rho_expected"] = 8
        report["details"]["rho_match"] = abs(rho - 8) < 2 if rho is not None else False
        report["details"]["epistemic"] = site.epistemic

        # Check Phase 4 coupling strength for GLU697
        coupling_hits = [
            c for c in phase4_result if "697" in str(c.get("doorway_node", ""))
        ]
        if coupling_hits:
            best = max(coupling_hits, key=lambda x: x.get("coupling_strength", 0))
            report["details"]["coupling_strength"] = best.get("coupling_strength")
            report["details"]["pathway_length"] = len(best.get("pathway", []))

        logger.info(
            f"GLU697 FOUND | rho={rho} (expected ~8) | "
            f"match={'YES' if report['details'].get('rho_match') else 'CHECK'}"
        )

    # Write validation report
    report_path = output_dir / "glu697_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"GLU697 validation report → {report_path}")

    return report


# ── Full pipeline ─────────────────────────────────────────────────────────────


def _run_source_leak_v4_pipeline(
    gdp_ensemble,
    gtp_ensemble,
    evidential_gnn,
    phase2_result,
    effector_sites,
    logger,
    results,
    internal_results,
):
    # ── Phase 1 v4 ───────────────────────────────────────────────────────
    p1_v4 = _import_phase("archive.phase1_v4_residue_hyperbolic", "Phase 1 v4")
    utils._log_phase_io(
        "Phase 1 v4",
        "input",
        {
            "gdp_atoms": utils._count_structure_atoms(gdp_ensemble),
            "gtp_atoms": utils._count_structure_atoms(gtp_ensemble),
            "phase2_doorways": len(phase2_result.doorways),
        },
        logger,
    )
    phase1_v4_result = utils._timed(
        "Phase 1 v4: Residue Hyperbolic Inference",
        lambda: (
            p1_v4.execute_phase_1_v4_residue_hyperbolic(
                gdp_ensemble,
                gtp_ensemble,
                evidential_gnn,
            )
        ),
        logger,
    )
    utils._log_phase_io("Phase 1 v4", "output", phase1_v4_result, logger)

    # ── Phase 3 v4 ───────────────────────────────────────────────────────
    p3_v4 = _import_phase("archive.phase3_v4_cone_persistence", "Phase 3 v4")
    utils._log_phase_io(
        "Phase 3 v4",
        "input",
        {
            "phase1_v4_keys": (
                list(phase1_v4_result.keys())
                if isinstance(phase1_v4_result, dict)
                else type(phase1_v4_result).__name__
            ),
            "phase2_doorways": len(phase2_result.doorways),
        },
        logger,
    )
    phase35_result = utils._timed(
        "Phase 3 v4: Cone-Space Persistence",
        lambda: (
            p3_v4.execute_phase_3_v4_cone_persistence(
                phase1_v4_result,
                asdict(phase2_result),
            )
        ),
        logger,
    )
    utils._assert_required_keys(
        "Phase 3 v4 output", phase35_result, ["lifted_sites"], logger
    )
    utils._log_phase_io("Phase 3 v4", "output", phase35_result, logger)
    results["phase3"] = {
        "mode": "source_leak_v4",
        "n_terminal_leaks": len(phase35_result.get("lifted_sites", [])),
        "barcode_length": None,
        "generators_stored": False,
    }
    results["phase35"] = {
        "n_lifted_sites": len(phase35_result.get("lifted_sites", [])),
        "method": phase35_result.get("method", "cone_depth_residue_persistence_v4"),
        "sites": [
            {
                "birth": s.get("birth"),
                "n_vertices": s.get("num_vertices", 1),
                "barycenter_xyz": (
                    s["barycenter_xyz"].tolist()
                    if isinstance(s.get("barycenter_xyz"), np.ndarray)
                    else s.get("barycenter_xyz")
                ),
            }
            for s in phase35_result.get("lifted_sites", [])
        ],
    }
    internal_results["phase1_v4_result"] = phase1_v4_result
    internal_results["phase35_result"] = phase35_result

    # ── Phase 4 v4 ───────────────────────────────────────────────────────
    p4_v4 = _import_phase("archive.phase4_v4_leak_flow", "Phase 4 v4")
    resolved_effector_sites = effector_sites
    if not resolved_effector_sites:
        graph_summary = phase35_result.get("graph_summary", {})
        residue_ids = graph_summary.get("residue_ids", [])
        if residue_ids:
            tail = min(10, len(residue_ids))
            resolved_effector_sites = list(
                range(len(residue_ids) - tail, len(residue_ids))
            )
            logger.warning(
                "No effector_sites provided; using fallback terminal graph indices: %s",
                resolved_effector_sites,
            )
        else:
            resolved_effector_sites = [0]
            logger.warning(
                "No effector_sites provided and no residue_ids available; using [0] fallback"
            )

    utils._log_phase_io(
        "Phase 4 v4",
        "input",
        {
            "phase35_lifted_sites": len(phase35_result.get("lifted_sites", [])),
            "effector_sites": resolved_effector_sites,
        },
        logger,
    )
    phase4_result = utils._timed(
        "Phase 4 v4: Source-First Leak Flow",
        lambda: (
            p4_v4.execute_phase_4_v4_leak_flow(
                phase35_result,
                resolved_effector_sites,
            )
        ),
        logger,
    )
    utils._log_phase_io("Phase 4 v4", "output", phase4_result, logger)
    results["phase4"] = {
        "mode": "source_leak_v4",
        "n_conductance_paths": len(phase4_result),
        "top_coupling": (
            _result_get(phase4_result[0], "coupling_strength") if phase4_result else None
        ),
        "top_doorway": _result_get(phase4_result[0], "doorway_node") if phase4_result else None,
    }
    internal_results["phase4_result"] = phase4_result
    return phase35_result, phase4_result


def _run_legacy_witness_pipeline(
    phase1_result,
    phase2_result,
    effector_sites,
    logger,
    results,
    internal_results,
    gnn_output=None,
    mode_label: str = "legacy_witness",
):
    # ── Phase 3 ───────────────────────────────────────────────────────────
    p3 = _import_phase("phase3_witness_persistence", "Phase 3")

    utils._log_phase_io(
        "Phase 3",
        "input",
        {
            "phase1_landmarks": len(phase1_result.landmarks),
            "phase2_doorways": len(phase2_result.doorways),
            "max_alpha_candidates": [20.0, 24.0, 30.0],
        },
        logger,
    )

    max_alpha_candidates = [20.0, 24.0, 30.0]
    phase3_result = None
    phase3_attempts = []
    selected_alpha = max_alpha_candidates[0]

    for alpha in max_alpha_candidates:
        phase3_input = Phase3Input(
            phase1_result=phase1_result,
            phase2_result=asdict(phase2_result),
            max_alpha=alpha,
        )
        candidate_result = utils._timed(
            f"Phase 3: Witness Complex & Persistence (max_alpha={alpha})",
            lambda phase3_input=phase3_input: p3.execute_phase_3_witness_persistence(
                phase3_input
            ),
            logger,
        )
        leak_count = len(candidate_result.terminal_leaks)
        phase3_attempts.append({"max_alpha": alpha, "terminal_leaks": leak_count})

        phase3_result = candidate_result
        selected_alpha = alpha
        if leak_count > 0:
            break

    results["phase3"] = {
        "mode": mode_label,
        "n_terminal_leaks": len(phase3_result.terminal_leaks),
        "barcode_length": len(phase3_result.barcode),
        "generators_stored": getattr(phase3_result, "generators", None) is not None,
        "max_alpha_used": selected_alpha,
        "max_alpha_attempts": phase3_attempts,
    }
    logger.info(
        f"Phase 3: {results['phase3']['n_terminal_leaks']} terminal leaks detected "
        f"(max_alpha={selected_alpha})"
    )

    if not phase3_result.terminal_leaks:
        logger.warning(
            "No terminal leaks (infinite H1 bars) found. "
            "Consider increasing max_alpha or checking witness graph construction."
        )
    phase3_output_view = {
        "barcode": phase3_result.barcode,
        "terminal_leaks": phase3_result.terminal_leaks,
        "h1_edges": phase3_result.h1_edges,
        "witness_mask": phase3_result.witness_mask,
        "landmarks": phase3_result.landmarks,
    }
    utils._assert_required_keys(
        "Phase 3 output", phase3_output_view, ["terminal_leaks", "barcode"], logger
    )
    utils._log_phase_io("Phase 3", "output", phase3_output_view, logger)
    internal_results["phase3_result"] = phase3_result

    # ── Phase 3.5 ─────────────────────────────────────────────────────────
    p35 = _import_phase("phase35_topological_lift", "Phase 3.5")

    utils._log_phase_io(
        "Phase 3.5",
        "input",
        {
            "phase3_terminal_leaks": len(phase3_result.terminal_leaks),
            "phase3_barcode_len": len(phase3_result.barcode),
        },
        logger,
    )

    phase35_input = Phase35Input(
        phase1_result=phase1_result,
        phase3_result=phase3_result,
    )
    phase35_result = utils._timed(
        "Phase 3.5: Topological Lift → 3D Coordinates",
        lambda: p35.execute_phase_35_topological_lift(phase35_input),
        logger,
    )
    results["phase35"] = {
        "n_lifted_sites": len(phase35_result.lifted_sites),
        "method": phase35_result.method,
        "sites": [
            {
                "birth": s.birth,
                "n_vertices": s.num_vertices,
                "barycenter_xyz": (
                    s.barycenter_xyz.tolist()
                    if isinstance(s.barycenter_xyz, np.ndarray)
                    else s.barycenter_xyz
                ),
            }
            for s in phase35_result.lifted_sites
        ],
    }
    logger.info(
        f"Phase 3.5: {results['phase35']['n_lifted_sites']} persistence bars lifted to 3D"
    )
    utils._assert_required_keys(
        "Phase 3.5 output", asdict(phase35_result), ["lifted_sites"], logger
    )
    utils._log_phase_io("Phase 3.5", "output", asdict(phase35_result), logger)
    internal_results["phase35_result"] = phase35_result

    # ── Phase 4 ───────────────────────────────────────────────────────────
    p4 = _import_phase("phase4_resistance_mapping", "Phase 4")

    gnn_nodes = 0
    if gnn_output is not None and "gdp_ca_coords" in gnn_output:
        gnn_nodes = len(gnn_output["gdp_ca_coords"])

    utils._log_phase_io(
        "Phase 4",
        "input",
        {
            "gnn_nodes": gnn_nodes,
            "phase2_doorways": len(phase2_result.doorways),
            "phase35_lifted_sites": len(phase35_result.lifted_sites),
            "effector_sites": effector_sites or [],
        },
        logger,
    )

    phase4_input = Phase4Input(
        hyperbolic_graph=None,
        phase2_result=asdict(phase2_result),
        phase35_result=phase35_result,
        effector_sites=effector_sites or [],
        landmark_coords=None,
    )
    phase4_result = utils._timed(
        "Phase 4: Effective Resistance & Conductance",
        lambda: p4.execute_phase_4_resistance_mapping(phase4_input, gnn_output=gnn_output),
        logger,
    )
    results["phase4"] = {
        "mode": mode_label,
        "n_conductance_paths": len(phase4_result),
        "top_coupling": (
            _result_get(phase4_result[0], "coupling_strength") if phase4_result else None
        ),
        "top_doorway": _result_get(phase4_result[0], "doorway_node") if phase4_result else None,
    }
    logger.info(
        f"Phase 4: {results['phase4']['n_conductance_paths']} allosteric pathways mapped | "
        f"top coupling = {results['phase4']['top_coupling']}"
    )
    utils._log_phase_io("Phase 4", "output", phase4_result, logger)
    internal_results["phase4_result"] = phase4_result
    return phase35_result, phase4_result


def _run_phase6a_virtual_screening(
    phase5_result: List[Dict],
    library_path: str,
    top_n_screen: int,
    logger: logging.Logger,
    results: Dict,
    internal_results: Dict,
) -> Optional[List[Dict]]:
    p6a = _import_phase("phase6a_virtual_screening", "Phase 6a")
    utils._log_phase_io(
        "Phase 6a",
        "input",
        {
            "phase5_pharmacophores": len(phase5_result),
            "library_path": library_path,
            "top_n": top_n_screen,
        },
        logger,
    )
    phase6a_result = utils._timed(
        "Phase 6a: Virtual Screening",
        lambda: p6a.execute_phase_6a_virtual_screening(
            phase5_result,
            library_path,
            top_n=top_n_screen,
        ),
        logger,
    )
    results["phase6a"] = {
        "n_hits": len(phase6a_result),
        "note": "Centroid pre-filter — not full pharmacophore feature matching",
    }
    utils._log_phase_io("Phase 6a", "output", phase6a_result, logger)
    logger.info(f"Phase 6a: {len(phase6a_result)} hits after geometric pre-filter")
    if not phase6a_result:
        logger.warning(
            "Phase 6a returned no hits. Check library path and pharmacophore centers."
        )
        results["phase6"] = {"status": "no_hits"}
        return None
    return phase6a_result


def _run_phase6b_binding_affinity(
    phase6a_result: List[Dict],
    protein_pdb: str,
    gtp_pdb: str,
    top_n_dock: int,
    logger: logging.Logger,
    results: Dict,
) -> List[Dict]:
    p6b = _import_phase("phase6b_binding_affinity", "Phase 6b")
    utils._log_phase_io(
        "Phase 6b",
        "input",
        {
            "phase6a_hits": len(phase6a_result),
            "protein_pdb": protein_pdb,
            "gtp_pdb": gtp_pdb,
            "top_n": top_n_dock,
        },
        logger,
    )
    phase6b_result = utils._timed(
        "Phase 6b: Binding Affinity Estimation",
        lambda: p6b.execute_phase_6b_binding_affinity(
            phase6a_result,
            protein_pdb,
            gtp_pdb_path=gtp_pdb,
            method="vina",
            top_n=top_n_dock,
        ),
        logger,
    )
    valid_affinities = [
        h["delta_G"] for h in phase6b_result if h.get("delta_G") is not None
    ]
    results["phase6b"] = {
        "n_docked": len(phase6b_result),
        "n_valid_affinities": len(valid_affinities),
        "best_delta_G": min(valid_affinities) if valid_affinities else None,
        "mean_delta_G": float(np.mean(valid_affinities)) if valid_affinities else None,
    }
    logger.info(
        f"Phase 6b: {len(valid_affinities)}/{len(phase6b_result)} valid ΔG values | "
        f"best = {results['phase6b']['best_delta_G']} kcal/mol"
    )
    utils._log_phase_io("Phase 6b", "output", phase6b_result, logger)
    return phase6b_result


def _run_phase6c_admet_filter(
    phase6b_result: List[Dict], logger: logging.Logger, results: Dict
) -> Optional[List[Dict]]:
    p6c = _import_phase("phase6c_admet_filter", "Phase 6c")
    utils._log_phase_io(
        "Phase 6c",
        "input",
        {
            "phase6b_ranked_hits": len(phase6b_result),
            "filters": ["lipinski_ro5", "herg", "cyp3a4", "ames"],
        },
        logger,
    )
    phase6c_result = utils._timed(
        "Phase 6c: ADMET Filter",
        lambda: p6c.execute_phase_6c_admet_filter(
            phase6b_result,
            filters=["lipinski_ro5", "herg", "cyp3a4", "ames"],
        ),
        logger,
    )
    passed_admet = phase6c_result["passed"]
    results["phase6c"] = {
        "total_input": phase6c_result["total_input"],
        "n_passed": len(passed_admet),
        "n_failed": len(phase6c_result["failed"]),
        "pass_rate": (
            round(len(passed_admet) / phase6c_result["total_input"], 3)
            if phase6c_result["total_input"] > 0
            else 0
        ),
    }
    logger.info(
        f"Phase 6c: {results['phase6c']['n_passed']}/{results['phase6c']['total_input']} "
        f"passed ADMET ({results['phase6c']['pass_rate']*100:.1f}%)"
    )
    utils._assert_required_keys(
        "Phase 6c output", phase6c_result, ["passed", "failed", "total_input"], logger
    )
    utils._log_phase_io("Phase 6c", "output", phase6c_result, logger)
    if not passed_admet:
        logger.warning("No compounds passed ADMET filters.")
        results["phase6d"] = {"status": "skipped", "reason": "no_admet_passed"}
        return None
    return passed_admet


def _run_phase6d_state_selectivity(
    passed_admet: List[Dict],
    gtp_pdb: str,
    protein_pdb: str,
    logger: logging.Logger,
    results: Dict,
):
    p6d = _import_phase("phase6d_state_selectivity", "Phase 6d")
    utils._log_phase_io(
        "Phase 6d",
        "input",
        {
            "phase6c_passed": len(passed_admet),
            "gtp_pdb": gtp_pdb,
            "gdp_pdb": protein_pdb,
            "selectivity_threshold": 1.3,
        },
        logger,
    )
    phase6d_result = utils._timed(
        "Phase 6d: State Selectivity Verification",
        lambda: p6d.execute_phase_6d_state_selectivity_check(
            passed_admet,
            gtp_pdb,
            selectivity_threshold=1.3,
            gdp_protein_pdb_path=protein_pdb,
        ),
        logger,
    )
    selective = phase6d_result["selective"]
    results["phase6d"] = {
        "n_selective": len(selective),
        "n_non_selective": len(phase6d_result["non_selective"]),
        "selectivity_threshold": phase6d_result["selectivity_threshold_used"],
        "top_candidates": [
            {
                "smiles": h.get("smiles", "")[:60],
                "delta_G_gdp": h.get("delta_G"),
                "delta_G_gtp": h.get("gtp_delta_G"),
                "selectivity_ratio": h.get("selectivity_ratio"),
                "pharmacophore_center": h.get("pharmacophore_center"),
            }
            for h in selective[:10]
        ],
    }
    logger.info(
        f"Phase 6d: {len(selective)} state-selective candidates | "
        f"{len(phase6d_result['non_selective'])} rejected (non-selective or failed)"
    )
    utils._assert_required_keys(
        "Phase 6d output",
        phase6d_result,
        ["selective", "non_selective", "selectivity_threshold_used"],
        logger,
    )
    utils._log_phase_io("Phase 6d", "output", phase6d_result, logger)


def _run_phase6(
    phase5_result: List[Dict],
    library_path: Optional[str],
    top_n_screen: int,
    top_n_dock: int,
    protein_pdb: str,
    gtp_pdb: str,
    logger: logging.Logger,
    results: Dict,
    internal_results: Dict,
):
    if library_path is None:
        logger.info("No library provided; skipping Phase 6 (screening/docking/ADMET).")
        return
    if not Path(library_path).exists():
        raise FileNotFoundError(f"Fragment library not found: {library_path}")

    phase6a_result = _run_phase6a_virtual_screening(
        phase5_result,
        library_path,
        top_n_screen,
        logger,
        results,
        internal_results,
    )
    if phase6a_result is None:
        return

    phase6b_result = _run_phase6b_binding_affinity(
        phase6a_result,
        protein_pdb,
        gtp_pdb,
        top_n_dock,
        logger,
        results,
    )

    passed_admet = _run_phase6c_admet_filter(phase6b_result, logger, results)
    if passed_admet is None:
        return

    _run_phase6d_state_selectivity(
        passed_admet,
        gtp_pdb,
        protein_pdb,
        logger,
        results,
    )


def run_dtie_full_pipeline(
    gdp_pdb: str,
    gtp_pdb: str,
    protein_pdb: str,
    library_path: Optional[str],
    checkpoint_path: Optional[str],
    output_dir: Path,
    logger: logging.Logger,
    n_landmarks: int = 200,
    top_n_screen: int = 500,
    top_n_dock: int = 100,
    effector_sites: Optional[List[int]] = None,
    return_internal: bool = False,
    curvature_c_override: Optional[float] = None,
    pipeline_mode: str = "source_leak_v4",
) -> Dict:
    """
    Execute all pipeline phases in sequence.
    Each phase result is passed explicitly to the next — no global state.
    """

    results: Dict[str, Any] = {}
    internal_results: Dict[str, Any] = {}

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not checkpoint_path:
        raise ValueError(
            "GNN checkpoint is required. Provide --checkpoint with a compatible trained model."
        )

    # DEPRECATED: load_calibrated_c is superseded by gnn_output.npz curvature_c.
    # We still call it as a fallback if the ingestion/GNN preamble fails.
    curvature_c = load_calibrated_c(checkpoint_path, logger)
    if curvature_c_override is not None:
        logger.info(
            f"Curvature override applied: c={curvature_c_override} (was {curvature_c})"
        )
        curvature_c = curvature_c_override

    results["provenance"] = utils.build_provenance(
        gdp_pdb=gdp_pdb,
        gtp_pdb=gtp_pdb,
        protein_pdb=protein_pdb,
        library_path=library_path,
        checkpoint_path=checkpoint_path,
        n_landmarks=n_landmarks,
        curvature_c=curvature_c,
        top_n_screen=top_n_screen,
        top_n_dock=top_n_dock,
        selectivity_threshold=1.3,
        effector_sites=effector_sites,
    )

    gdp_ensemble = load_structure_ensemble(gdp_pdb, "GDP", logger)
    gtp_ensemble = load_structure_ensemble(gtp_pdb, "GTP", logger)
    protein_atoms = load_protein_atoms(protein_pdb, logger)

    import torch as _torch

    evidential_gnn = load_evidential_gnn(
        checkpoint_path=checkpoint_path,
        logger=logger,
        device="cuda" if _torch.cuda.is_available() else "cpu",
        gdp_pdb_path=gdp_pdb,
        gdp_structure=gdp_ensemble._structure,
    )

    # ── Ingestion + GNN Runner Preamble ───────────────────────────────────────
    # Run ingestion for both GDP and GTP PDB files, then run the GNN to
    # produce gnn_output.npz as the single source of truth for all phases.
    logger.info("── Ingestion + GNN Runner Preamble ──")

    gdp_ingestion = utils._timed(
        "Ingestion: GDP",
        lambda: ingest_pdb(Path(gdp_pdb), output_dir, label="gdp"),
        logger,
    )
    gtp_ingestion = utils._timed(
        "Ingestion: GTP",
        lambda: ingest_pdb(Path(gtp_pdb), output_dir, label="gtp"),
        logger,
    )
    logger.info(
        f"Ingestion complete: GDP {len(gdp_ingestion['residue_ids'])} residues, "
        f"GTP {len(gtp_ingestion['residue_ids'])} residues"
    )

    _checkpoint_path = Path(checkpoint_path)
    gnn_output = utils._timed(
        "GNN Runner",
        lambda: run_gnn(
            gdp_ingestion=gdp_ingestion,
            gtp_ingestion=gtp_ingestion,
            checkpoint_path=_checkpoint_path,
            output_path=output_dir,
        ),
        logger,
    )

    # Source curvature_c from gnn_output.npz (supersedes load_calibrated_c)
    gnn_curvature_c = float(gnn_output["curvature_c"])
    if curvature_c_override is not None:
        logger.info(
            f"Curvature override active: c={curvature_c_override} "
            f"(gnn_output.npz had c={gnn_curvature_c})"
        )
        curvature_c = curvature_c_override
    else:
        logger.info(
            f"Using curvature_c={gnn_curvature_c} from gnn_output.npz "
            f"(load_calibrated_c returned {curvature_c})"
        )
        curvature_c = gnn_curvature_c

    # Update provenance with GNN-sourced curvature
    results["provenance"]["curvature_c"] = curvature_c
    results["provenance"]["curvature_source"] = "gnn_output.npz"

    internal_results["gdp_ingestion"] = gdp_ingestion
    internal_results["gtp_ingestion"] = gtp_ingestion
    internal_results["gnn_output"] = gnn_output

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    p1 = _import_phase("phase1_witness_embedding", "Phase 1")
    utils._log_phase_io(
        "Phase 1",
        "input",
        {
            "gdp_residues": len(gdp_ingestion["residue_ids"]),
            "n_landmarks": n_landmarks,
            "curvature_c": curvature_c,
        },
        logger,
    )
    phase1_result = utils._timed(
        "Phase 1: Witness Embedding",
        lambda: p1.execute_phase_1_witness_embedding(
            ingestion_data=gdp_ingestion,
            gnn_output=gnn_output,
            n_landmarks=n_landmarks,
        ),
        logger,
    )
    results["phase1"] = {
        "n_witnesses": len(phase1_result.witnesses),
        "n_landmarks": len(phase1_result.landmarks),
        "curvature_c": curvature_c,
    }
    utils._assert_required_keys(
        "Phase 1 output",
        asdict(phase1_result),
        ["witnesses", "landmarks", "landmark_indices"],
        logger,
    )
    utils._log_phase_io("Phase 1", "output", asdict(phase1_result), logger)
    internal_results["phase1_result"] = phase1_result

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    p2 = _import_phase("phase2_vulnerability_scan", "Phase 2")
    utils._log_phase_io(
        "Phase 2",
        "input",
        {
            "gdp_atoms": utils._count_structure_atoms(gdp_ensemble),
            "gtp_atoms": utils._count_structure_atoms(gtp_ensemble),
            "phase1_landmarks": len(phase1_result.landmarks),
        },
        logger,
    )
    phase2_input = Phase2Input(
        gdp_ensemble=gdp_ensemble,
        gtp_ensemble=gtp_ensemble,
        evidential_gnn=evidential_gnn,
    )
    phase2_result = utils._timed(
        "Phase 2: Differential Vulnerability Scan",
        lambda: p2.execute_phase_2_vulnerability_scan(phase2_input, gnn_output=gnn_output),
        logger,
    )
    results["phase2"] = {
        "n_doorways": len(phase2_result.doorways),
        "n_constitutive": len(phase2_result.constitutive),
        "doorway_ids": [d.id for d in phase2_result.doorways],
    }
    logger.info(
        f"Phase 2: {results['phase2']['n_doorways']} state-selective doorways | "
        f"{results['phase2']['n_constitutive']} constitutive (deprioritised)"
    )
    utils._assert_required_keys(
        "Phase 2 output", asdict(phase2_result), ["doorways"], logger
    )
    utils._log_phase_io("Phase 2", "output", asdict(phase2_result), logger)

    if not phase2_result.doorways:
        logger.warning(
            "No state-selective doorways found. Check TAU calibration and structure quality."
        )
    internal_results["phase2_result"] = phase2_result

    # ── Phases 3 & 4 (Mode-dependent) ─────────────────────────────────────────
    if pipeline_mode == "source_leak_v4":
        logger.info(
            "source_leak_v4 uses the current Phase 3/3.5/4 pipeline with source_leak_scoring integration."
        )
        phase35_result, phase4_result = _run_legacy_witness_pipeline(
            phase1_result,
            phase2_result,
            effector_sites,
            logger,
            results,
            internal_results,
            gnn_output=gnn_output,
            mode_label="source_leak_v4",
        )
    else:
        phase35_result, phase4_result = _run_legacy_witness_pipeline(
            phase1_result,
            phase2_result,
            effector_sites,
            logger,
            results,
            internal_results,
            gnn_output=gnn_output,
            mode_label="legacy_witness",
        )

    # ── Phase 5 ───────────────────────────────────────────────────────────────
    p5 = _import_phase("phase5_pharmacophore", "Phase 5")
    utils._log_phase_io(
        "Phase 5",
        "input",
        {
            "phase35_lifted_sites": (
                len(phase35_result.lifted_sites)
                if isinstance(phase35_result, Phase35Output)
                else len(phase35_result.get("lifted_sites", []))
            ),
            "phase4_paths": len(phase4_result),
            "protein_atoms": len(protein_atoms),
        },
        logger,
    )
    phase5_input = Phase5Input(
        phase35_result=(
            asdict(phase35_result)
            if isinstance(phase35_result, Phase35Output)
            else phase35_result
        ),
        phase4_result=phase4_result,
        protein_atoms=protein_atoms,
        protein_pdb_path=protein_pdb,
        gnn_output=gnn_output,
    )
    phase5_raw = utils._timed(
        "Phase 5: Pharmacophore Generation",
        lambda: p5.execute_phase_5_pharmacophore_generation(phase5_input),
        logger,
    )

    if hasattr(phase5_raw, "pharmacophores") and not isinstance(
        phase5_raw, (dict, list, tuple)
    ):
        phase5_result = [asdict(p) for p in phase5_raw.pharmacophores]
    elif isinstance(phase5_raw, dict):
        phase5_result = list(phase5_raw.get("pharmacophores", []))
    else:
        phase5_result = list(phase5_raw)

    phase5_result = [
        asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in phase5_result
    ]

    results["phase5"] = {
        "n_pharmacophores": len(phase5_result),
        "pharmacophores": phase5_result,
    }
    logger.info(
        f"Phase 5: {results['phase5']['n_pharmacophores']} pharmacophores generated"
    )
    utils._log_phase_io("Phase 5", "output", phase5_result, logger)
    internal_results["phase5_result"] = phase5_result

    # ── Phase 6 ───────────────────────────────────────────────────────────────
    _run_phase6(
        phase5_result,
        library_path,
        top_n_screen,
        top_n_dock,
        protein_pdb,
        gtp_pdb,
        logger,
        results,
        internal_results,
    )

    return (results, internal_results) if return_internal else results


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Eidetix Bio DTIE v3.0 — Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    parser.add_argument("--gdp", required=True, help="GDP (inactive) state PDB file")
    parser.add_argument(
        "--gtp",
        required=True,
        help="GTP (active) state PDB file for differential scan and Phase 6d",
    )
    parser.add_argument(
        "--protein",
        required=True,
        help="Protein PDB for Phase 5 pocket analysis (typically same as --gdp)",
    )

    # Optional inputs
    parser.add_argument(
        "--library",
        default=None,
        help="SDF compound library for Phase 6a–6d (if omitted, pipeline runs Phases 1–5 only)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="GOSP-GNN model checkpoint (.ckpt) containing curvature_c",
    )
    parser.add_argument(
        "--effectors",
        nargs="+",
        type=int,
        default=None,
        help="Known effector site landmark indices for Phase 4 conductance mapping",
    )

    # Pipeline controls
    parser.add_argument(
        "--n-landmarks",
        type=int,
        default=200,
        help="Number of landmark witness points for Phase 1 (default: 200)",
    )
    parser.add_argument(
        "--curvature-c",
        type=float,
        default=None,
        help="Override Poincaré curvature c (overrides checkpoint value; default: from checkpoint or 1.0)",
    )
    parser.add_argument(
        "--top-n-screen",
        type=int,
        default=500,
        help="Max hits from Phase 6a to carry into Phase 6b (default: 500)",
    )
    parser.add_argument(
        "--top-n-dock",
        type=int,
        default=100,
        help="Max compounds to dock in Phase 6b (default: 100)",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["legacy_witness", "source_leak_v4"],
        default="source_leak_v4",
        help="Pipeline topology mode (default: source_leak_v4)",
    )

    # Validation flags
    parser.add_argument(
        "--test-glu697",
        action="store_true",
        help="Run GLU697 (P-loop) validation after Phase 2 and 4",
    )

    # Output
    parser.add_argument(
        "--output",
        default="dtie_results",
        help="Output directory (default: dtie_results)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    logger = utils.setup_logging(output_dir, verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("  Eidetix Bio — Dynamic Topology Inference Engine v3.0")
    logger.info("=" * 70)
    logger.info(f"  GDP:        {args.gdp}")
    logger.info(f"  GTP:        {args.gtp}")
    logger.info(f"  Protein:    {args.protein}")
    logger.info(f"  Library:    {args.library or '(none — Phases 1–5 only)'}")
    logger.info(f"  Checkpoint: {args.checkpoint or '(required)'}")
    logger.info(f"  Mode:       {args.pipeline_mode}")
    logger.info(f"  Output:     {output_dir}")
    logger.info("=" * 70)

    # Input validation
    for flag, path in [
        ("--gdp", args.gdp),
        ("--gtp", args.gtp),
        ("--protein", args.protein),
    ]:
        if not Path(path).exists():
            logger.error(f"{flag} file not found: {path}")
            sys.exit(1)
    if args.library and not Path(args.library).exists():
        logger.error(f"--library file not found: {args.library}")
        sys.exit(1)
    if not args.checkpoint:
        logger.error("--checkpoint is required: pipeline cannot run without GNN checkpoint")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        logger.error(f"--checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    t_start = time.time()

    try:
        pipeline_result = run_dtie_full_pipeline(
            gdp_pdb=args.gdp,
            gtp_pdb=args.gtp,
            protein_pdb=args.protein,
            library_path=args.library,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir,
            logger=logger,
            n_landmarks=args.n_landmarks,
            top_n_screen=args.top_n_screen,
            top_n_dock=args.top_n_dock,
            effector_sites=args.effectors,
            return_internal=args.test_glu697,
            curvature_c_override=args.curvature_c,
            pipeline_mode=args.pipeline_mode,
        )

        if args.test_glu697:
            results, internal = pipeline_result
        else:
            results = pipeline_result
            internal = {}

        # GLU697 validation (runs post Phase 2 + 4 if requested)
        if args.test_glu697:
            if "phase2_result" not in internal or "phase4_result" not in internal:
                logger.error("--test-glu697 requires Phases 2 and 4 to complete first.")
            else:
                glu_report = test_glu697_validation(
                    phase2_result=internal["phase2_result"],
                    phase4_result=internal["phase4_result"],
                    output_dir=output_dir,
                    logger=logger,
                )
                results["glu697_validation"] = glu_report

        # Save all results
        utils.save_results(results, output_dir, logger)

        # Summary
        elapsed = time.time() - t_start
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info(f"  Total runtime: {elapsed:.1f}s")
        if "phase6d" in results and isinstance(results["phase6d"], dict):
            n = results["phase6d"].get("n_selective", 0)
            logger.info(f"  State-selective candidates: {n}")
        if "phase5" in results:
            logger.info(
                f"  Pharmacophores generated:   {results['phase5']['n_pharmacophores']}"
            )
        logger.info(f"  Results written to:         {output_dir}")
        logger.info("")
        logger.info("  Remaining open items:")
        logger.info(
            "    - Effector sites: supply known effector residue indices for target-specific conductance mapping"
        )
        logger.info("    - GNN checkpoint: train GOSP-GNN on labelled allosteric data")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
