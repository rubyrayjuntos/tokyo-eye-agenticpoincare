# Migrated from: SRC_DEM/DTIE_GNN_ORCHESTRATION/contracts.py on 2026-05-27
"""
contracts.py
============
Eidetix Bio — Dynamic Topology Inference Engine v3.0
Data contracts for the DTIE pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem


@dataclass
class Phase1Input:
    """Input for Phase 1: Witness Embedding."""

    gdp_ensemble: Any
    n_landmarks: int
    curvature_c: float


@dataclass
class Phase1Output:
    """Output of Phase 1: Witness Embedding."""

    witnesses: np.ndarray
    landmarks: np.ndarray
    landmarks_euclidean: np.ndarray
    landmark_indices: np.ndarray
    landmark_to_residue_map: Dict[int, Any]
    mapping_function: str
    curvature_c: float
    residue_ids: List[Any]


@dataclass
class Doorway:
    """A single state-selective doorway."""

    id: Any
    rho: float
    aleatoric: float
    epistemic: float
    state: str
    status: str
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0


@dataclass
class Phase2Input:
    """Input for Phase 2: Vulnerability Scan."""

    gdp_ensemble: Any
    gtp_ensemble: Any
    evidential_gnn: Any


@dataclass
class Phase2Output:
    """Output of Phase 2: Vulnerability Scan."""

    doorways: List[Doorway]
    constitutive: List[Dict[str, Any]]
    gdp_all: List[Dict[str, Any]]
    gtp_all: List[Dict[str, Any]]


@dataclass
class LiftedSite:
    """A single lifted site from the topological lift."""

    leak: Any
    barycenter_xyz: np.ndarray
    cycle_vertices: List[int]
    birth: float
    num_vertices: int
    generator_simplices: List[List[int]]


@dataclass
class Phase3Input:
    """Input for Phase 3: Witness Persistence."""

    phase1_result: Phase1Output
    phase2_result: Phase2Output
    max_alpha: float


@dataclass
class Phase3Output:
    """Output of Phase 3: Witness Persistence."""

    barcode: Any
    terminal_leaks: List[Any]
    simplex_tree: Any
    h1_edges: List[List[int]]
    generators: Any
    witness_mask: List[int]
    landmarks: np.ndarray


@dataclass
class Phase35Input:
    """Input for Phase 3.5: Topological Lift."""

    phase1_result: Phase1Output
    phase3_result: Phase3Output


@dataclass
class Phase35Output:
    """Output of Phase 3.5: Topological Lift."""

    lifted_sites: List[LiftedSite]
    method: str


@dataclass
class SpectralAnalysis:
    """Spectral graph analysis results from Phase 4."""

    lambda_2: float
    fiedler_vector: np.ndarray
    hinge_residues: List[int]
    spectral_embedding: np.ndarray  # [N, 2]
    lambda_n_minus1: float
    lambda_n: float
    fragment_scores: List[Dict[str, Any]]


@dataclass
class Phase4Input:
    """Input for Phase 4: Resistance Mapping."""

    phase35_result: Phase35Output
    effector_sites: List[int]
    hyperbolic_graph: Any | None = None
    phase2_result: Phase2Output | None = None
    landmark_coords: Dict[int, np.ndarray] | None = None


@dataclass
class Phase4Output:
    """Output of Phase 4: Resistance Mapping."""

    doorway_node: Any
    target: Any
    coupling_strength: float
    r_eff: float
    pathway: List[Any]
    doorway_xyz: List[float]
    source_score: Optional[float] = None
    propagation_score: Optional[float] = None
    candidate_priority: Optional[float] = None
    fragility_index: Optional[float] = None
    uncertainty_robustness: Optional[float] = None
    persistence_span: Optional[float] = None
    genotype_persistence: Optional[float] = None
    state_stability: Optional[float] = None
    contrast_tier: Optional[str] = None
    spectral: Optional[SpectralAnalysis] = None


@dataclass
class Pharmacophore:
    """A single pharmacophore model."""

    center_xyz: List[float]
    druggability_score: float
    volume: float
    feasible_volumes: List[np.ndarray]
    atom_type_constraints: Dict[str, int]
    hbond_donors: List[List[float]]
    hbond_acceptors: List[List[float]]
    charge_complementarity: Dict[str, Any]
    pocket_source: str
    allosteric_coupling_strength: float
    action: str


@dataclass
class Phase5Input:
    """Input for Phase 5: Pharmacophore Generation."""

    phase35_result: Phase35Output
    phase4_result: List[Phase4Output]
    protein_atoms: List[Any]
    protein_pdb_path: str
    gnn_output: Optional[Dict[str, Any]] = None


@dataclass
class Phase5Output:
    """Output of Phase 5: Pharmacophore Generation."""

    pharmacophores: List[Pharmacophore]


@dataclass
class ScreeningHit:
    """A single hit from virtual screening."""

    smiles: str
    mol: Chem.Mol
    pharm_score: float
    mw: float
    logp: float
    pharmacophore_idx: int
    pharmacophore_center: List[float]


@dataclass
class Phase6aInput:
    """Input for Phase 6a: Virtual Screening."""

    phase5_result: Phase5Output
    library_path: str
    top_n: int


@dataclass
class Phase6aOutput:
    """Output of Phase 6a: Virtual Screening."""

    hits: List[ScreeningHit]


@dataclass
class DockingResult(ScreeningHit):
    """A single docking result."""

    docking_method: str
    delta_G: float
    gtp_delta_G: float


@dataclass
class Phase6bInput:
    """Input for Phase 6b: Binding Affinity Estimation."""

    phase6a_result: Phase6aOutput
    protein_pdb: str
    gtp_pdb_path: str
    method: str
    top_n: int


@dataclass
class Phase6bOutput:
    """Output of Phase 6b: Binding Affinity Estimation."""

    docking_results: List[DockingResult]


@dataclass
class AdmetResult:
    """A single ADMET result."""

    smiles: str
    admet_fail_reasons: List[str]


@dataclass
class Phase6cInput:
    """Input for Phase 6c: ADMET Filter."""

    phase6b_result: Phase6bOutput
    filters: List[str]


@dataclass
class Phase6cOutput:
    """Output of Phase 6c: ADMET Filter."""

    passed: List[DockingResult]
    failed: List[AdmetResult]
    total_input: int
    filters: List[str]


@dataclass
class SelectivityResult(DockingResult):
    """A single selectivity result."""

    selectivity_ratio: float


@dataclass
class Phase6dInput:
    """Input for Phase 6d: State Selectivity."""

    passed_admet: Phase6cOutput
    gtp_pdb: str
    selectivity_threshold: float
    gdp_protein_pdb_path: str


@dataclass
class Phase6dOutput:
    """Output of Phase 6d: State Selectivity Verification."""

    selective: List[SelectivityResult]
    non_selective: List[SelectivityResult]
    undetermined: List[DockingResult]
    selectivity_threshold_used: float
