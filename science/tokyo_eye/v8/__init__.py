"""TokyoEye-v8 isolated lineage package.

Self-contained under ``science/tokyo_eye/v8/``. Does **not** import v7 model,
v66 trainers, ``biology_graph``, or ingest/Normalizer paths.

Shared platform only: torch / numpy / PyG / Bio.PDB / MLflow / science container.
"""

from science.tokyo_eye.v8.attention import HyperbolicGraphAttention
from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    EpsilonGreedySchedule,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
    train_v8_step,
)
from science.tokyo_eye.v8.equiformer_frontend import (
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    load_weight_map,
)
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset, load_structure_batch
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.moe import TopologyAwareHardMoE
from science.tokyo_eye.v8.r0_r5_graph import (
    build_r0_r5_graph,
    load_r0_r5_from_pdb,
)

__all__ = [
    "CurriculumRadiusController",
    "EpsilonGreedySchedule",
    "GumbelTemperatureSchedule",
    "HyperbolicGraphAttention",
    "PoincareDiagnosticsEngine",
    "StubEquiformerFrontend",
    "TokyoEyeCuratedDataset",
    "TokyoEyeV8WithFrontend",
    "TokyoEyesHyperbolicV8",
    "TopologyAwareHardMoE",
    "build_r0_r5_graph",
    "load_r0_r5_from_pdb",
    "load_structure_batch",
    "load_weight_map",
    "train_v8_step",
]
