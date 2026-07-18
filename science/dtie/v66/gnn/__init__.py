"""V6.6 GNN — GOSPConeMapperV66 (fork of v6 for feeler cold-start experiments)."""

from science.dtie.v66.gnn.model import (
    GOSPConeMapperV66,
    infer_v66_model_kwargs,
    load_v66_state_dict,
    verify_v66_checkpoint,
)
from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel
from science.dtie.v66.gnn.equivariant_conv_thermo import EquivariantConvThermo

__all__ = [
    "EquivariantConvMultiRel",
    "EquivariantConvThermo",
    "GOSPConeMapperV66",
    "infer_v66_model_kwargs",
    "load_v66_state_dict",
    "verify_v66_checkpoint",
]
