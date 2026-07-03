"""V6 GNN — GOSPConeMapperV6 with hyperbolic prototype MoE specialization."""

from science.dtie.v6.gnn.model import (
    GOSPConeMapperV6,
    TopologicalMoEGateV6,
    infer_v6_model_kwargs,
    load_v6_state_dict,
    verify_v6_checkpoint,
)
from science.dtie.v6.gnn.runner import V6GNNRunner

__all__ = [
    "GOSPConeMapperV6",
    "TopologicalMoEGateV6",
    "V6GNNRunner",
    "infer_v6_model_kwargs",
    "load_v6_state_dict",
    "verify_v6_checkpoint",
]
