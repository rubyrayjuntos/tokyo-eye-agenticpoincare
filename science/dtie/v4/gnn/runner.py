# Migrated from: new (Phase 3 scaffold) on 2026-05-27
"""V4 GNN Runner — wraps GOSPConeMapper-v4 for governed inference.

This runner:
1. Loads the v4 model checkpoint
2. Runs inference on a prepared graph
3. Returns results in the standard GNNInferenceResult format
4. Does NOT write to the database (that's the adapter's job)

The actual GOSPConeMapper-v4 model will be migrated from:
  SRC_VIZ/src/components/TokyoEyesv4/Gnnv4.py

Per AGENTS.md rules:
- Science core remains framework-free (no FastAPI, no DB writes)
- v4 code stays strictly separate from v3
- All outputs go through the Normalizer via adapters

See: docs/audit/DEEP_AUDIT_PHASE_0.1.md for v4 model details
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput

logger = logging.getLogger(__name__)

# Default checkpoint location (configurable)
DEFAULT_CHECKPOINT = "checkpoints/tokyo_eyes_v4.pt"


class V4GNNRunner:
    """Runner for GOSPConeMapper-v4 inference.

    Implements the GNNRunner protocol from science.dtie.common.interfaces.

    Usage:
        runner = V4GNNRunner(checkpoint_path="path/to/tokyo_eyes_v4.pt")
        result = await runner.run_inference(structure_id="4obe", graph_data=data)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        curvature_override: float | None = None,
    ):
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._device = device
        self._curvature_override = curvature_override
        self._model = None
        self._loaded = False

    @property
    def model_version(self) -> str:
        return "GOSPConeMapper-v4"

    @property
    def space_type(self) -> str:
        return "hyperbolic"

    def load_model(self) -> None:
        """Load the v4 model checkpoint.

        This is separated from __init__ to allow lazy loading and
        to keep the import of torch optional until actually needed.
        """
        try:
            import torch
            from torch_geometric.data import Data  # noqa: F401

            from science.dtie.v4.gnn.model import GOSPConeMapper

            checkpoint = Path(self._checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"V4 checkpoint not found: {self._checkpoint_path}"
                )

            self._model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
            state_dict = torch.load(checkpoint, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model.to(self._device)

            logger.info("V4 model loaded from %s", self._checkpoint_path)
            self._loaded = True

        except ImportError as e:
            raise ImportError(
                "torch and torch_geometric required for GNN inference. "
                "Install with: pip install torch torch-geometric"
            ) from e

    async def run_inference(
        self,
        structure_id: str,
        graph_data: Any,
        checkpoint_path: str | None = None,
    ) -> GNNInferenceResult:
        """Run v4 GNN inference on a prepared graph.

        Args:
            structure_id: Canonical structure_id.
            graph_data: PyG Data object with:
                - x: node features [N, 4] (rho, tau_flag, ss_type, sasa)
                - edge_index: [2, E] graph connectivity
                - edge_attr: [E, d] edge features
                - Optional: clustering, batch
            checkpoint_path: Override checkpoint (uses default if None).

        Returns:
            GNNInferenceResult with per-node outputs.
        """
        import torch

        if not self._loaded:
            self.load_model()

        # Ensure clustering is precomputed
        if not hasattr(graph_data, 'clustering') or graph_data.clustering is None:
            from science.dtie.v4.gnn.model import precompute_clustering
            graph_data = precompute_clustering(graph_data)

        # Run inference
        with torch.no_grad():
            output = self._model(graph_data)

        return self._extract_results(output, graph_data, structure_id)

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        """Convert raw model output dict to GNNInferenceResult.

        This method handles the v4-specific output format:
        - projections: [N, 64] Euclidean
        - hyp_projections: [N, 2] native disc
        - x_hyp: [N, hidden_dim] full hyperbolic
        - x_routed_hyp: [N, hidden_dim] post-MoE hyperbolic
        - cone_depth, cone_width: [N]
        - epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty: [N]
        - expert_weights: [N, n_experts]
        """
        import torch

        num_nodes = output["projections"].shape[0]
        chain_ids = getattr(graph_data, 'chain_ids', ['A'] * num_nodes)
        residue_indices = getattr(graph_data, 'residue_indices', list(range(1, num_nodes + 1)))

        nodes = []
        for i in range(num_nodes):
            node = GNNNodeOutput(
                residue_index=int(residue_indices[i]),
                chain_label=str(chain_ids[i]),
                input_features=graph_data.x[i].cpu().numpy(),
                projections=output["projections"][i].cpu().numpy(),
                cone_depth=float(output["cone_depth"][i]),
                cone_width=float(output["cone_width"][i]),
                epistemic_uncertainty=float(output["epistemic_uncertainty"][i]),
                aleatoric_uncertainty=float(output["aleatoric_uncertainty"][i]),
                total_uncertainty=float(output["total_uncertainty"][i]),
                x_hyp=output["x_hyp"][i].cpu().numpy(),
                x_routed_hyp=output["x_routed_hyp"][i].cpu().numpy(),
                hyp_projections=output["hyp_projections"][i].cpu().numpy(),
                expert_weights=output["expert_weights"][i].cpu().numpy(),
            )
            nodes.append(node)

        # Extract learned curvature
        curvature = self._curvature_override
        if curvature is None and hasattr(self._model, 'log_c'):
            import torch.nn.functional as F
            curvature = float(F.softplus(self._model.log_c).item())

        return GNNInferenceResult(
            structure_id=structure_id,
            model_version=self.model_version,
            checkpoint_path=self._checkpoint_path,
            nodes=nodes,
            curvature=curvature or 1.0,
            embedding_dim=output["x_hyp"].shape[1],
            space_type="hyperbolic",
            metadata={
                "device": self._device,
                "num_nodes": num_nodes,
                "num_edges": graph_data.edge_index.shape[1] if hasattr(graph_data, 'edge_index') else 0,
            },
        )
