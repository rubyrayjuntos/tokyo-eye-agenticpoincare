# Migrated from: new (Phase 3 scaffold) on 2026-05-27
"""V3 GNN Runner — wraps GOSPConeMapper-v3 for governed inference.

The actual GOSPConeMapper-v3 model will be migrated from:
  SRC_DEM/DTIE_GNN_ORCHESTRATION/Gnnv3.py

Key differences from v4 (per Deep Audit):
- Euclidean-only output (no native hyperbolic projections)
- MoE experts receive Euclidean features (not tangent space)
- Uncertainty head is simpler (epistemic only, no aleatoric)
- Checkpoint: robust_experts.pt (incompatible with v4)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints/robust_experts.pt"


class V3GNNRunner:
    """Runner for GOSPConeMapper-v3 inference.

    Implements the GNNRunner protocol.

    Usage:
        runner = V3GNNRunner(checkpoint_path="path/to/robust_experts.pt")
        result = await runner.run_inference(structure_id="4obe", graph_data=data)
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ):
        self._checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._device = device
        self._model = None
        self._loaded = False

    @property
    def model_version(self) -> str:
        return "GOSPConeMapper-v3"

    @property
    def space_type(self) -> str:
        return "euclidean"

    def load_model(self) -> None:
        """Load the v3 model checkpoint."""
        try:
            import torch

            from science.dtie.v3.gnn.model import GOSPConeMapper

            checkpoint = Path(self._checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"V3 checkpoint not found: {self._checkpoint_path}"
                )

            self._model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
            state_dict = torch.load(checkpoint, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._model.to(self._device)

            logger.info("V3 model loaded from %s", self._checkpoint_path)
            self._loaded = True

        except ImportError as e:
            raise ImportError(
                "torch and torch_geometric required for GNN inference."
            ) from e

    async def run_inference(
        self,
        structure_id: str,
        graph_data: Any,
        checkpoint_path: str | None = None,
    ) -> GNNInferenceResult:
        """Run v3 GNN inference on a prepared graph.

        Args:
            structure_id: Canonical structure_id.
            graph_data: PyG Data object with node features and edges.
            checkpoint_path: Override checkpoint.

        Returns:
            GNNInferenceResult with per-node outputs (Euclidean only).
        """
        import torch

        if not self._loaded:
            self.load_model()

        # Ensure clustering is precomputed
        if not hasattr(graph_data, 'clustering') or graph_data.clustering is None:
            from science.dtie.v3.gnn.model import precompute_clustering
            graph_data = precompute_clustering(graph_data)

        with torch.no_grad():
            output = self._model(graph_data)

        return self._extract_results(output, graph_data, structure_id)

    def _extract_results(
        self,
        output: dict[str, Any],
        graph_data: Any,
        structure_id: str,
    ) -> GNNInferenceResult:
        """Convert raw v3 model output to GNNInferenceResult.

        V3 output format (simpler than v4):
        - projections: [N, 64] Euclidean
        - cone_depth, cone_width: [N]
        - epistemic_uncertainty: [N] (no aleatoric)
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
                # v3 does NOT produce aleatoric or hyperbolic outputs
                expert_weights=output["expert_weights"][i].cpu().numpy(),
            )
            nodes.append(node)

        return GNNInferenceResult(
            structure_id=structure_id,
            model_version=self.model_version,
            checkpoint_path=self._checkpoint_path,
            nodes=nodes,
            curvature=None,  # v3 doesn't use learned curvature
            embedding_dim=64,
            space_type="euclidean",
        )
