"""V6.5 GNN Runner — wraps GOSPConeMapperV65 for governed inference."""

from __future__ import annotations

from science.dtie.v6.gnn.runner import V6GNNRunner


class V65GNNRunner(V6GNNRunner):
    """Same inference path as v6; loads GOSPConeMapperV65 weights."""

    @property
    def model_version(self) -> str:
        return "GOSPConeMapper-v6.5"

    def _load_model_sync(self) -> None:
        """Load the v6.5 model checkpoint (synchronous, CPU-bound)."""
        try:
            import torch
            from pathlib import Path

            from science.contracts.model_registry import resolve_checkpoint_file
            from science.dtie.v65.gnn.model import (
                GOSPConeMapperV65,
                infer_legacy_disc_projection_from_checkpoint,
                infer_v65_model_kwargs,
                load_v65_state_dict,
            )

            resolved = resolve_checkpoint_file(self._checkpoint_path)
            checkpoint = resolved if resolved is not None else Path(self._checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"V6.5 checkpoint not found: {self._checkpoint_path}"
                )

            checkpoint_data = torch.load(
                checkpoint, map_location=self._device, weights_only=False
            )

            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data
                checkpoint_data = {}

            kwargs = infer_v65_model_kwargs(
                state_dict,
                checkpoint_data.get("architecture") if isinstance(checkpoint_data, dict) else None,
                checkpoint_data.get("training_config") if isinstance(checkpoint_data, dict) else None,
            )
            kwargs["legacy_disc_projection"] = infer_legacy_disc_projection_from_checkpoint(
                training_config=checkpoint_data.get("training_config")
                if isinstance(checkpoint_data, dict)
                else None,
                metrics=checkpoint_data.get("metrics") if isinstance(checkpoint_data, dict) else None,
                override=self._legacy_disc_projection_override,
            )

            node_dim = int(kwargs.pop("node_dim", 4))
            self._model = GOSPConeMapperV65(node_dim=node_dim, **kwargs)

            incompatible = load_v65_state_dict(self._model, state_dict)
            missing = getattr(
                incompatible, "missing_keys", incompatible[0] if isinstance(incompatible, tuple) else []
            )
            unexpected = getattr(
                incompatible,
                "unexpected_keys",
                incompatible[1] if isinstance(incompatible, tuple) else [],
            )
            if missing:
                import logging

                logging.getLogger(__name__).info(
                    "V6.5 model: %d missing keys (new modules init from scratch)",
                    len(missing),
                )
            if unexpected:
                import logging

                logging.getLogger(__name__).warning(
                    "V6.5 model: %d unexpected keys ignored", len(unexpected)
                )

            self._model.eval()
            self._model.to(self._device)
            self._checkpoint_path = str(checkpoint)

        except ImportError as e:
            raise ImportError(
                f"Failed to import V6.5 GNN dependencies: {e}. "
                "Ensure torch and torch_geometric are installed."
            ) from e
