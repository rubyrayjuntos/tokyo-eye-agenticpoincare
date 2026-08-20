"""TokyoEye runner adapter to governed GNNInferenceResult."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.asyncio
async def test_tokyoeye_runner_maps_forward_output_to_gnn_result() -> None:
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from science.tokyo_eye.v8.runner import TokyoEyeV8Runner

    class _Spine:
        c = 1.25

    class _System:
        spine = _Spine()

        def __call__(self, coords, edge_index, edge_type):
            n = coords.shape[0]
            z = torch.full((n, 4), 0.05, dtype=torch.float32, device=coords.device)
            h = torch.arange(n * 6, dtype=torch.float32, device=coords.device).reshape(n, 6)
            routing = torch.zeros((n, 4), dtype=torch.float32, device=coords.device)
            routing[:, 1] = 1.0
            return {
                "z_hyp": z,
                "h_euc": h,
                "evidence": torch.tensor(
                    [[0.0, 1.0, 2.0, 0.5]] * n,
                    dtype=torch.float32,
                    device=coords.device,
                ),
                "mechanism_score": torch.ones(n, dtype=torch.float32, device=coords.device),
                "moe_aux": {
                    "routing": routing,
                    "load": routing.mean(dim=0),
                    "hard": "argmax",
                },
            }

    data = Data(
        x=torch.tensor(
            [
                [0.1, 1.0, 0.0, 12.0],
                [0.2, 0.0, 0.5, 20.0],
                [0.3, 1.0, 1.0, 30.0],
            ],
            dtype=torch.float32,
        ),
        ca_coords=torch.tensor(
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[3.8, 0.0, 0.0, 3.8], [-3.8, 0.0, 0.0, 3.8], [3.8, 0.0, 0.0, 3.8], [-3.8, 0.0, 0.0, 3.8]],
            dtype=torch.float32,
        ),
    )
    data.sasa = data.x[:, 3]
    data.chain_ids = ["A", "A", "A"]
    data.residue_indices = [1, 2, 3]

    runner = TokyoEyeV8Runner(checkpoint_path="unused.pt", device="cpu")
    runner._system = _System()  # noqa: SLF001 - focused adapter test
    runner._loaded = True  # noqa: SLF001 - avoid heavy checkpoint load

    result = await runner.run_inference("4obe", data)

    assert result.model_version == "TokyoEye@champion"
    assert result.space_type == "hyperbolic"
    assert result.curvature == pytest.approx(1.25)
    assert len(result.nodes) == 3
    assert result.nodes[0].input_features.shape[0] == 4
    assert result.nodes[0].x_hyp is not None
    assert result.nodes[0].hyp_projections is not None
    assert result.metadata["coordinate_source"] == "ca_coords"
    assert result.metadata["edge_type_policy"] == "governed_ca_contact_r0_r1_r5_interim"
