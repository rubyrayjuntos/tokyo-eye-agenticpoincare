"""Tests for phase_loader helpers."""

from __future__ import annotations

import pytest

from science.compute.phase_loader import load_pharmacophore_phase_result, load_phase_output


class TestPhaseLoader:
    @pytest.mark.asyncio
    async def test_load_phase_output_empty(self):
        class _DB:
            async def fetch_one(self, *_args, **_kwargs):
                return None

        assert await load_phase_output(_DB(), "4obe", "phase35") is None

    @pytest.mark.asyncio
    async def test_load_pharmacophore_from_rows(self):
        class _DB:
            async def fetch_all(self, *_args, **_kwargs):
                return [
                    {
                        "pocket_index": 0,
                        "center_x": 1.0,
                        "center_y": 2.0,
                        "center_z": 3.0,
                        "druggability_score": 0.8,
                        "residue_count": 5,
                        "residue_indices": [12, 13],
                        "allosteric_coupling": 0.1,
                        "volume_estimate": 400.0,
                    }
                ]

        result = await load_pharmacophore_phase_result(_DB(), "4obe")
        assert result is not None
        assert result.success is True
        assert result.outputs["pharmacophore_count"] == 1
