# Migrated from: new (Phase 1/3 implementation) on 2026-05-27
"""Adapters that translate science code outputs into Normalizer payloads.

These adapters are the bridge between the internal GNN/phase representations
and the governed data layer. They handle:
1. Converting numpy arrays to lists (for Pydantic serialization)
2. Generating canonical residue_ids from structure + chain + index
3. Constructing proper provenance context
4. Splitting v4 dual-space outputs into separate Normalizer calls

The science code calls these adapters; the adapters call the Normalizer.
Science code never touches the Normalizer directly.

See: science/INTEGRATION_STRATEGY.md
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput, PhaseResult
from science.dtie.common.keys import make_residue_id
from science.dtie.common.normalizer_payloads import (
    GNNNodeResult,
    GNNOutputPayload,
    NormalizerResult,
    PersistenceBarcode,
    Phase3PersistencePayload,
    Phase3ResidueContribution,
    ProvenanceContext,
    RunType,
    SourceType,
    SpaceType,
)


def _to_list(arr: np.ndarray | None) -> list[float] | None:
    """Convert numpy array to list, handling None."""
    if arr is None:
        return None
    return arr.tolist()


class GNNOutputAdapter:
    """Adapts GNNInferenceResult into Normalizer payloads.

    For v4 GNN (which produces both Euclidean and Hyperbolic outputs),
    this adapter generates TWO payloads — one per embedding space.
    For v3 GNN (Euclidean only), it generates one payload.

    Usage:
        adapter = GNNOutputAdapter(normalizer=normalizer)
        results = await adapter.normalize(gnn_result, run_context)
    """

    def __init__(self, normalizer: Any):
        """Args: normalizer — a Normalizer instance (data.normalizer.core.Normalizer)."""
        self._normalizer = normalizer

    async def normalize(
        self,
        result: GNNInferenceResult,
        run_id: str | None = None,
        code_version: str | None = None,
        parent_run_id: str | None = None,
    ) -> list[NormalizerResult]:
        """Convert GNN inference result to governed data.

        For v4/v5, this produces two Normalizer calls:
        1. Hyperbolic embedding (primary, 32-dim Poincaré ball)
        2. Euclidean projection (secondary, 64-dim backward compat)

        For v3, this produces one call:
        1. Euclidean embedding (64-dim)

        The hyperbolic run is treated as the primary; the euclidean run
        links back to it via parent_run_id for provenance lineage.

        Args:
            result: The GNN inference output.
            run_id: Override run_id (auto-generated if None).
            code_version: Git commit hash.
            parent_run_id: Parent run for provenance chain.

        Returns:
            List of NormalizerResult (1 for v3, 2 for v4/v5).
        """
        if run_id is None:
            run_id = f"run_{uuid.uuid4().hex[:12]}"

        # Validate input features on first node (all nodes share the same format)
        if result.nodes:
            self._validate_input_features(result.nodes[0])

        results: list[NormalizerResult] = []

        if result.space_type == "hyperbolic":
            # v4/v5: emit hyperbolic primary + euclidean secondary
            hyp_run_id = f"{run_id}_hyp"
            hyp_result = await self._normalize_hyperbolic(
                result, run_id, code_version, parent_run_id
            )
            results.append(hyp_result)

            # Euclidean run links back to hyperbolic as its parent
            euc_result = await self._normalize_euclidean(
                result, run_id, code_version, parent_run_id=hyp_run_id
            )
            results.append(euc_result)
        else:
            # v3: emit euclidean only
            euc_result = await self._normalize_euclidean(
                result, run_id, code_version, parent_run_id
            )
            results.append(euc_result)

        return results

    @staticmethod
    def _validate_input_features(node: GNNNodeOutput) -> None:
        """Validate that input_features has the expected shape.

        The adapter assumes input_features[0:4] maps to:
        [rho, tau_flag, ss_type, sasa]. If the feature vector is
        shorter than expected, fail loudly rather than silently
        producing wrong values.
        """
        n_features = len(node.input_features) if node.input_features is not None else 0
        if n_features < 4:
            raise ValueError(
                f"Expected ≥4 input features per node, got {n_features}. "
                f"The GNN output format may have changed — update the adapter."
            )

    async def _normalize_hyperbolic(
        self,
        result: GNNInferenceResult,
        run_id: str,
        code_version: str | None,
        parent_run_id: str | None,
    ) -> NormalizerResult:
        """Emit the hyperbolic embedding payload (v4 primary)."""
        nodes = []
        for node in result.nodes:
            if node.x_hyp is None:
                continue

            residue_id = make_residue_id(result.structure_id, node.chain_label, node.residue_index)
            # For hyperbolic space, also populate high-precision double array for exact Lorentz math.
            hyp_emb = node.x_hyp
            embedding_double = hyp_emb.astype(np.float64).tolist() if hasattr(hyp_emb, "astype") else _to_list(hyp_emb)

            nodes.append(
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=node.residue_index,
                    chain_label=node.chain_label,
                    input_rho=float(node.input_features[0]),
                    input_tau_flag=float(node.input_features[1]),
                    input_ss_type=float(node.input_features[2]),
                    input_sasa=float(node.input_features[3]),
                    embedding=node.x_hyp.tolist(),
                    embedding_double=embedding_double,
                    cone_depth=node.cone_depth,
                    cone_width=node.cone_width,
                    epistemic_uncertainty=node.epistemic_uncertainty,
                    aleatoric_uncertainty=node.aleatoric_uncertainty,
                    total_uncertainty=node.total_uncertainty,
                    x_hyp=_to_list(node.x_hyp),
                    hyp_projections=_to_list(node.hyp_projections),
                    x_routed_hyp=_to_list(node.x_routed_hyp),
                    expert_weights=_to_list(node.expert_weights),
                )
            )

        hyp_dim = result.nodes[0].x_hyp.shape[0] if result.nodes and result.nodes[0].x_hyp is not None else 32

        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=f"{run_id}_hyp",
                structure_id=result.structure_id,
                model_version=result.model_version,
                pipeline_name="dtie_v4",
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
                checkpoint_uri=result.checkpoint_path,
                code_version=code_version,
                parent_run_id=parent_run_id,
            ),
            space_type=SpaceType.HYPERBOLIC,
            space_name=f"{result.model_version.lower().replace('-', '_')}_hyp{hyp_dim}",
            dimensionality=hyp_dim,
            curvature=require_learned_curvature(result.curvature, context="legacy hyperbolic gnn adapter"),
            nodes=nodes,
        )

        return await self._normalizer.normalize_gnn_output(payload)

    async def _normalize_euclidean(
        self,
        result: GNNInferenceResult,
        run_id: str,
        code_version: str | None,
        parent_run_id: str | None,
    ) -> NormalizerResult:
        """Emit the Euclidean projection payload (v3 primary, v4 secondary)."""
        nodes = []
        for node in result.nodes:
            residue_id = make_residue_id(result.structure_id, node.chain_label, node.residue_index)
            nodes.append(
                GNNNodeResult(
                    residue_id=residue_id,
                    residue_index=node.residue_index,
                    chain_label=node.chain_label,
                    input_rho=float(node.input_features[0]),
                    input_tau_flag=float(node.input_features[1]),
                    input_ss_type=float(node.input_features[2]),
                    input_sasa=float(node.input_features[3]),
                    embedding=node.projections.tolist(),
                    cone_depth=node.cone_depth,
                    cone_width=node.cone_width,
                    epistemic_uncertainty=node.epistemic_uncertainty,
                    aleatoric_uncertainty=node.aleatoric_uncertainty,
                    total_uncertainty=node.total_uncertainty,
                    expert_weights=_to_list(node.expert_weights),
                )
            )

        euc_dim = result.nodes[0].projections.shape[0] if result.nodes else 64
        pipeline = "dtie_v4" if result.space_type == "hyperbolic" else "dtie_v3"

        payload = GNNOutputPayload(
            provenance=ProvenanceContext(
                run_id=f"{run_id}_euc",
                structure_id=result.structure_id,
                model_version=result.model_version,
                pipeline_name=pipeline,
                run_type=RunType.INFERENCE,
                source_type=SourceType.PROBABILISTIC,
                checkpoint_uri=result.checkpoint_path,
                code_version=code_version,
                parent_run_id=parent_run_id,
            ),
            space_type=SpaceType.EUCLIDEAN,
            space_name=f"{result.model_version.lower().replace('-', '_')}_euc{euc_dim}",
            dimensionality=euc_dim,
            nodes=nodes,
        )

        return await self._normalizer.normalize_gnn_output(payload)


class Phase3Adapter:
    """Adapts Phase 3 persistence results into Normalizer payloads.

    Handles both v3 (standard witness persistence) and v4 (enhanced
    with hyperbolic distance integration).

    Usage:
        adapter = Phase3Adapter(normalizer=normalizer)
        result = await adapter.normalize(phase_result, run_context)
    """

    def __init__(self, normalizer: Any):
        self._normalizer = normalizer

    async def normalize(
        self,
        phase_result: PhaseResult,
        run_id: str | None = None,
        code_version: str | None = None,
        parent_run_id: str | None = None,
    ) -> NormalizerResult:
        """Convert Phase 3 result to governed data.

        Args:
            phase_result: The phase execution output.
            run_id: Override run_id.
            code_version: Git commit hash.
            parent_run_id: Parent run for provenance chain.

        Returns:
            NormalizerResult from the write.
        """
        if run_id is None:
            run_id = f"run_{uuid.uuid4().hex[:12]}"

        outputs = phase_result.outputs

        # Extract barcodes
        barcodes = [
            PersistenceBarcode(
                birth=b["birth"],
                death=b["death"],
                dimension=b.get("dimension", 0),
                generator_residues=b.get("generator_residues"),
            )
            for b in outputs.get("barcodes", [])
        ]

        # Extract per-residue contributions
        residue_contributions = None
        if phase_result.residue_contributions:
            residue_contributions = [
                Phase3ResidueContribution(
                    residue_id=rid,
                    persistence_score=score,
                )
                for rid, score in phase_result.residue_contributions.items()
            ]

        payload = Phase3PersistencePayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=phase_result.structure_id,
                model_version=phase_result.model_version,
                pipeline_name="dtie_v4" if "v4" in phase_result.model_version else "dtie_v3",
                run_type=RunType.INFERENCE,
                source_type=SourceType.DETERMINISTIC,
                code_version=code_version,
                parent_run_id=parent_run_id,
            ),
            barcodes=barcodes,
            max_alpha=outputs.get("max_alpha", 0.0),
            n_witnesses=outputs.get("n_witnesses", 0),
            n_landmarks=outputs.get("n_landmarks", 0),
            hyperbolic_distances_used=outputs.get("hyperbolic_distances_used", False),
            curvature_c=outputs.get("curvature_c"),
            residue_contributions=residue_contributions,
            landmark_to_residue=outputs.get("landmark_to_residue"),
        )

        return await self._normalizer.normalize_phase3_output(payload)
