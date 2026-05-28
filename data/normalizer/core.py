# Migrated from: new (Phase 1 implementation) on 2026-05-27
"""Core Normalizer implementation — the single governed write path.

This module enforces:
1. Schema validation (via Pydantic payload models)
2. Provenance enforcement (every write must have a valid run_id)
3. Asset registration (every output is cataloged in governed_asset)
4. Idempotent upsert semantics (re-processing the same run is safe)
5. Atomic writes (partial failures leave no orphaned data)
6. Audit trail (every attempt is logged for monitoring and compliance)

Deployment: Import directly for in-process use, or wrap with the
FastAPI service layer (see data/normalizer/service.py).
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from science.dtie.common.keys import make_residue_id, validate_residue_id
from science.dtie.common.normalizer_payloads import (
    GNNNodeResult,
    GNNOutputPayload,
    NormalizerResult,
    Phase3PersistencePayload,
    ProvenanceContext,
)

logger = logging.getLogger(__name__)


class DatabaseConnection(Protocol):
    """Protocol for database access. Allows testing with mocks."""

    async def execute(self, query: str, params: dict[str, Any]) -> None: ...
    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None: ...
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...
    async def begin(self) -> Any: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class NormalizerError(Exception):
    """Raised when normalization fails."""

    def __init__(self, message: str, run_id: str | None = None, details: Any = None):
        super().__init__(message)
        self.run_id = run_id
        self.details = details


class Normalizer:
    """The governed write path for all scientific outputs.

    Usage:
        normalizer = Normalizer(db=connection)
        result = await normalizer.normalize_gnn_output(payload)
    """

    def __init__(self, db: DatabaseConnection, caller_identity: str = "unknown"):
        self._db = db
        self._caller_identity = caller_identity

    # ------------------------------------------------------------------
    # Path 1: GNN Node Output
    # ------------------------------------------------------------------

    async def normalize_gnn_output(self, payload: GNNOutputPayload) -> NormalizerResult:
        """Validate and write GNN node embeddings to the governed layer.

        This handles the highest-volume write path: per-residue GNN outputs
        for an entire structure.

        Args:
            payload: Validated GNNOutputPayload (Pydantic model).

        Returns:
            NormalizerResult with created asset IDs.

        Raises:
            NormalizerError: If validation or write fails.
        """
        start_time = time.monotonic()
        warnings: list[str] = []
        prov = payload.provenance

        # 1. Ensure provenance run exists
        await self._ensure_provenance_run(prov)

        # 2. Ensure embedding space is registered
        space_id = await self._ensure_embedding_space(
            space_name=payload.space_name,
            space_type=payload.space_type.value,
            dimensionality=payload.dimensionality,
            curvature=payload.curvature,
            model_name=prov.model_version,
        )

        # 3. Validate all residue_ids
        for node in payload.nodes:
            if not validate_residue_id(node.residue_id):
                await self._log_audit(
                    run_id=prov.run_id,
                    structure_id=prov.structure_id,
                    payload_type="gnn_output",
                    status="validation_error",
                    error_message=f"Invalid residue_id format: '{node.residue_id}'",
                    duration_ms=self._elapsed_ms(start_time),
                    payload_summary={"node_count": len(payload.nodes), "space": payload.space_name},
                )
                raise NormalizerError(
                    f"Invalid residue_id format: '{node.residue_id}'",
                    run_id=prov.run_id,
                )

        # 4. Write embeddings (atomic transaction)
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            for node in payload.nodes:
                asset_id = await self._write_gnn_node(
                    node=node,
                    prov=prov,
                    space_id=space_id,
                    computed_at=payload.computed_at,
                )
                asset_ids.append(asset_id)

            # 5. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="gnn_node_embedding",
                prov=prov,
            )

            await self._db.commit()

        except Exception as e:
            await self._db.rollback()
            await self._log_audit(
                run_id=prov.run_id,
                structure_id=prov.structure_id,
                payload_type="gnn_output",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"node_count": len(payload.nodes), "space": payload.space_name},
            )
            raise NormalizerError(
                f"Failed to write GNN output: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 6. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="gnn_output",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "node_count": len(payload.nodes),
                "space": payload.space_name,
                "space_type": payload.space_type.value,
                "dimensionality": payload.dimensionality,
                "model_version": prov.model_version,
            },
        )

        logger.info(
            "Normalized GNN output: run_id=%s, structure=%s, nodes=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.nodes),
            duration_ms,
        )

        return NormalizerResult(
            success=True,
            run_id=prov.run_id,
            assets_created=len(asset_ids),
            asset_ids=asset_ids,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Path 2: Phase 3 Persistence Output
    # ------------------------------------------------------------------

    async def normalize_phase3_output(
        self, payload: Phase3PersistencePayload
    ) -> NormalizerResult:
        """Validate and write Phase 3 persistence results.

        Args:
            payload: Validated Phase3PersistencePayload.

        Returns:
            NormalizerResult with created asset IDs.
        """
        start_time = time.monotonic()
        warnings: list[str] = []
        prov = payload.provenance

        await self._ensure_provenance_run(prov)

        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # Write the structure-level persistence record
            phase3_id = f"phase3_{prov.run_id}_{prov.structure_id}"
            await self._db.execute(
                """
                INSERT INTO fact_phase3_persistence (
                    phase3_id, run_id, structure_id, barcode_length,
                    max_alpha, persistence_data, source_type, computed_at
                ) VALUES (
                    :phase3_id, :run_id, :structure_id, :barcode_length,
                    :max_alpha, :persistence_data, :source_type, :computed_at
                )
                ON CONFLICT (phase3_id) DO UPDATE SET
                    persistence_data = EXCLUDED.persistence_data,
                    computed_at = EXCLUDED.computed_at
                """,
                {
                    "phase3_id": phase3_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "barcode_length": len(payload.barcodes),
                    "max_alpha": payload.max_alpha,
                    "persistence_data": {
                        "barcodes": [b.model_dump() for b in payload.barcodes],
                        "n_witnesses": payload.n_witnesses,
                        "n_landmarks": payload.n_landmarks,
                        "hyperbolic_distances_used": payload.hyperbolic_distances_used,
                        "curvature_c": payload.curvature_c,
                        "landmark_to_residue": payload.landmark_to_residue,
                    },
                    "source_type": prov.source_type.value,
                    "computed_at": payload.computed_at.isoformat(),
                },
            )
            asset_ids.append(phase3_id)

            # Write per-residue contributions in batch if provided
            if payload.residue_contributions:
                contrib_params_list: list[dict[str, Any]] = []
                for contrib in payload.residue_contributions:
                    if not validate_residue_id(contrib.residue_id):
                        warnings.append(
                            f"Skipping invalid residue_id: {contrib.residue_id}"
                        )
                        continue

                    contrib_id = f"phase3_res_{prov.run_id}_{contrib.residue_id}"
                    contrib_params_list.append({
                        "phase3_id": contrib_id,
                        "run_id": prov.run_id,
                        "structure_id": prov.structure_id,
                        "residue_id": contrib.residue_id,
                        "barcode_length": None,
                        "max_alpha": payload.max_alpha,
                        "persistence_data": {
                            "persistence_score": contrib.persistence_score,
                            "max_barcode_length": contrib.max_barcode_length,
                            "topological_significance": contrib.topological_significance,
                        },
                        "source_type": prov.source_type.value,
                        "computed_at": payload.computed_at.isoformat(),
                    })
                    asset_ids.append(contrib_id)

                if contrib_params_list:
                    await self._db.execute_many(
                        """
                        INSERT INTO fact_phase3_persistence (
                            phase3_id, run_id, structure_id, residue_id,
                            barcode_length, max_alpha, persistence_data,
                            source_type, computed_at
                        ) VALUES (
                            :phase3_id, :run_id, :structure_id, :residue_id,
                            :barcode_length, :max_alpha, :persistence_data,
                            :source_type, :computed_at
                        )
                        ON CONFLICT (phase3_id) DO UPDATE SET
                            persistence_data = EXCLUDED.persistence_data,
                            computed_at = EXCLUDED.computed_at
                        """,
                        contrib_params_list,
                    )

            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="dtie_phase3_persistence",
                prov=prov,
            )

            await self._db.commit()

        except Exception as e:
            await self._db.rollback()
            await self._log_audit(
                run_id=prov.run_id,
                structure_id=prov.structure_id,
                payload_type="phase3_persistence",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"barcode_count": len(payload.barcodes)},
            )
            raise NormalizerError(
                f"Failed to write Phase 3 output: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="phase3_persistence",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "barcode_count": len(payload.barcodes),
                "n_witnesses": payload.n_witnesses,
                "hyperbolic_distances_used": payload.hyperbolic_distances_used,
                "residue_contributions": len(payload.residue_contributions or []),
            },
        )

        logger.info(
            "Normalized Phase 3 output: run_id=%s, structure=%s, barcodes=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.barcodes),
            duration_ms,
        )

        return NormalizerResult(
            success=True,
            run_id=prov.run_id,
            assets_created=len(asset_ids),
            asset_ids=asset_ids,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_provenance_run(self, prov: ProvenanceContext) -> None:
        """Ensure a provenance_run record exists for this run_id.

        Uses ON CONFLICT DO NOTHING to handle concurrent inserts safely.
        Two callers with the same run_id will both succeed without error.
        """
        await self._db.execute(
            """
            INSERT INTO provenance_run (
                run_id, structure_id, model_version, checkpoint_uri,
                checkpoint_sha256, code_version, pipeline_name,
                run_type, source_type, parameters, parent_run_id, started_at
            ) VALUES (
                :run_id, :structure_id, :model_version, :checkpoint_uri,
                :checkpoint_sha256, :code_version, :pipeline_name,
                :run_type, :source_type, :parameters, :parent_run_id, :started_at
            )
            ON CONFLICT (run_id) DO NOTHING
            """,
            {
                "run_id": prov.run_id,
                "structure_id": prov.structure_id,
                "model_version": prov.model_version,
                "checkpoint_uri": prov.checkpoint_uri,
                "checkpoint_sha256": prov.checkpoint_sha256,
                "code_version": prov.code_version,
                "pipeline_name": prov.pipeline_name,
                "run_type": prov.run_type.value,
                "source_type": prov.source_type.value,
                "parameters": prov.parameters,
                "parent_run_id": prov.parent_run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _ensure_embedding_space(
        self,
        space_name: str,
        space_type: str,
        dimensionality: int,
        curvature: float | None,
        model_name: str,
    ) -> str:
        """Ensure the embedding space is registered. Returns space_id."""
        existing = await self._db.fetch_one(
            "SELECT space_id FROM embedding_space WHERE name = :name",
            {"name": space_name},
        )
        if existing:
            return existing["space_id"]

        space_id = f"space_{space_name}"
        await self._db.execute(
            """
            INSERT INTO embedding_space (
                space_id, name, space_type, dimensionality, curvature,
                model_name, is_active
            ) VALUES (
                :space_id, :name, :space_type, :dimensionality, :curvature,
                :model_name, TRUE
            )
            """,
            {
                "space_id": space_id,
                "name": space_name,
                "space_type": space_type,
                "dimensionality": dimensionality,
                "curvature": curvature,
                "model_name": model_name,
            },
        )
        return space_id

    async def _write_gnn_node(
        self,
        node: GNNNodeResult,
        prov: ProvenanceContext,
        space_id: str,
        computed_at: datetime,
    ) -> str:
        """Write a single GNN node embedding. Returns the embedding_id.

        Idempotency is on (run_id, residue_id, space_id) — the natural key
        for "this residue's embedding from this run in this space". Re-processing
        the same run updates the embedding data in place.
        """
        embedding_id = str(uuid.uuid4())

        await self._db.execute(
            """
            INSERT INTO fact_gnn_node_embedding (
                embedding_id, run_id, structure_id, residue_id, space_id,
                input_rho, input_tau_flag, input_ss_type, input_sasa,
                embedding, hyp_projections, cone_depth, cone_width,
                epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
                expert_weights, source_type, model_version, computed_at
            ) VALUES (
                :embedding_id, :run_id, :structure_id, :residue_id, :space_id,
                :input_rho, :input_tau_flag, :input_ss_type, :input_sasa,
                :embedding, :hyp_projections, :cone_depth, :cone_width,
                :epistemic_uncertainty, :aleatoric_uncertainty, :total_uncertainty,
                :expert_weights, :source_type, :model_version, :computed_at
            )
            ON CONFLICT (run_id, residue_id, space_id) DO UPDATE SET
                embedding_id = EXCLUDED.embedding_id,
                embedding = EXCLUDED.embedding,
                hyp_projections = EXCLUDED.hyp_projections,
                cone_depth = EXCLUDED.cone_depth,
                cone_width = EXCLUDED.cone_width,
                epistemic_uncertainty = EXCLUDED.epistemic_uncertainty,
                aleatoric_uncertainty = EXCLUDED.aleatoric_uncertainty,
                total_uncertainty = EXCLUDED.total_uncertainty,
                expert_weights = EXCLUDED.expert_weights,
                computed_at = EXCLUDED.computed_at
            """,
            {
                "embedding_id": embedding_id,
                "run_id": prov.run_id,
                "structure_id": prov.structure_id,
                "residue_id": node.residue_id,
                "space_id": space_id,
                "input_rho": node.input_rho,
                "input_tau_flag": node.input_tau_flag,
                "input_ss_type": node.input_ss_type,
                "input_sasa": node.input_sasa,
                "embedding": node.embedding,
                "hyp_projections": node.hyp_projections,
                "cone_depth": node.cone_depth,
                "cone_width": node.cone_width,
                "epistemic_uncertainty": node.epistemic_uncertainty,
                "aleatoric_uncertainty": node.aleatoric_uncertainty,
                "total_uncertainty": node.total_uncertainty,
                "expert_weights": node.expert_weights,
                "source_type": prov.source_type.value,
                "model_version": prov.model_version,
                "computed_at": computed_at.isoformat(),
            },
        )
        return embedding_id

    async def _register_governed_assets(
        self,
        asset_ids: list[str],
        asset_type: str,
        prov: ProvenanceContext,
    ) -> None:
        """Register assets in the governed_asset catalog."""
        params_list = [
            {
                "asset_id": aid,
                "asset_type": asset_type,
                "structure_id": prov.structure_id,
                "run_id": prov.run_id,
                "access_level": "internal",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for aid in asset_ids
        ]
        await self._db.execute_many(
            """
            INSERT INTO governed_asset (
                asset_id, asset_type, structure_id, run_id, access_level, created_at
            ) VALUES (
                :asset_id, :asset_type, :structure_id, :run_id, :access_level, :created_at
            )
            ON CONFLICT (asset_id) DO NOTHING
            """,
            params_list,
        )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    async def _log_audit(
        self,
        run_id: str | None,
        structure_id: str | None,
        payload_type: str,
        status: str,
        error_message: str | None = None,
        assets_created: int = 0,
        duration_ms: int | None = None,
        payload_summary: dict[str, Any] | None = None,
    ) -> None:
        """Log a normalization attempt to the audit trail.

        This is fire-and-forget — audit failures should not block the
        primary write path.
        """
        try:
            await self._db.execute(
                """
                INSERT INTO normalization_audit (
                    run_id, structure_id, payload_type, status,
                    assets_created, error_message, payload_summary,
                    duration_ms, caller_identity
                ) VALUES (
                    :run_id, :structure_id, :payload_type, :status,
                    :assets_created, :error_message, :payload_summary,
                    :duration_ms, :caller_identity
                )
                """,
                {
                    "run_id": run_id,
                    "structure_id": structure_id,
                    "payload_type": payload_type,
                    "status": status,
                    "assets_created": assets_created,
                    "error_message": error_message,
                    "payload_summary": payload_summary,
                    "duration_ms": duration_ms,
                    "caller_identity": self._caller_identity,
                },
            )
        except Exception as e:
            # Audit logging must never break the primary path
            logger.warning("Failed to write audit log: %s", e)

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        """Calculate elapsed milliseconds since start_time."""
        return int((time.monotonic() - start_time) * 1000)
