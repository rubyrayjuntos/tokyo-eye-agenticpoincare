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

import asyncio

from science.dtie.common.keys import make_residue_id, validate_residue_id
from science.dtie.common.normalizer_payloads import (
    EvidencePayload,
    GNNNodeResult,
    GNNOutputPayload,
    GraphEdge,
    GraphTopologyPayload,
    HypothesisPayload,
    NormalizerResult,
    Phase3PersistencePayload,
    ProvenanceContext,
)

logger = logging.getLogger(__name__)


def _compute_graph_metrics(edges: list[Any]) -> dict[str, dict[str, Any]]:
    """Compute per-node graph metrics using networkx.

    This runs in a background thread via asyncio.to_thread to avoid
    blocking the event loop.

    Args:
        edges: List of GraphEdge objects.

    Returns:
        Dict mapping residue_id → metric dict with keys:
        degree, betweenness, clustering_coefficient, closeness,
        eigenvector_centrality, is_bridge, conductance.
    """
    import networkx as nx

    G = nx.Graph()
    for edge in edges:
        weight = edge.distance_angstrom if edge.distance_angstrom is not None else edge.weight
        G.add_edge(
            edge.source_residue_id,
            edge.target_residue_id,
            weight=weight,
            edge_type=edge.edge_type,
        )

    if G.number_of_nodes() == 0:
        return {}

    # Compute centrality metrics
    degree_dict = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    closeness = nx.closeness_centrality(G)

    # Eigenvector centrality can fail on disconnected graphs
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    # Bridge detection (articulation points in undirected graph)
    bridges_set = set(nx.articulation_points(G))

    # Conductance approximation via algebraic connectivity (Fiedler value)
    # For disconnected graphs, compute per-component
    conductance_dict: dict[str, float] = {}
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(component) <= 2:
            for node in component:
                conductance_dict[node] = 0.0
        else:
            try:
                fiedler = nx.algebraic_connectivity(subgraph)
                for node in component:
                    conductance_dict[node] = fiedler
            except Exception:
                for node in component:
                    conductance_dict[node] = 0.0

    # Assemble per-node metrics
    metrics: dict[str, dict[str, Any]] = {}
    for node in G.nodes():
        metrics[node] = {
            "degree": degree_dict[node],
            "betweenness": betweenness[node],
            "clustering_coefficient": clustering[node],
            "closeness": closeness[node],
            "eigenvector_centrality": eigenvector.get(node, 0.0),
            "is_bridge": node in bridges_set,
            "conductance": conductance_dict.get(node, 0.0),
        }

    return metrics


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
    # Path 3: Graph Topology
    # ------------------------------------------------------------------

    async def normalize_graph_topology(
        self, payload: GraphTopologyPayload
    ) -> NormalizerResult:
        """Validate and write graph topology (edges + computed metrics).

        Flow: provenance → validate residue_ids → transaction →
              upsert edges → compute metrics (asyncio.to_thread) →
              upsert metrics → register assets → commit → audit.

        Args:
            payload: Validated GraphTopologyPayload.

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

        # 2. Validate all residue_ids in edges
        for edge in payload.edges:
            for rid in (edge.source_residue_id, edge.target_residue_id):
                if not validate_residue_id(rid):
                    await self._log_audit(
                        run_id=prov.run_id,
                        structure_id=payload.structure_id,
                        payload_type="graph_topology",
                        status="validation_error",
                        error_message=f"Invalid residue_id format: '{rid}'",
                        duration_ms=self._elapsed_ms(start_time),
                        payload_summary={"edge_count": len(payload.edges)},
                    )
                    raise NormalizerError(
                        f"Invalid residue_id format: '{rid}'",
                        run_id=prov.run_id,
                    )

        # 3. Atomic transaction: edges + metrics
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 4. Upsert edges
            edge_params_list: list[dict[str, Any]] = []
            for edge in payload.edges:
                edge_id = str(uuid.uuid4())
                edge_params_list.append({
                    "edge_id": edge_id,
                    "run_id": prov.run_id,
                    "structure_id": payload.structure_id,
                    "source_residue_id": edge.source_residue_id,
                    "target_residue_id": edge.target_residue_id,
                    "edge_type": edge.edge_type,
                    "distance_angstrom": edge.distance_angstrom,
                    "hyperbolic_distance": edge.hyperbolic_distance,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(edge_id)

            await self._db.execute_many(
                """
                INSERT INTO fact_graph_edge (
                    edge_id, run_id, structure_id, source_residue_id,
                    target_residue_id, edge_type, distance_angstrom,
                    hyperbolic_distance, weight, metadata, computed_at
                ) VALUES (
                    :edge_id, :run_id, :structure_id, :source_residue_id,
                    :target_residue_id, :edge_type, :distance_angstrom,
                    :hyperbolic_distance, :weight, :metadata, :computed_at
                )
                ON CONFLICT (run_id, source_residue_id, target_residue_id, edge_type)
                DO UPDATE SET
                    distance_angstrom = EXCLUDED.distance_angstrom,
                    hyperbolic_distance = EXCLUDED.hyperbolic_distance,
                    weight = EXCLUDED.weight,
                    metadata = EXCLUDED.metadata,
                    computed_at = EXCLUDED.computed_at
                """,
                edge_params_list,
            )

            # 5. Compute metrics via asyncio.to_thread (non-blocking)
            metrics = await asyncio.to_thread(
                _compute_graph_metrics, payload.edges
            )

            # 6. Upsert metrics
            metric_params_list: list[dict[str, Any]] = []
            for residue_id, m in metrics.items():
                metric_id = str(uuid.uuid4())
                metric_params_list.append({
                    "metric_id": metric_id,
                    "run_id": prov.run_id,
                    "structure_id": payload.structure_id,
                    "residue_id": residue_id,
                    "degree": m["degree"],
                    "betweenness": m["betweenness"],
                    "clustering_coefficient": m["clustering_coefficient"],
                    "closeness": m["closeness"],
                    "eigenvector_centrality": m["eigenvector_centrality"],
                    "is_bridge": m["is_bridge"],
                    "conductance": m["conductance"],
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(metric_id)

            if metric_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_graph_node_metrics (
                        metric_id, run_id, structure_id, residue_id,
                        degree, betweenness, clustering_coefficient,
                        closeness, eigenvector_centrality, is_bridge,
                        conductance, computed_at
                    ) VALUES (
                        :metric_id, :run_id, :structure_id, :residue_id,
                        :degree, :betweenness, :clustering_coefficient,
                        :closeness, :eigenvector_centrality, :is_bridge,
                        :conductance, :computed_at
                    )
                    ON CONFLICT (run_id, residue_id) DO UPDATE SET
                        degree = EXCLUDED.degree,
                        betweenness = EXCLUDED.betweenness,
                        clustering_coefficient = EXCLUDED.clustering_coefficient,
                        closeness = EXCLUDED.closeness,
                        eigenvector_centrality = EXCLUDED.eigenvector_centrality,
                        is_bridge = EXCLUDED.is_bridge,
                        conductance = EXCLUDED.conductance,
                        computed_at = EXCLUDED.computed_at
                    """,
                    metric_params_list,
                )

            # 7. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="graph_topology",
                prov=prov,
            )

            await self._db.commit()

        except NormalizerError:
            await self._db.rollback()
            raise
        except Exception as e:
            await self._db.rollback()
            await self._log_audit(
                run_id=prov.run_id,
                structure_id=payload.structure_id,
                payload_type="graph_topology",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"edge_count": len(payload.edges)},
            )
            raise NormalizerError(
                f"Failed to write graph topology: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 8. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=payload.structure_id,
            payload_type="graph_topology",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "edge_count": len(payload.edges),
                "node_count": len(metrics),
                "structure_id": payload.structure_id,
            },
        )

        logger.info(
            "Normalized graph topology: run_id=%s, structure=%s, edges=%d, nodes=%d, duration=%dms",
            prov.run_id,
            payload.structure_id,
            len(payload.edges),
            len(metrics),
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
    # Path 4: Hypothesis Engine
    # ------------------------------------------------------------------

    async def normalize_hypothesis(
        self, payload: HypothesisPayload
    ) -> NormalizerResult:
        """Validate and write a hypothesis with its predictions.

        Flow: provenance → validate → transaction → upsert hypothesis →
              upsert predictions → register assets → commit → audit.

        Args:
            payload: Validated HypothesisPayload.

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

        # 2. Atomic transaction: hypothesis + predictions
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 3. Upsert hypothesis
            await self._db.execute(
                """
                INSERT INTO hypothesis (
                    hypothesis_id, structure_id, statement, mechanism,
                    status, confidence, created_by, created_at, updated_at
                ) VALUES (
                    :hypothesis_id, :structure_id, :statement, :mechanism,
                    :status, :confidence, :created_by, :created_at, :updated_at
                )
                ON CONFLICT (hypothesis_id) DO UPDATE SET
                    statement = EXCLUDED.statement,
                    mechanism = EXCLUDED.mechanism,
                    status = EXCLUDED.status,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "hypothesis_id": payload.hypothesis_id,
                    "structure_id": payload.structure_id,
                    "statement": payload.statement,
                    "mechanism": payload.mechanism,
                    "status": payload.status,
                    "confidence": payload.confidence,
                    "created_by": payload.created_by,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            asset_ids.append(payload.hypothesis_id)

            # 4. Upsert predictions
            pred_params_list: list[dict[str, Any]] = []
            for pred in payload.predictions:
                pred_params_list.append({
                    "prediction_id": pred.prediction_id,
                    "hypothesis_id": payload.hypothesis_id,
                    "statement": pred.statement,
                    "test_tool": pred.test_tool,
                    "test_params": pred.test_params,
                    "threshold": pred.threshold,
                })
                asset_ids.append(pred.prediction_id)

            await self._db.execute_many(
                """
                INSERT INTO hypothesis_prediction (
                    prediction_id, hypothesis_id, statement,
                    test_tool, test_params, threshold
                ) VALUES (
                    :prediction_id, :hypothesis_id, :statement,
                    :test_tool, :test_params, :threshold
                )
                ON CONFLICT (prediction_id) DO UPDATE SET
                    statement = EXCLUDED.statement,
                    test_tool = EXCLUDED.test_tool,
                    test_params = EXCLUDED.test_params,
                    threshold = EXCLUDED.threshold
                """,
                pred_params_list,
            )

            # 5. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="hypothesis",
                prov=prov,
            )

            await self._db.commit()

        except NormalizerError:
            await self._db.rollback()
            raise
        except Exception as e:
            await self._db.rollback()
            await self._log_audit(
                run_id=prov.run_id,
                structure_id=payload.structure_id,
                payload_type="hypothesis",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={
                    "hypothesis_id": payload.hypothesis_id,
                    "prediction_count": len(payload.predictions),
                },
            )
            raise NormalizerError(
                f"Failed to write hypothesis: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 6. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=payload.structure_id,
            payload_type="hypothesis",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "hypothesis_id": payload.hypothesis_id,
                "prediction_count": len(payload.predictions),
                "structure_id": payload.structure_id,
            },
        )

        logger.info(
            "Normalized hypothesis: run_id=%s, structure=%s, hypothesis=%s, predictions=%d, duration=%dms",
            prov.run_id,
            payload.structure_id,
            payload.hypothesis_id,
            len(payload.predictions),
            duration_ms,
        )

        return NormalizerResult(
            success=True,
            run_id=prov.run_id,
            assets_created=len(asset_ids),
            asset_ids=asset_ids,
            warnings=warnings,
        )

    async def normalize_evidence(
        self, payload: EvidencePayload
    ) -> NormalizerResult:
        """Validate and write evidence for an existing hypothesis.

        Flow: provenance → transaction → insert evidence →
              register asset → commit → audit.

        Args:
            payload: Validated EvidencePayload.

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

        # 2. Atomic transaction: insert evidence
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 3. Insert evidence record
            await self._db.execute(
                """
                INSERT INTO hypothesis_evidence (
                    evidence_id, hypothesis_id, source_tool, source_run_id,
                    supports, strength, description, gathered_at
                ) VALUES (
                    :evidence_id, :hypothesis_id, :source_tool, :source_run_id,
                    :supports, :strength, :description, :gathered_at
                )
                ON CONFLICT (evidence_id) DO UPDATE SET
                    supports = EXCLUDED.supports,
                    strength = EXCLUDED.strength,
                    description = EXCLUDED.description
                """,
                {
                    "evidence_id": payload.evidence_id,
                    "hypothesis_id": payload.hypothesis_id,
                    "source_tool": payload.source_tool,
                    "source_run_id": payload.source_run_id,
                    "supports": payload.supports,
                    "strength": payload.strength,
                    "description": payload.description,
                    "gathered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            asset_ids.append(payload.evidence_id)

            # 4. Register governed asset
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="hypothesis_evidence",
                prov=prov,
            )

            await self._db.commit()

        except NormalizerError:
            await self._db.rollback()
            raise
        except Exception as e:
            await self._db.rollback()
            await self._log_audit(
                run_id=prov.run_id,
                structure_id=prov.structure_id,
                payload_type="hypothesis_evidence",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={
                    "evidence_id": payload.evidence_id,
                    "hypothesis_id": payload.hypothesis_id,
                },
            )
            raise NormalizerError(
                f"Failed to write evidence: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 5. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="hypothesis_evidence",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "evidence_id": payload.evidence_id,
                "hypothesis_id": payload.hypothesis_id,
                "supports": payload.supports,
                "strength": payload.strength,
            },
        )

        logger.info(
            "Normalized evidence: run_id=%s, hypothesis=%s, evidence=%s, supports=%s, duration=%dms",
            prov.run_id,
            payload.hypothesis_id,
            payload.evidence_id,
            payload.supports,
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

    async def run_contradiction_check(
        self,
        structure_id: str,
        tool_dispatcher: Any = None,
    ) -> list[dict[str, Any]]:
        """Run contradiction detection for active hypotheses after pipeline writes.

        This should be called after any pipeline write (GNN output, Phase 3,
        graph topology) that produces new results for a structure. It checks
        active hypotheses and adds contradicting evidence if predictions flip.

        Args:
            structure_id: The structure that received new pipeline results.
            tool_dispatcher: Callable(tool_name, params) -> result for executing tools.

        Returns:
            List of contradiction records (empty if none found or on error).
        """
        try:
            from agent.tools.hypothesis.contradiction import check_contradictions

            return await check_contradictions(
                structure_id=structure_id,
                tool_dispatcher=tool_dispatcher,
                db=self._db,
            )
        except Exception as e:
            # Contradiction check must never break the primary pipeline
            logger.warning(
                "Contradiction check failed for structure %s: %s",
                structure_id, e,
            )
            return []

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        """Calculate elapsed milliseconds since start_time."""
        return int((time.monotonic() - start_time) * 1000)
