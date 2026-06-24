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

from psycopg.types.json import Json

from science.dtie.common.keys import make_residue_id, validate_residue_id
from science.dtie.common.normalizer_payloads import (
    AllostericSitePayload,
    DrugCandidatePayload,
    EvidencePayload,
    GNNNodeResult,
    GNNOutputPayload,
    GraphEdge,
    GraphTopologyPayload,
    HypothesisPayload,
    NormalizerResult,
    Phase2VulnerabilityPayload,
    Phase3PersistencePayload,
    PharmacophorePayload,
    ProvenanceContext,
    ResistancePathwayPayload,
    SourceLeakPayload,
    TopologicalLiftPayload,
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
                    expert_load_per_run=payload.expert_load,
                    routing_entropy=payload.routing_entropy,
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
    # Path 2b: Source-Leak Detection
    # ------------------------------------------------------------------

    async def normalize_source_leaks(
        self, payload: SourceLeakPayload
    ) -> NormalizerResult:
        """Validate and write source-leak detection results.

        Flow: provenance → validate residue_ids → transaction →
              upsert leak records → register assets → commit → audit.

        Args:
            payload: Validated SourceLeakPayload.

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

        # 2. Validate all residue_ids
        for residue in payload.leak_residues:
            if not validate_residue_id(residue.residue_id):
                await self._log_audit(
                    run_id=prov.run_id,
                    structure_id=prov.structure_id,
                    payload_type="source_leak",
                    status="validation_error",
                    error_message=f"Invalid residue_id format: '{residue.residue_id}'",
                    duration_ms=self._elapsed_ms(start_time),
                    payload_summary={"leak_count": len(payload.leak_residues)},
                )
                raise NormalizerError(
                    f"Invalid residue_id format: '{residue.residue_id}'",
                    run_id=prov.run_id,
                )

        # 3. Atomic transaction: upsert leak records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            leak_params_list: list[dict[str, Any]] = []
            for residue in payload.leak_residues:
                # Use deterministic ID based on natural key (run_id, residue_id)
                # so re-runs produce the same asset_id for governed_asset tracking.
                leak_id = f"leak_{prov.run_id}_{residue.residue_id}"
                leak_params_list.append({
                    "leak_id": leak_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "residue_id": residue.residue_id,
                    "epistemic_uncertainty": residue.epistemic_uncertainty,
                    "cone_depth": residue.cone_depth,
                    "leak_score": residue.leak_score,
                    "is_confirmed": residue.is_confirmed,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(leak_id)

            if leak_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_source_leak (
                        leak_id, run_id, structure_id, residue_id,
                        epistemic_uncertainty, cone_depth, leak_score,
                        is_confirmed, computed_at
                    ) VALUES (
                        :leak_id, :run_id, :structure_id, :residue_id,
                        :epistemic_uncertainty, :cone_depth, :leak_score,
                        :is_confirmed, :computed_at
                    )
                    ON CONFLICT (run_id, residue_id) DO UPDATE SET
                        leak_id = EXCLUDED.leak_id,
                        epistemic_uncertainty = EXCLUDED.epistemic_uncertainty,
                        cone_depth = EXCLUDED.cone_depth,
                        leak_score = EXCLUDED.leak_score,
                        is_confirmed = EXCLUDED.is_confirmed,
                        computed_at = EXCLUDED.computed_at
                    """,
                    leak_params_list,
                )

            # 4. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="source_leak",
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
                payload_type="source_leak",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"leak_count": len(payload.leak_residues)},
            )
            raise NormalizerError(
                f"Failed to write source-leak results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 5. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="source_leak",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "leak_count": len(payload.leak_residues),
                "total_leaks": payload.total_leaks,
                "threshold_used": payload.threshold_used,
            },
        )

        logger.info(
            "Normalized source-leak output: run_id=%s, structure=%s, leaks=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.leak_residues),
            duration_ms,
        )

        return NormalizerResult(
            success=True,
            run_id=prov.run_id,
            assets_created=len(asset_ids),
            asset_ids=asset_ids,
            warnings=warnings,
            asset_metadata={"source_leak_count": len(payload.leak_residues)},
        )

    # ------------------------------------------------------------------
    # Path 2c: Allosteric Site Persistence
    # ------------------------------------------------------------------

    async def normalize_allosteric_sites(
        self, payload: AllostericSitePayload
    ) -> NormalizerResult:
        """Validate and write allosteric site predictions.

        Flow: provenance → validate residue_ids → transaction →
              upsert site records → upsert site-residue junction →
              register assets → commit → audit.

        Args:
            payload: Validated AllostericSitePayload.

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

        # 2. Validate all residue_ids in all sites
        for site in payload.sites:
            for rid in site.residue_ids:
                if not validate_residue_id(rid):
                    await self._log_audit(
                        run_id=prov.run_id,
                        structure_id=prov.structure_id,
                        payload_type="allosteric_site",
                        status="validation_error",
                        error_message=f"Invalid residue_id format: '{rid}'",
                        duration_ms=self._elapsed_ms(start_time),
                        payload_summary={"site_count": len(payload.sites)},
                    )
                    raise NormalizerError(
                        f"Invalid residue_id format: '{rid}'",
                        run_id=prov.run_id,
                    )

        # 3. Atomic transaction: upsert sites + junction records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 3a. Upsert site records
            site_params_list: list[dict[str, Any]] = []
            for site in payload.sites:
                site_params_list.append({
                    "site_id": site.site_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "centroid_x": site.centroid_x,
                    "centroid_y": site.centroid_y,
                    "centroid_z": site.centroid_z,
                    "confidence_score": site.confidence_score,
                    "cluster_method": site.cluster_method,
                    "n_residues": site.n_residues,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(site.site_id)

            if site_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_allosteric_site (
                        site_id, run_id, structure_id,
                        centroid_x, centroid_y, centroid_z,
                        confidence_score, cluster_method, n_residues,
                        computed_at
                    ) VALUES (
                        :site_id, :run_id, :structure_id,
                        :centroid_x, :centroid_y, :centroid_z,
                        :confidence_score, :cluster_method, :n_residues,
                        :computed_at
                    )
                    ON CONFLICT (run_id, site_id) DO UPDATE SET
                        centroid_x = EXCLUDED.centroid_x,
                        centroid_y = EXCLUDED.centroid_y,
                        centroid_z = EXCLUDED.centroid_z,
                        confidence_score = EXCLUDED.confidence_score,
                        cluster_method = EXCLUDED.cluster_method,
                        n_residues = EXCLUDED.n_residues,
                        computed_at = EXCLUDED.computed_at
                    """,
                    site_params_list,
                )

            # 3b. Upsert site-residue junction records
            junction_params_list: list[dict[str, Any]] = []
            for site in payload.sites:
                for rid in site.residue_ids:
                    junction_id = str(uuid.uuid4())
                    junction_params_list.append({
                        "id": junction_id,
                        "run_id": prov.run_id,
                        "site_id": site.site_id,
                        "residue_id": rid,
                        "contribution_score": None,
                    })

            if junction_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_allosteric_site_residue (
                        id, run_id, site_id, residue_id, contribution_score
                    ) VALUES (
                        :id, :run_id, :site_id, :residue_id, :contribution_score
                    )
                    ON CONFLICT (run_id, site_id, residue_id) DO NOTHING
                    """,
                    junction_params_list,
                )

            # 4. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="allosteric_site",
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
                payload_type="allosteric_site",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"site_count": len(payload.sites)},
            )
            raise NormalizerError(
                f"Failed to write allosteric site results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 5. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="allosteric_site",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "site_count": len(payload.sites),
                "total_sites": payload.total_sites,
                "total_residues": sum(s.n_residues for s in payload.sites),
            },
        )

        logger.info(
            "Normalized allosteric site output: run_id=%s, structure=%s, sites=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.sites),
            duration_ms,
        )

        return NormalizerResult(
            success=True,
            run_id=prov.run_id,
            assets_created=len(asset_ids),
            asset_ids=asset_ids,
            warnings=warnings,
            asset_metadata={"site_count": len(payload.sites)},
        )

    # ------------------------------------------------------------------
    # Path 5: Phase 2 Vulnerability Persistence
    # ------------------------------------------------------------------

    async def normalize_phase2_vulnerability(
        self, payload: Phase2VulnerabilityPayload
    ) -> NormalizerResult:
        """Validate and write Phase 2 vulnerability doorway results.

        Flow: provenance → validate residue_ids → transaction →
              upsert vulnerability records → register assets → commit → audit.

        Args:
            payload: Validated Phase2VulnerabilityPayload.

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

        # 2. Validate all residue_ids
        for doorway in payload.doorways:
            if not validate_residue_id(doorway.residue_id):
                await self._log_audit(
                    run_id=prov.run_id,
                    structure_id=prov.structure_id,
                    payload_type="phase2_vulnerability",
                    status="validation_error",
                    error_message=f"Invalid residue_id format: '{doorway.residue_id}'",
                    duration_ms=self._elapsed_ms(start_time),
                    payload_summary={"doorway_count": len(payload.doorways)},
                )
                raise NormalizerError(
                    f"Invalid residue_id format: '{doorway.residue_id}'",
                    run_id=prov.run_id,
                )

        # 3. Atomic transaction: upsert vulnerability records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            vuln_params_list: list[dict[str, Any]] = []
            for doorway in payload.doorways:
                vuln_id = f"vuln_{prov.run_id}_{doorway.residue_id}"
                vuln_params_list.append({
                    "vulnerability_id": vuln_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "residue_id": doorway.residue_id,
                    "cone_depth": doorway.cone_depth,
                    "epistemic_uncertainty": doorway.epistemic_uncertainty,
                    "aleatoric_uncertainty": doorway.aleatoric_uncertainty,
                    "depth_threshold": payload.depth_threshold,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(vuln_id)

            if vuln_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_phase2_vulnerability (
                        vulnerability_id, run_id, structure_id, residue_id,
                        cone_depth, epistemic_uncertainty, aleatoric_uncertainty,
                        depth_threshold, computed_at
                    ) VALUES (
                        :vulnerability_id, :run_id, :structure_id, :residue_id,
                        :cone_depth, :epistemic_uncertainty, :aleatoric_uncertainty,
                        :depth_threshold, :computed_at
                    )
                    ON CONFLICT (run_id, residue_id) DO UPDATE SET
                        vulnerability_id = EXCLUDED.vulnerability_id,
                        cone_depth = EXCLUDED.cone_depth,
                        epistemic_uncertainty = EXCLUDED.epistemic_uncertainty,
                        aleatoric_uncertainty = EXCLUDED.aleatoric_uncertainty,
                        depth_threshold = EXCLUDED.depth_threshold,
                        computed_at = EXCLUDED.computed_at
                    """,
                    vuln_params_list,
                )

            # 4. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="phase2_vulnerability",
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
                payload_type="phase2_vulnerability",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"doorway_count": len(payload.doorways)},
            )
            raise NormalizerError(
                f"Failed to write Phase 2 vulnerability results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 5. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="phase2_vulnerability",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "doorway_count": len(payload.doorways),
                "epistemic_median": payload.epistemic_median,
                "depth_threshold": payload.depth_threshold,
                "total_residues": payload.total_residues,
            },
        )

        logger.info(
            "Normalized Phase 2 vulnerability: run_id=%s, structure=%s, doorways=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.doorways),
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
    # Path 6: Phase 3.5 Topological Lift Persistence
    # ------------------------------------------------------------------

    async def normalize_topological_lift(
        self, payload: TopologicalLiftPayload
    ) -> NormalizerResult:
        """Validate and write Phase 3.5 topological lift results.

        Flow: provenance → transaction → upsert lift records →
              register assets → commit → audit.

        Args:
            payload: Validated TopologicalLiftPayload.

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

        # 2. Atomic transaction: upsert lift records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            lift_params_list: list[dict[str, Any]] = []
            for site in payload.lifted_sites:
                lift_id = f"lift_{prov.run_id}_{site.site_index}"
                lift_params_list.append({
                    "lift_id": lift_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "site_index": site.site_index,
                    "lifted_x": site.lifted_x,
                    "lifted_y": site.lifted_y,
                    "lifted_z": site.lifted_z,
                    "vertex_count": site.vertex_count,
                    "source_method": site.source_method,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(lift_id)

            if lift_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_topological_lift (
                        lift_id, run_id, structure_id, site_index,
                        lifted_x, lifted_y, lifted_z, vertex_count,
                        source_method, computed_at
                    ) VALUES (
                        :lift_id, :run_id, :structure_id, :site_index,
                        :lifted_x, :lifted_y, :lifted_z, :vertex_count,
                        :source_method, :computed_at
                    )
                    ON CONFLICT (run_id, site_index) DO UPDATE SET
                        lift_id = EXCLUDED.lift_id,
                        lifted_x = EXCLUDED.lifted_x,
                        lifted_y = EXCLUDED.lifted_y,
                        lifted_z = EXCLUDED.lifted_z,
                        vertex_count = EXCLUDED.vertex_count,
                        source_method = EXCLUDED.source_method,
                        computed_at = EXCLUDED.computed_at
                    """,
                    lift_params_list,
                )

            # 3. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="topological_lift",
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
                payload_type="topological_lift",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"site_count": len(payload.lifted_sites)},
            )
            raise NormalizerError(
                f"Failed to write topological lift results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 4. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="topological_lift",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "site_count": len(payload.lifted_sites),
                "method": payload.method,
            },
        )

        logger.info(
            "Normalized topological lift: run_id=%s, structure=%s, sites=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.lifted_sites),
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
    # Path 7: Phase 4 Resistance Pathway Persistence
    # ------------------------------------------------------------------

    async def normalize_resistance_pathways(
        self, payload: ResistancePathwayPayload
    ) -> NormalizerResult:
        """Validate and write Phase 4 resistance pathway results.

        Flow: provenance → transaction → upsert pathway records →
              upsert spectral summary → register assets → commit → audit.

        Args:
            payload: Validated ResistancePathwayPayload.

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

        # 2. Atomic transaction: upsert pathway + spectral records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 2a. Upsert pathway records
            pathway_params_list: list[dict[str, Any]] = []
            for pathway in payload.pathways:
                pathway_id = f"pathway_{prov.run_id}_{pathway.source_node}_{pathway.target_node}"
                pathway_params_list.append({
                    "pathway_id": pathway_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "source_node": pathway.source_node,
                    "target_node": pathway.target_node,
                    "source_residue": pathway.source_residue,
                    "target_residue": pathway.target_residue,
                    "effective_resistance": pathway.effective_resistance,
                    "coupling_strength": pathway.coupling_strength,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(pathway_id)

            if pathway_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_resistance_pathway (
                        pathway_id, run_id, structure_id, source_node,
                        target_node, source_residue, target_residue,
                        effective_resistance, coupling_strength, computed_at
                    ) VALUES (
                        :pathway_id, :run_id, :structure_id, :source_node,
                        :target_node, :source_residue, :target_residue,
                        :effective_resistance, :coupling_strength, :computed_at
                    )
                    ON CONFLICT (run_id, source_node, target_node) DO UPDATE SET
                        pathway_id = EXCLUDED.pathway_id,
                        source_residue = EXCLUDED.source_residue,
                        target_residue = EXCLUDED.target_residue,
                        effective_resistance = EXCLUDED.effective_resistance,
                        coupling_strength = EXCLUDED.coupling_strength,
                        computed_at = EXCLUDED.computed_at
                    """,
                    pathway_params_list,
                )

            # 2b. Upsert spectral summary (one row per run)
            import json as _json

            spectral_id = f"spectral_{prov.run_id}"
            await self._db.execute(
                """
                INSERT INTO fact_resistance_spectral (
                    spectral_id, run_id, structure_id, lambda_2,
                    hinge_residues, graph_nodes, graph_edges, computed_at
                ) VALUES (
                    :spectral_id, :run_id, :structure_id, :lambda_2,
                    :hinge_residues, :graph_nodes, :graph_edges, :computed_at
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    spectral_id = EXCLUDED.spectral_id,
                    lambda_2 = EXCLUDED.lambda_2,
                    hinge_residues = EXCLUDED.hinge_residues,
                    graph_nodes = EXCLUDED.graph_nodes,
                    graph_edges = EXCLUDED.graph_edges,
                    computed_at = EXCLUDED.computed_at
                """,
                {
                    "spectral_id": spectral_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "lambda_2": payload.lambda_2,
                    "hinge_residues": _json.dumps(payload.hinge_residues),
                    "graph_nodes": payload.graph_nodes,
                    "graph_edges": payload.graph_edges,
                    "computed_at": payload.computed_at.isoformat(),
                },
            )
            asset_ids.append(spectral_id)

            # 3. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="resistance_pathway",
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
                payload_type="resistance_pathway",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"pathway_count": len(payload.pathways)},
            )
            raise NormalizerError(
                f"Failed to write resistance pathway results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 4. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="resistance_pathway",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "pathway_count": len(payload.pathways),
                "lambda_2": payload.lambda_2,
                "graph_nodes": payload.graph_nodes,
                "graph_edges": payload.graph_edges,
            },
        )

        logger.info(
            "Normalized resistance pathways: run_id=%s, structure=%s, pathways=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.pathways),
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
    # Path 8: Phase 5 Pharmacophore Persistence
    # ------------------------------------------------------------------

    async def normalize_pharmacophores(
        self, payload: PharmacophorePayload
    ) -> NormalizerResult:
        """Validate and write Phase 5 pharmacophore results.

        Flow: provenance → transaction → upsert pharmacophore records →
              register assets → commit → audit.

        Args:
            payload: Validated PharmacophorePayload.

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

        # 2. Atomic transaction: upsert pharmacophore records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            import json as _json

            pharma_params_list: list[dict[str, Any]] = []
            for pharma in payload.pharmacophores:
                pharma_id = f"pharma_{prov.run_id}_{pharma.pocket_index}"
                pharma_params_list.append({
                    "pharmacophore_id": pharma_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "pocket_index": pharma.pocket_index,
                    "center_x": pharma.center_x,
                    "center_y": pharma.center_y,
                    "center_z": pharma.center_z,
                    "druggability_score": pharma.druggability_score,
                    "residue_count": pharma.residue_count,
                    "residue_indices": _json.dumps(pharma.residue_indices),
                    "allosteric_coupling": pharma.allosteric_coupling,
                    "volume_estimate": pharma.volume_estimate,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(pharma_id)

            if pharma_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_pharmacophore (
                        pharmacophore_id, run_id, structure_id, pocket_index,
                        center_x, center_y, center_z, druggability_score,
                        residue_count, residue_indices, allosteric_coupling,
                        volume_estimate, computed_at
                    ) VALUES (
                        :pharmacophore_id, :run_id, :structure_id, :pocket_index,
                        :center_x, :center_y, :center_z, :druggability_score,
                        :residue_count, :residue_indices, :allosteric_coupling,
                        :volume_estimate, :computed_at
                    )
                    ON CONFLICT (run_id, pocket_index) DO UPDATE SET
                        pharmacophore_id = EXCLUDED.pharmacophore_id,
                        center_x = EXCLUDED.center_x,
                        center_y = EXCLUDED.center_y,
                        center_z = EXCLUDED.center_z,
                        druggability_score = EXCLUDED.druggability_score,
                        residue_count = EXCLUDED.residue_count,
                        residue_indices = EXCLUDED.residue_indices,
                        allosteric_coupling = EXCLUDED.allosteric_coupling,
                        volume_estimate = EXCLUDED.volume_estimate,
                        computed_at = EXCLUDED.computed_at
                    """,
                    pharma_params_list,
                )

            # 3. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="pharmacophore",
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
                payload_type="pharmacophore",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"pharmacophore_count": len(payload.pharmacophores)},
            )
            raise NormalizerError(
                f"Failed to write pharmacophore results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 4. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="pharmacophore",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "pharmacophore_count": len(payload.pharmacophores),
                "druggability_threshold": payload.druggability_threshold,
            },
        )

        logger.info(
            "Normalized pharmacophores: run_id=%s, structure=%s, pharmacophores=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.pharmacophores),
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
    # Path 9: Phase 6 Drug Candidate Persistence
    # ------------------------------------------------------------------

    async def normalize_drug_candidates(
        self, payload: DrugCandidatePayload
    ) -> NormalizerResult:
        """Validate and write Phase 6 drug candidate results.

        Flow: provenance → transaction → upsert candidate records →
              register assets → commit → audit.

        Args:
            payload: Validated DrugCandidatePayload.

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

        # 2. Atomic transaction: upsert candidate records
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            candidate_params_list: list[dict[str, Any]] = []
            for candidate in payload.candidates:
                candidate_id = f"candidate_{prov.run_id}_{candidate.pocket_index}"
                candidate_params_list.append({
                    "candidate_id": candidate_id,
                    "run_id": prov.run_id,
                    "structure_id": prov.structure_id,
                    "pocket_index": candidate.pocket_index,
                    "center_x": candidate.center_x,
                    "center_y": candidate.center_y,
                    "center_z": candidate.center_z,
                    "accessibility_score": candidate.accessibility_score,
                    "binding_potential": candidate.binding_potential,
                    "admet_pass": candidate.admet_pass,
                    "selectivity_ratio": candidate.selectivity_ratio,
                    "is_state_selective": candidate.is_state_selective,
                    "combined_druggability": candidate.combined_druggability,
                    "computed_at": payload.computed_at.isoformat(),
                })
                asset_ids.append(candidate_id)

            if candidate_params_list:
                await self._db.execute_many(
                    """
                    INSERT INTO fact_drug_candidate (
                        candidate_id, run_id, structure_id, pocket_index,
                        center_x, center_y, center_z, accessibility_score,
                        binding_potential, admet_pass, selectivity_ratio,
                        is_state_selective, combined_druggability, computed_at
                    ) VALUES (
                        :candidate_id, :run_id, :structure_id, :pocket_index,
                        :center_x, :center_y, :center_z, :accessibility_score,
                        :binding_potential, :admet_pass, :selectivity_ratio,
                        :is_state_selective, :combined_druggability, :computed_at
                    )
                    ON CONFLICT (run_id, pocket_index) DO UPDATE SET
                        candidate_id = EXCLUDED.candidate_id,
                        center_x = EXCLUDED.center_x,
                        center_y = EXCLUDED.center_y,
                        center_z = EXCLUDED.center_z,
                        accessibility_score = EXCLUDED.accessibility_score,
                        binding_potential = EXCLUDED.binding_potential,
                        admet_pass = EXCLUDED.admet_pass,
                        selectivity_ratio = EXCLUDED.selectivity_ratio,
                        is_state_selective = EXCLUDED.is_state_selective,
                        combined_druggability = EXCLUDED.combined_druggability,
                        computed_at = EXCLUDED.computed_at
                    """,
                    candidate_params_list,
                )

            # 3. Register governed assets
            await self._register_governed_assets(
                asset_ids=asset_ids,
                asset_type="drug_candidate",
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
                payload_type="drug_candidate",
                status="write_error",
                error_message=str(e),
                duration_ms=self._elapsed_ms(start_time),
                payload_summary={"candidate_count": len(payload.candidates)},
            )
            raise NormalizerError(
                f"Failed to write drug candidate results: {e}",
                run_id=prov.run_id,
                details=str(e),
            ) from e

        duration_ms = self._elapsed_ms(start_time)

        # 4. Log successful audit
        await self._log_audit(
            run_id=prov.run_id,
            structure_id=prov.structure_id,
            payload_type="drug_candidate",
            status="success",
            assets_created=len(asset_ids),
            duration_ms=duration_ms,
            payload_summary={
                "candidate_count": len(payload.candidates),
                "admet_passed_count": payload.admet_passed_count,
                "state_selective_count": payload.state_selective_count,
            },
        )

        logger.info(
            "Normalized drug candidates: run_id=%s, structure=%s, candidates=%d, duration=%dms",
            prov.run_id,
            prov.structure_id,
            len(payload.candidates),
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

        # 3. Compute graph metrics BEFORE the transaction to avoid holding
        # a DB connection idle during CPU-bound networkx work.
        metrics = await asyncio.to_thread(
            _compute_graph_metrics, payload.edges
        )

        # 4. Atomic transaction: edges + metrics
        asset_ids: list[str] = []
        try:
            await self._db.begin()

            # 4a. Upsert edges
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

            # 4b. Upsert metrics (pre-computed above)
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
                # Stringify complex fields (test_params dicts etc.) for psycopg executemany compatibility
                # (avoids "cannot adapt type 'dict'" on %s / jsonb columns in some pool/adapter configs).
                import json as _json_local
                tp = pred.test_params
                if isinstance(tp, (dict, list)):
                    tp = _json_local.dumps(tp)
                th = pred.threshold
                if isinstance(th, (dict, list)):
                    th = _json_local.dumps(th)
                pred_params_list.append({
                    "prediction_id": pred.prediction_id,
                    "hypothesis_id": payload.hypothesis_id,
                    "statement": pred.statement,
                    "test_tool": pred.test_tool,
                    "test_params": tp,
                    "threshold": th,
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
        """Ensure the embedding space is registered. Returns space_id.

        Uses INSERT ... ON CONFLICT to handle concurrent registration safely.
        """
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
            ON CONFLICT (name) DO NOTHING
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
        expert_load_per_run: list[float] | None = None,
        routing_entropy: float | None = None,
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
                embedding, embedding_double, hyp_projections, hyp_projection_2d, cone_depth, cone_width,
                epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
                expert_weights, source_type, model_version, computed_at,
                expert_load_per_run, routing_entropy
            ) VALUES (
                :embedding_id, :run_id, :structure_id, :residue_id, :space_id,
                :input_rho, :input_tau_flag, :input_ss_type, :input_sasa,
                :embedding, :embedding_double, :hyp_projections, :hyp_projection_2d, :cone_depth, :cone_width,
                :epistemic_uncertainty, :aleatoric_uncertainty, :total_uncertainty,
                :expert_weights, :source_type, :model_version, :computed_at,
                :expert_load_per_run, :routing_entropy
            )
            ON CONFLICT (run_id, residue_id, space_id) DO UPDATE SET
                embedding_id = EXCLUDED.embedding_id,
                embedding = EXCLUDED.embedding,
                embedding_double = EXCLUDED.embedding_double,
                hyp_projections = EXCLUDED.hyp_projections,
                hyp_projection_2d = EXCLUDED.hyp_projection_2d,
                cone_depth = EXCLUDED.cone_depth,
                cone_width = EXCLUDED.cone_width,
                epistemic_uncertainty = EXCLUDED.epistemic_uncertainty,
                aleatoric_uncertainty = EXCLUDED.aleatoric_uncertainty,
                total_uncertainty = EXCLUDED.total_uncertainty,
                expert_weights = EXCLUDED.expert_weights,
                computed_at = EXCLUDED.computed_at,
                expert_load_per_run = EXCLUDED.expert_load_per_run,
                routing_entropy = EXCLUDED.routing_entropy
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
                "embedding_double": node.embedding_double,
                "hyp_projections": Json(node.hyp_projections) if node.hyp_projections else None,
                "hyp_projection_2d": ("[" + ",".join(str(v) for v in node.hyp_projections) + "]") if (node.hyp_projections and len(node.hyp_projections) == 2) else None,
                "cone_depth": node.cone_depth,
                "cone_width": node.cone_width,
                "epistemic_uncertainty": node.epistemic_uncertainty,
                "aleatoric_uncertainty": node.aleatoric_uncertainty,
                "total_uncertainty": node.total_uncertainty,
                "expert_weights": Json(node.expert_weights) if node.expert_weights else None,
                "source_type": prov.source_type.value,
                "model_version": prov.model_version,
                "computed_at": computed_at.isoformat(),
                "expert_load_per_run": Json(expert_load_per_run) if expert_load_per_run else None,
                "routing_entropy": routing_entropy,
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
        import json as _json

        # psycopg3 does not auto-serialize dicts to JSONB — must be explicit
        summary_json = _json.dumps(payload_summary) if payload_summary is not None else None

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
                    "payload_summary": summary_json,
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
