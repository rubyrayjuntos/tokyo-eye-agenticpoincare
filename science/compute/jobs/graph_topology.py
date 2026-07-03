"""Atomic job: graph_topology (Act 01 — Signal)."""

from __future__ import annotations

import logging
from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


async def run_graph_topology(
    db: Any,
    config: PipelineConfig,
    *,
    run_id: str,
    gnn_run_id: str | None,
    pipeline_name: str,
    caller_identity: str = "compute_job_graph_topology",
    contact_cutoff_angstrom: float = 8.0,
    chain_filter: str | None = None,
) -> PhaseResult:
    """Build the Cα contact graph and persist edges + metrics via Normalizer."""
    import networkx as nx

    from data.normalizer.core import Normalizer
    from science.dtie.common.graph_builder import GraphBuilder
    from science.dtie.common.normalizer_payloads import (
        GraphTopologyPayload,
        ProvenanceContext,
        RunType,
        SourceType,
    )

    structure_id = config.structure_id
    builder = GraphBuilder(db=db, contact_cutoff=contact_cutoff_angstrom)
    try:
        graph = await builder.build_graph(structure_id, chain_filter=chain_filter)
    except ValueError as exc:
        return PhaseResult(
            phase_name="graph_topology",
            structure_id=structure_id,
            model_version="graph-topology-v1",
            success=False,
            outputs={"error": str(exc)},
        )

    edges = builder.extract_edges_for_persistence(graph)
    if not edges:
        return PhaseResult(
            phase_name="graph_topology",
            structure_id=structure_id,
            model_version="graph-topology-v1",
            success=False,
            outputs={"error": "No graph edges produced for structure"},
        )

    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge.source_residue_id, edge.target_residue_id)
    bridge_count = len(list(nx.articulation_points(G)))

    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    graph_payload = GraphTopologyPayload(
        structure_id=structure_id,
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version="graph-topology-v1",
            pipeline_name=pipeline_name,
            parent_run_id=gnn_run_id or run_id,
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DETERMINISTIC,
            code_version=config.code_version,
        ),
        edges=edges,
    )
    await normalizer.normalize_graph_topology(graph_payload)

    logger.info(
        "Graph topology persisted structure=%s edges=%d nodes=%d bridges=%d",
        structure_id,
        len(edges),
        len(graph.residues),
        bridge_count,
    )
    return PhaseResult(
        phase_name="graph_topology",
        structure_id=structure_id,
        model_version="graph-topology-v1",
        success=True,
        outputs={
            "edge_count": len(edges),
            "node_count": len(graph.residues),
            "run_id": run_id,
            "bridge_count": bridge_count,
            "metrics_computed": [
                "degree",
                "betweenness",
                "clustering_coefficient",
                "closeness",
                "eigenvector_centrality",
                "is_bridge",
                "conductance",
            ],
        },
    )
