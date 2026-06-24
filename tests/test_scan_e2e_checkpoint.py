"""End-to-end checkpoint verification for binding site scan at ingestion.

Verifies:
1. Full pipeline produces ranked binding sites for a test structure
2. query_binding_sites returns pre-computed results instantly

This is task 9 in the binding-site-scan-at-ingestion spec.
"""

from __future__ import annotations

import pytest

from agent.tools.cryptic.scan_phase import run_full_structure_scan, ScanResult
from agent.tools.cryptic.tool import query_binding_sites
from agent.tools.cryptic.seed_generator import GNNNodeOutput
from agent.tools.cryptic.site_merger import UnifiedCandidate
from tests.conftest import QueryValidatingMockDB


# ---------------------------------------------------------------------------
# Fixtures: Synthetic GNN data that will produce clusters
# ---------------------------------------------------------------------------


def _build_mock_db_with_gnn_data() -> QueryValidatingMockDB:
    """Build a mock DB pre-loaded with synthetic GNN + coordinate data.

    Creates 2 clusters of qualifying residues:
    - Cluster A: residues A:10, A:11, A:12 (close together, high signal)
    - Cluster B: residues A:20, A:21, A:22 (close together, moderate signal)
    Plus some non-qualifying residues that should be filtered out.
    """
    db = QueryValidatingMockDB()

    # GNN node embeddings — qualifying residues for two clusters + noise
    gnn_rows = [
        # Cluster A — high uncertainty + high cone_depth (all qualify)
        {"residue_id": "4obe:A:10", "epistemic_uncertainty": 15.0, "cone_depth": 8.0},
        {"residue_id": "4obe:A:11", "epistemic_uncertainty": 18.0, "cone_depth": 9.0},
        {"residue_id": "4obe:A:12", "epistemic_uncertainty": 12.0, "cone_depth": 7.5},
        # Cluster B — moderate signal (still qualifies at default thresholds)
        {"residue_id": "4obe:A:20", "epistemic_uncertainty": 10.0, "cone_depth": 6.5},
        {"residue_id": "4obe:A:21", "epistemic_uncertainty": 11.0, "cone_depth": 7.0},
        {"residue_id": "4obe:A:22", "epistemic_uncertainty": 9.5, "cone_depth": 6.0},
        # Non-qualifying residues (below thresholds)
        {"residue_id": "4obe:A:30", "epistemic_uncertainty": 5.0, "cone_depth": 3.0},
        {"residue_id": "4obe:A:31", "epistemic_uncertainty": 2.0, "cone_depth": 1.0},
    ]
    db.register_response("fact_gnn_node_embedding", gnn_rows)

    # Cα coordinates — cluster A near (10,10,10), cluster B near (50,50,50)
    # Include all columns needed by both _fetch_ca_coordinates and _write_pdb_from_dim_atom
    ca_rows = [
        {"residue_id": "4obe:A:10", "x": 10.0, "y": 10.0, "z": 10.0,
         "residue_index": 10, "residue_name": "ALA", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:11", "x": 11.0, "y": 10.5, "z": 10.0,
         "residue_index": 11, "residue_name": "GLY", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:12", "x": 10.5, "y": 11.0, "z": 9.5,
         "residue_index": 12, "residue_name": "VAL", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:20", "x": 50.0, "y": 50.0, "z": 50.0,
         "residue_index": 20, "residue_name": "LEU", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:21", "x": 51.0, "y": 50.5, "z": 50.0,
         "residue_index": 21, "residue_name": "ILE", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:22", "x": 50.5, "y": 51.0, "z": 49.5,
         "residue_index": 22, "residue_name": "PHE", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:30", "x": 80.0, "y": 80.0, "z": 80.0,
         "residue_index": 30, "residue_name": "SER", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
        {"residue_id": "4obe:A:31", "x": 81.0, "y": 80.0, "z": 80.0,
         "residue_index": 31, "residue_name": "THR", "chain_label": "A",
         "atom_name": "CA", "element": "C", "occupancy": 1.0, "b_factor": 0.0},
    ]
    db.register_response("dim_atom", ca_rows)

    # Graph metrics (provides classification heuristic data)
    graph_rows = [
        {"residue_id": "4obe:A:10", "betweenness": 0.15, "clustering_coefficient": 0.2,
         "is_bridge": True, "degree": 5, "closeness": 0.5, "eigenvector_centrality": 0.3,
         "conductance": 0.1},
        {"residue_id": "4obe:A:11", "betweenness": 0.12, "clustering_coefficient": 0.18,
         "is_bridge": False, "degree": 4, "closeness": 0.45, "eigenvector_centrality": 0.25,
         "conductance": 0.08},
        {"residue_id": "4obe:A:12", "betweenness": 0.08, "clustering_coefficient": 0.22,
         "is_bridge": True, "degree": 3, "closeness": 0.4, "eigenvector_centrality": 0.2,
         "conductance": 0.12},
        {"residue_id": "4obe:A:20", "betweenness": 0.05, "clustering_coefficient": 0.4,
         "is_bridge": False, "degree": 4, "closeness": 0.35, "eigenvector_centrality": 0.15,
         "conductance": 0.06},
        {"residue_id": "4obe:A:21", "betweenness": 0.04, "clustering_coefficient": 0.45,
         "is_bridge": False, "degree": 3, "closeness": 0.3, "eigenvector_centrality": 0.12,
         "conductance": 0.05},
        {"residue_id": "4obe:A:22", "betweenness": 0.03, "clustering_coefficient": 0.5,
         "is_bridge": False, "degree": 3, "closeness": 0.28, "eigenvector_centrality": 0.1,
         "conductance": 0.04},
    ]
    db.register_response("fact_graph_node_metrics", graph_rows)

    # dim_structure for pocket detector pdb_id lookup
    db.register_response("dim_structure", [{"pdb_id": "4OBE"}])

    # Provenance run for model version lookup
    db.register_response("provenance_run", [{"model_version": "GOSPConeMapper-v5"}])

    # Empty scan metadata (first scan for this structure)
    db.register_response("fact_binding_site_scan", [])

    return db


# ---------------------------------------------------------------------------
# Test: Full pipeline produces ranked binding sites
# ---------------------------------------------------------------------------


class TestFullPipelineScanProducesRankedSites:
    """Verify the scan phase produces correctly ranked binding sites."""

    @pytest.mark.asyncio
    async def test_scan_produces_candidates(self):
        """Full scan should produce candidate sites from synthetic GNN data."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(
            structure_id="4obe",
            db=db,
            uncertainty_threshold=9.5,
            cone_depth_threshold=6.0,
            eps_angstrom=8.0,
            min_cluster_size=3,
            max_clusters=25,
        )

        assert isinstance(result, ScanResult)
        assert result.structure_id == "4obe"
        assert result.run_id.startswith("scan_4obe_")
        assert len(result.candidates) == 2  # Two distinct clusters
        assert result.heuristic_version == "v1.0"
        assert result.model_version == "GOSPConeMapper-v5"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_candidates_have_valid_ranks(self):
        """All candidates should have consecutive ranks from 1 to N."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        ranks = [c.site_rank for c in result.candidates]
        assert ranks == list(range(1, len(result.candidates) + 1))

    @pytest.mark.asyncio
    async def test_candidates_ranked_by_druggability_descending(self):
        """Higher-ranked sites should have >= druggability scores."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].druggability_score >= result.candidates[i + 1].druggability_score

    @pytest.mark.asyncio
    async def test_candidates_have_valid_site_types(self):
        """Each candidate should have a recognized site type."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        valid_types = {
            "cryptic_wedge", "structural_stent", "dynamic_lid",
            "allosteric_clamp", "strain_relief_insert", "surface_pocket",
        }
        for c in result.candidates:
            assert c.site_type in valid_types, f"Invalid site_type: {c.site_type}"

    @pytest.mark.asyncio
    async def test_candidates_have_discovery_method(self):
        """Each candidate should have a valid discovery method."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        valid_methods = {"gnn_strain", "geometry", "hybrid"}
        for c in result.candidates:
            assert c.discovery_method in valid_methods

    @pytest.mark.asyncio
    async def test_candidates_have_druggability_in_bounds(self):
        """All druggability scores should be in [0, 1]."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        for c in result.candidates:
            assert 0.0 <= c.druggability_score <= 1.0

    @pytest.mark.asyncio
    async def test_scan_provenance_fields_present(self):
        """Scan result should include all required provenance fields."""
        db = _build_mock_db_with_gnn_data()

        result = await run_full_structure_scan(structure_id="4obe", db=db)

        assert result.run_id is not None
        assert result.model_version is not None
        assert result.heuristic_version is not None
        assert "uncertainty_threshold" in result.scan_parameters
        assert "cone_depth_threshold" in result.scan_parameters
        assert "eps_angstrom" in result.scan_parameters
        assert "min_cluster_size" in result.scan_parameters

    @pytest.mark.asyncio
    async def test_scan_persists_to_database(self):
        """Scan should execute INSERT queries to persist candidates and metadata."""
        db = _build_mock_db_with_gnn_data()

        await run_full_structure_scan(structure_id="4obe", db=db)

        # Should have written to fact_cryptic_site
        db.assert_query_executed_containing("fact_cryptic_site")
        # Should have written scan metadata
        db.assert_query_executed_containing("fact_binding_site_scan")


# ---------------------------------------------------------------------------
# Test: query_binding_sites returns pre-computed results
# ---------------------------------------------------------------------------


class TestQueryBindingSitesReturnsPrecomputed:
    """Verify the query interface returns instant results from DB."""

    @pytest.mark.asyncio
    async def test_query_returns_sites_when_scan_exists(self):
        """query_binding_sites should return pre-computed results."""
        db = QueryValidatingMockDB()

        # Simulate a completed scan metadata record
        db.register_response("fact_binding_site_scan", [{
            "scan_id": "test-scan-id",
            "status": "complete",
            "sites_found": 2,
            "heuristic_version": "v1.0",
            "created_at": "2026-06-01T00:00:00Z",
        }])

        # Simulate pre-computed sites in fact_cryptic_site
        db.register_response("fact_cryptic_site", [
            {
                "site_id": "site-001",
                "site_type": "cryptic_wedge",
                "residue_ids": ["4obe:A:10", "4obe:A:11", "4obe:A:12"],
                "druggability_score": 0.85,
                "site_rank": 1,
                "discovery_method": "gnn_strain",
                "provenance_gate": "scan_phase",
                "md_validation_status": "pending",
                "heuristic_version": "v1.0",
                "composite_gnn_score": 0.75,
                "fpocket_druggability": None,
                "volume_angstrom3": None,
            },
            {
                "site_id": "site-002",
                "site_type": "dynamic_lid",
                "residue_ids": ["4obe:A:20", "4obe:A:21", "4obe:A:22"],
                "druggability_score": 0.62,
                "site_rank": 2,
                "discovery_method": "gnn_strain",
                "provenance_gate": "scan_phase",
                "md_validation_status": "pending",
                "heuristic_version": "v1.0",
                "composite_gnn_score": 0.55,
                "fpocket_druggability": None,
                "volume_angstrom3": None,
            },
        ])

        result = await query_binding_sites(structure_id="4obe", db=db)

        assert result.success is True
        assert result.data["count"] == 2
        assert len(result.data["sites"]) == 2
        # Sites should be ordered by rank
        assert result.data["sites"][0]["site_rank"] == 1
        assert result.data["sites"][1]["site_rank"] == 2

    @pytest.mark.asyncio
    async def test_query_returns_message_when_no_scan(self):
        """query_binding_sites should indicate pipeline must run when no scan exists."""
        db = QueryValidatingMockDB()
        # No scan metadata
        db.register_response("fact_binding_site_scan", [])

        result = await query_binding_sites(structure_id="4obe", db=db)

        assert result.success is True
        assert result.data["count"] == 0
        assert "pipeline" in result.message.lower() or "scan" in result.message.lower()

    @pytest.mark.asyncio
    async def test_query_with_site_type_filter(self):
        """query_binding_sites should support filtering by site_type."""
        db = QueryValidatingMockDB()

        db.register_response("fact_binding_site_scan", [{
            "scan_id": "test-scan-id",
            "status": "complete",
            "sites_found": 1,
            "heuristic_version": "v1.0",
            "created_at": "2026-06-01T00:00:00Z",
        }])

        # Only one site matches the filter
        db.register_response("fact_cryptic_site", [{
            "site_id": "site-001",
            "site_type": "cryptic_wedge",
            "residue_ids": ["4obe:A:10"],
            "druggability_score": 0.85,
            "site_rank": 1,
            "discovery_method": "gnn_strain",
            "provenance_gate": "scan_phase",
            "md_validation_status": "pending",
            "heuristic_version": "v1.0",
            "composite_gnn_score": 0.75,
            "fpocket_druggability": None,
            "volume_angstrom3": None,
        }])

        result = await query_binding_sites(
            structure_id="4obe", db=db, site_type="cryptic_wedge"
        )

        assert result.success is True
        assert "cryptic_wedge" in result.data["filters_applied"]

    @pytest.mark.asyncio
    async def test_query_with_min_druggability_filter(self):
        """query_binding_sites should support filtering by minimum druggability."""
        db = QueryValidatingMockDB()

        db.register_response("fact_binding_site_scan", [{
            "scan_id": "test-scan-id",
            "status": "complete",
            "sites_found": 1,
            "heuristic_version": "v1.0",
            "created_at": "2026-06-01T00:00:00Z",
        }])

        db.register_response("fact_cryptic_site", [{
            "site_id": "site-001",
            "site_type": "cryptic_wedge",
            "residue_ids": ["4obe:A:10"],
            "druggability_score": 0.9,
            "site_rank": 1,
            "discovery_method": "gnn_strain",
            "provenance_gate": "scan_phase",
            "md_validation_status": "pending",
            "heuristic_version": "v1.0",
            "composite_gnn_score": 0.8,
            "fpocket_druggability": None,
            "volume_angstrom3": None,
        }])

        result = await query_binding_sites(
            structure_id="4obe", db=db, min_druggability=0.8
        )

        assert result.success is True
        assert "druggability≥0.8" in result.data["filters_applied"]

    @pytest.mark.asyncio
    async def test_query_result_has_required_fields(self):
        """Each query result should include all fields per Requirement 5.3."""
        db = QueryValidatingMockDB()

        db.register_response("fact_binding_site_scan", [{
            "scan_id": "test-scan-id",
            "status": "complete",
            "sites_found": 1,
            "heuristic_version": "v1.0",
            "created_at": "2026-06-01T00:00:00Z",
        }])

        db.register_response("fact_cryptic_site", [{
            "site_id": "site-001",
            "site_type": "structural_stent",
            "residue_ids": ["4obe:A:10", "4obe:A:11"],
            "druggability_score": 0.72,
            "site_rank": 1,
            "discovery_method": "hybrid",
            "provenance_gate": "scan_phase",
            "md_validation_status": "pending",
            "heuristic_version": "v1.0",
            "composite_gnn_score": 0.65,
            "fpocket_druggability": 0.7,
            "volume_angstrom3": 450.0,
        }])

        result = await query_binding_sites(structure_id="4obe", db=db)

        assert result.success is True
        site = result.data["sites"][0]
        required_fields = [
            "site_id", "site_type", "residue_ids", "druggability_score",
            "site_rank", "discovery_method", "provenance_gate",
            "md_validation_status", "heuristic_version",
        ]
        for field in required_fields:
            assert field in site, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Test: Full pipeline integration via orchestrator
# ---------------------------------------------------------------------------


class TestPipelineIntegrationWithScanPhase:
    """Verify scan outputs round-trip independently of the orchestrator."""

    @pytest.mark.asyncio
    async def test_scan_then_query_round_trip(self):
        """After scan produces sites, query should find them via the DB."""
        db = _build_mock_db_with_gnn_data()

        # Run the scan (this writes to the mock DB via execute_many/execute)
        scan_result = await run_full_structure_scan(structure_id="4obe", db=db)
        assert len(scan_result.candidates) == 2

        # Now register the scan metadata + site results that would be in DB
        # (simulating what query_binding_sites will read back)
        db.register_response("fact_binding_site_scan", [{
            "scan_id": "test-scan-id",
            "status": "complete",
            "sites_found": 2,
            "heuristic_version": "v1.0",
            "created_at": "2026-06-19T00:00:00Z",
        }])

        site_rows = [
            {
                "site_id": c.site_id,
                "site_type": c.site_type,
                "residue_ids": c.residue_ids,
                "druggability_score": c.druggability_score,
                "site_rank": c.site_rank,
                "discovery_method": c.discovery_method,
                "provenance_gate": c.provenance_gate,
                "md_validation_status": "pending",
                "heuristic_version": c.heuristic_version,
                "composite_gnn_score": c.composite_gnn_score,
                "fpocket_druggability": c.fpocket_druggability,
                "volume_angstrom3": c.volume_angstrom3,
            }
            for c in scan_result.candidates
        ]
        db.register_response("fact_cryptic_site", site_rows)

        # Query should return the same sites
        query_result = await query_binding_sites(structure_id="4obe", db=db)
        assert query_result.success is True
        assert query_result.data["count"] == 2
        assert query_result.data["sites"][0]["site_rank"] == 1
        assert query_result.data["sites"][1]["site_rank"] == 2
