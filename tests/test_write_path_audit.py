"""P1 defensibility gates: write-path audit + idempotency registry."""

from __future__ import annotations

from data.audit.write_paths import audit_write_path_gate


def test_write_path_audit_gate_passes() -> None:
    errors = audit_write_path_gate()
    assert errors == [], "Write-path violations:\n" + "\n".join(errors)


def test_write_path_inventory_scans_update_operations() -> None:
    from data.audit.write_paths import scan_repository

    hits = scan_repository(
        include_tests=False,
        operations=frozenset({"update"}),
    )
    paths = {hit.path for hit in hits}
    assert "agent/tools/hypothesis/tools.py" in paths


def test_idempotency_gates_are_registered() -> None:
    """Document the property tests that gate double-run idempotency."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    ingest = repo / "tests/test_normalizer_ingest_properties.py"
    graph = repo / "tests/test_graph_topology_normalizer.py"
    integration = repo / "tests/test_normalizer_integration.py"
    assert ingest.exists()
    assert graph.exists()
    assert integration.exists()
    ingest_text = ingest.read_text(encoding="utf-8")
    graph_text = graph.read_text(encoding="utf-8")
    integration_text = integration.read_text(encoding="utf-8")
    assert "test_double_ingest_no_duplicates" in ingest_text
    assert "test_property_2_idempotent_writes" in graph_text
    assert "test_normalize_gnn_output_is_idempotent" in integration_text
