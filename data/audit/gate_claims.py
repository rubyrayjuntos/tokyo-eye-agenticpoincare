"""§0 meta-gate: verify named enforcement gates still exist and assert claimed symbols.

**Existence, not liveness.** This module checks that gate artifact files exist and
that ``required_symbols`` still appear in source. That catches rename/deletion drift
but *not* vacuous assertions (``assert x or True``, unreachable tests, etc.). See
``data/audit/gate_canaries.py`` for liveness canaries on high-value claims.

**SSOT note.** ``GATE_CLAIMS`` is the machine-enforced registry. The prose table in
``docs/ENFORCEMENT_MATRIX.md`` is human-readable and may lag — when they disagree,
trust ``GATE_CLAIMS`` + these audits. End state: matrix shrinks to backlog-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GateClaim:
    """One enforceable artifact and symbols that must remain in its source."""

    gate_id: str
    artifact: str
    required_symbols: tuple[str, ...]
    description: str = ""


# SSOT for §0 / ENFORCEMENT_MATRIX GATED rows. Add a row here when the matrix
# gains a new GATED gate; the meta-gate fails if the artifact or symbol drifts.
GATE_CLAIMS: tuple[GateClaim, ...] = (
    # §0 audit table (June 2026)
    GateClaim(
        "onboard_contract_registry",
        "tests/test_onboard_contract.py",
        (
            "validate_registry_job_keys_match_contract",
            "test_all_artifacts_declare_geometric_space",
            "test_hyperbolic_jobs_authoritative",
        ),
        "Registry produces ⊆ contract; geometric consistency",
    ),
    GateClaim(
        "runner_registry_triangle",
        "science/compute/runner_dispatch.py",
        ("validate_runners_against_registry", "JOB_RUNNERS"),
        "Bidirectional registry ↔ JOB_RUNNERS",
    ),
    GateClaim(
        "compute_preconditions",
        "tests/test_compute_preconditions.py",
        ("check_job_preconditions",),
        "Precondition resolver exercised in isolation",
    ),
    GateClaim(
        "compute_precondition_entrypoints",
        "tests/test_compute_precondition_entrypoints.py",
        (
            "PRECONDITION_RESOLVER",
            "dispatch_job_in_process",
            "dispatch_compute_job",
        ),
        "Scheduler and HTTP entrypoints share one resolver",
    ),
    GateClaim(
        "readiness_contract_sync",
        "tests/test_readiness_contract_sync.py",
        ("get_artifact_probe_keys", "TIER1_ARTIFACTS"),
        "Tier/act/probe constants track contract reload",
    ),
    GateClaim(
        "write_path_inventory",
        "data/audit/write_paths.py",
        ("INSERT_RE", "UPDATE_RE", "COPY_RE", "audit_write_path_gate"),
        "Governed-table INSERT/UPDATE/COPY scan",
    ),
    GateClaim(
        "write_path_ci",
        "tests/test_write_path_audit.py",
        ("audit_write_path_gate", "test_double_ingest_no_duplicates"),
        "CI write-path gate + idempotency registry",
    ),
    GateClaim(
        "write_path_lint",
        "scripts/lint_write_paths.py",
        ("audit_write_path_gate",),
    ),
    GateClaim(
        "provenance_gate",
        "tests/test_provenance_gate.py",
        (
            "test_ingest_dimensions_creates_provenance_before_dim_writes",
            "test_missing_run_id_rejected_before_governed_write",
        ),
        "Provenance before governed writes",
    ),
    GateClaim(
        "provenance_normalizer",
        "data/normalizer/core.py",
        ("_ensure_provenance_run", "_assert_provenance_run_active"),
        "Normalizer enforces active provenance_run",
    ),
    GateClaim(
        "ingest_idempotency",
        "tests/test_normalizer_ingest_properties.py",
        ("test_double_ingest_no_duplicates",),
    ),
    GateClaim(
        "graph_idempotency",
        "tests/test_graph_topology_normalizer.py",
        ("test_property_2_idempotent_writes",),
    ),
    GateClaim(
        "gnn_idempotency",
        "tests/test_normalizer_integration.py",
        ("test_normalize_gnn_output_is_idempotent",),
    ),
    GateClaim(
        "import_contract_test",
        "tests/test_import_contracts.py",
        ("audit_import_contracts",),
    ),
    GateClaim(
        "import_contract_lint",
        "scripts/lint_import_contracts.py",
        ("audit_import_contracts",),
    ),
    GateClaim(
        "geometric_contract",
        "science/contracts/onboard_contract.py",
        ("validate_geometric_contract", "validate_registry_job_keys_match_contract"),
    ),
    GateClaim(
        "curvature_literal_test",
        "tests/test_curvature_literal_lint.py",
        ("scan_repository",),
    ),
    GateClaim(
        "curvature_literal_lint",
        "scripts/lint_curvature_literals.py",
        ("scan_repository",),
    ),
    GateClaim(
        "embedding_projection_gate",
        "tests/test_embedding_projection_gate.py",
        ("parse_embedding_projection_xy", "test_hydration_quarantines_malformed_rows"),
    ),
    GateClaim(
        "md_validation_gate",
        "tests/test_md_validation_gate.py",
        (
            "smd_result_qualifies_for_passed",
            "test_dry_run_runner_never_emits_md_validation_artifact",
        ),
    ),
    GateClaim(
        "contract_codegen_test",
        "tests/test_contract_codegen_gates.py",
        ("audit_contract_sync", "audit_contract_version_bump"),
    ),
    GateClaim(
        "contract_sync_lint",
        "scripts/lint_contract_sync.py",
        ("audit_contract_sync",),
    ),
    GateClaim(
        "contract_version_lint",
        "scripts/lint_contract_version.py",
        ("audit_contract_version_bump",),
    ),
    GateClaim(
        "audit_event_type_lint",
        "scripts/lint_audit_event_types.py",
        ("audit_event_type_literals",),
    ),
    GateClaim(
        "enforcement_meta_gate",
        "tests/test_enforcement_matrix_gate_claims.py",
        ("audit_gate_claims", "GATE_CLAIMS"),
    ),
    GateClaim(
        "readiness_probe_gate",
        "tests/test_readiness_probe_gate.py",
        ("ProbeInfrastructureError", "probe_errors"),
    ),
    GateClaim(
        "gate_liveness_canaries",
        "data/audit/gate_canaries.py",
        ("audit_gate_liveness_canaries", "GATE_LIVENESS_CANARY_IDS"),
    ),
    GateClaim(
        "science_container_integration",
        "tests/test_science_container_integration.py",
        ("seed_ingest_foundation", "DBAdapter"),
    ),
    GateClaim(
        "pipeline_job_id_gate",
        "tests/test_pipeline_job_id_gate.py",
        ("pipeline_job_id", "check_job_preconditions"),
    ),
)


def audit_gate_claims(
    *,
    root: Path | None = None,
    claims: tuple[GateClaim, ...] = GATE_CLAIMS,
) -> list[str]:
    """Return human-readable errors when a gate artifact or symbol is missing."""
    repo = root or ROOT
    errors: list[str] = []

    for claim in claims:
        path = repo / claim.artifact
        if not path.is_file():
            errors.append(f"{claim.gate_id}: missing artifact {claim.artifact}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{claim.gate_id}: cannot read {claim.artifact}: {exc}")
            continue
        for symbol in claim.required_symbols:
            if symbol not in text:
                errors.append(
                    f"{claim.gate_id}: {claim.artifact} missing symbol {symbol!r}"
                    + (f" ({claim.description})" if claim.description else "")
                )

    return errors
