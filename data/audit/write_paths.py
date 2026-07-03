"""Scan repository Python sources for governed-table write bypasses."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("science", "agent", "data", "tests")
SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
INSERT_RE = re.compile(r"INSERT\s+INTO\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
UPDATE_RE = re.compile(r"UPDATE\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
COPY_RE = re.compile(r"COPY\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
GOVERNED_PREFIXES = ("fact_", "dim_", "governed_", "embedding_", "provenance_", "hypothesis")
CANONICAL_NORMALIZER = "data/normalizer/core.py"

# Contract surfaces that must not bypass the Normalizer for fact/dim writes.
ZERO_BYPASS_PREFIXES: tuple[str, ...] = (
    "agent/tools/",
    "agent/coordinator/",
)

# Atomic runners may seed provenance_run before Normalizer payloads land.
RUNNER_PREFIX = "science/compute/runners/"
RUNNER_ALLOWED_TABLES = frozenset({"provenance_run", "provenance_event"})

# Grandfathered legacy bypass files — frozen; new files or tables fail the gate.
GRANDFATHERED_BYPASS_FILES: frozenset[str] = frozenset(
    {
        "agent/pipeline/jobs/gnn_inference.py",
        "agent/pipeline/normalizer/cdd.py",
        "agent/pipeline/normalizer/dehydron.py",
        "agent/pipeline/normalizer/energy.py",
        "agent/pipeline/normalizer/folding.py",
        "agent/pipeline/normalizer/gnn.py",
        "agent/pipeline/normalizer/hdx.py",
        "agent/pipeline/normalizer/jobs.py",
        "agent/pipeline/normalizer/redzone.py",
        "agent/pipeline/normalizer/synthesis.py",
        "agent/pipeline/normalizer/validation.py",
        "agent/pipeline/normalizer/void.py",
        "agent/pipeline/normalizer/structure.py",
        "agent/pipeline/services/gnn_writeback.py",
        "science/api/routers/ingest.py",
        "science/compute/persist.py",
        "science/compute/provenance.py",
        "science/dtie/common/adapters/buffering_atlas_adapter.py",
        "science/dtie/common/ingestion.py",
        "science/dtie/common/structure_parser.py",
        "science/dtie/v5/orchestrator/pipeline.py",
        "science/dtie/v5/resistance/profiler.py",
        "science/dtie/v5/resistance/sdrp_engine.py",
        "science/dtie/v5/workers/hyperbolic_distance_populator.py",
        "tests/test_hypothesis_contradiction.py",
        "tests/test_ingest_audit_controls.py",
    }
)

# UPDATE paths outside the Normalizer that are frozen (hypothesis/cryptic legacy).
UPDATE_GRANDFATHERED_FILES: frozenset[str] = frozenset(
    {
        "agent/orchestration/orchestrator.py",
        "agent/pipeline/jobs/tier1.py",
        "agent/pipeline/normalizer/jobs.py",
        "agent/tools/cryptic/tool.py",
        "agent/tools/hypothesis/tools.py",
        "agent/tools/hypothesis/contradiction.py",
        "science/compute/cryptic/md_validate.py",
    }
)


@dataclass(frozen=True)
class WritePathHit:
    path: str
    line_no: int
    table: str
    category: str
    operation: str = "insert"


def classify(path: str, table: str) -> str:
    if path == CANONICAL_NORMALIZER:
        return "canonical"
    if table.startswith("fact_"):
        return "bypass_fact"
    if table.startswith(("dim_", "governed_", "embedding_", "provenance_", "hypothesis")):
        return "bypass_governance"
    return "other"


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        base = ROOT / scan_dir
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def _governed_tables_on_line(line: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for operation, pattern in (
        ("insert", INSERT_RE),
        ("update", UPDATE_RE),
        ("copy", COPY_RE),
    ):
        for match in pattern.finditer(line):
            table = match.group(1).lower()
            if table.startswith(GOVERNED_PREFIXES):
                hits.append((operation, table))
    return hits


def scan_repository(
    *,
    include_tests: bool = True,
    operations: frozenset[str] = frozenset({"insert"}),
) -> list[WritePathHit]:
    hits: list[WritePathHit] = []
    for path in iter_source_files():
        rel = path.relative_to(ROOT).as_posix()
        if not include_tests and rel.startswith("tests/"):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for operation, table in _governed_tables_on_line(line):
                if operation not in operations:
                    continue
                category = classify(rel, table)
                if category == "canonical":
                    continue
                hits.append(
                    WritePathHit(
                        path=rel,
                        line_no=lineno,
                        table=table,
                        category=category,
                        operation=operation,
                    )
                )
    return hits


def _gate_violation(hit: WritePathHit) -> str | None:
    if hit.operation == "update":
        if hit.path.startswith(RUNNER_PREFIX):
            if hit.table in RUNNER_ALLOWED_TABLES:
                return None
            return (
                f"{hit.path}:{hit.line_no} runner UPDATE may only touch "
                f"{sorted(RUNNER_ALLOWED_TABLES)}; found {hit.table}"
            )
        if hit.path in UPDATE_GRANDFATHERED_FILES:
            return None
        if hit.path in GRANDFATHERED_BYPASS_FILES:
            return None
        return (
            f"{hit.path}:{hit.line_no} ungrandfathered UPDATE bypass "
            f"({hit.category} -> {hit.table})"
        )

    if any(hit.path.startswith(prefix) for prefix in ZERO_BYPASS_PREFIXES):
        return (
            f"{hit.path}:{hit.line_no} bypasses Normalizer "
            f"({hit.operation} {hit.category} -> {hit.table})"
        )

    if hit.path.startswith(RUNNER_PREFIX):
        if hit.table not in RUNNER_ALLOWED_TABLES:
            return (
                f"{hit.path}:{hit.line_no} runner may only INSERT "
                f"{sorted(RUNNER_ALLOWED_TABLES)}; found {hit.table}"
            )
        return None

    if hit.path not in GRANDFATHERED_BYPASS_FILES:
        return (
            f"{hit.path}:{hit.line_no} ungrandfathered bypass "
            f"({hit.operation} {hit.category} -> {hit.table})"
        )
    return None


def audit_write_path_gate() -> list[str]:
    """Return human-readable violations for CI gate (INSERT + runner/hypothesis UPDATE)."""
    errors: list[str] = []
    hits = scan_repository(
        include_tests=False,
        operations=frozenset({"insert", "update"}),
    )
    for hit in hits:
        violation = _gate_violation(hit)
        if violation:
            errors.append(violation)
    return errors


def grouped_hits() -> dict[str, list[WritePathHit]]:
    grouped: dict[str, list[WritePathHit]] = defaultdict(list)
    for hit in scan_repository(
        include_tests=True,
        operations=frozenset({"insert", "update", "copy"}),
    ):
        grouped[classify(hit.path, hit.table)].append(hit)

    canonical: list[WritePathHit] = []
    for path in iter_source_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel != CANONICAL_NORMALIZER:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            for operation, table in _governed_tables_on_line(line):
                if operation != "insert":
                    continue
                if table.startswith(GOVERNED_PREFIXES):
                    canonical.append(
                        WritePathHit(
                            path=rel,
                            line_no=lineno,
                            table=table,
                            category="canonical",
                            operation=operation,
                        )
                    )
    grouped["canonical"] = canonical
    return grouped
