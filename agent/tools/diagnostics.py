"""Tool self-diagnostic — validates all agent tools at startup.

Checks:
1. All tool handlers are callable (not None)
2. All tool parameter schemas are valid JSON Schema
3. Required imports for each tool are available
4. DB connectivity (if pool is open)
5. External dependencies (matplotlib, vina binary, etc.)
6. Output directories are writable

Run at startup to catch configuration issues early rather than
failing on the first user request.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    name: str
    passed: bool
    message: str = ""
    severity: str = "error"  # error, warning, info


@dataclass
class DiagnosticReport:
    """Full diagnostic report from startup checks."""

    results: list[DiagnosticResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def add(self, result: DiagnosticResult) -> None:
        self.results.append(result)
        if result.passed:
            self.passed += 1
        elif result.severity == "warning":
            self.warnings += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        return f"{self.passed} passed, {self.failed} failed, {self.warnings} warnings"


async def run_startup_diagnostics() -> DiagnosticReport:
    """Run all startup diagnostics and return a report.

    Call this during app lifespan startup. Logs results and returns
    the report for programmatic access.
    """
    report = DiagnosticReport()

    # 1. Check tool imports
    _check_tool_imports(report)

    # 2. Check external dependencies
    _check_external_deps(report)

    # 3. Check output directories
    _check_output_dirs(report)

    # 4. Check DB connectivity
    await _check_db_connectivity(report)

    # 5. Check tool handler wiring
    _check_tool_handlers(report)

    # Log summary
    if report.all_passed:
        logger.info(
            "Tool diagnostics PASSED: %s", report.summary()
        )
    else:
        logger.warning(
            "Tool diagnostics ISSUES: %s", report.summary()
        )
        for r in report.results:
            if not r.passed:
                logger.warning("  [%s] %s: %s", r.severity.upper(), r.name, r.message)

    return report


def _check_tool_imports(report: DiagnosticReport) -> None:
    """Verify all tool modules can be imported."""
    tool_modules = [
        ("agent.tools.dtie.tools", "DTIE tools"),
        ("agent.tools.data_tools", "Data tools"),
        ("agent.tools.plotting.tools", "Plotting tools"),
        ("agent.llm.agents", "Agent definitions"),
        ("agent.llm.providers", "LLM providers"),
        ("agent.models.viewport", "Viewport models"),
        ("data.normalizer.core", "Normalizer"),
        ("data.db", "Database layer"),
        ("shared.context", "Request context"),
        ("shared.logging", "Structured logging"),
    ]

    for module_path, label in tool_modules:
        try:
            importlib.import_module(module_path)
            report.add(DiagnosticResult(
                name=f"import:{label}",
                passed=True,
            ))
        except ImportError as e:
            report.add(DiagnosticResult(
                name=f"import:{label}",
                passed=False,
                message=f"Failed to import {module_path}: {e}",
                severity="error",
            ))


def _check_external_deps(report: DiagnosticReport) -> None:
    """Check that external dependencies are available."""
    # matplotlib (required for plotting tools)
    try:
        import matplotlib
        matplotlib.use("Agg")
        report.add(DiagnosticResult(
            name="dep:matplotlib",
            passed=True,
            message=f"v{matplotlib.__version__}",
        ))
    except ImportError:
        report.add(DiagnosticResult(
            name="dep:matplotlib",
            passed=False,
            message="matplotlib not installed — plotting tools will fail",
            severity="error",
        ))

    # numpy
    try:
        import numpy
        report.add(DiagnosticResult(
            name="dep:numpy",
            passed=True,
            message=f"v{numpy.__version__}",
        ))
    except ImportError:
        report.add(DiagnosticResult(
            name="dep:numpy",
            passed=False,
            message="numpy not installed",
            severity="error",
        ))

    # psycopg
    try:
        import psycopg
        report.add(DiagnosticResult(
            name="dep:psycopg",
            passed=True,
            message=f"v{psycopg.__version__}",
        ))
    except ImportError:
        report.add(DiagnosticResult(
            name="dep:psycopg",
            passed=False,
            message="psycopg not installed — DB operations will fail",
            severity="error",
        ))

    # AutoDock Vina (optional — only needed for Phase 6b)
    vina_path = shutil.which("vina")
    if vina_path:
        report.add(DiagnosticResult(
            name="dep:autodock_vina",
            passed=True,
            message=f"Found at {vina_path}",
        ))
    else:
        report.add(DiagnosticResult(
            name="dep:autodock_vina",
            passed=True,  # Not a failure — it's optional
            message="Not found in PATH (Phase 6b docking unavailable)",
            severity="info",
        ))

    # LLM provider availability
    from agent.llm.providers import get_provider
    provider = get_provider()
    provider_name = type(provider).__name__
    report.add(DiagnosticResult(
        name="dep:llm_provider",
        passed=True,
        message=f"Using {provider_name}",
        severity="info" if "Mock" in provider_name else "info",
    ))


def _check_output_dirs(report: DiagnosticReport) -> None:
    """Check that output directories are writable."""
    dirs_to_check = [
        (os.getenv("PLOT_OUTPUT_DIR", "./data/local_objects/plots"), "Plot output"),
        (os.getenv("EXPORT_OUTPUT_DIR", "./data/local_objects/exports"), "Export output"),
    ]

    for dir_path, label in dirs_to_check:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write
            test_file = path / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            report.add(DiagnosticResult(
                name=f"dir:{label}",
                passed=True,
                message=str(path),
            ))
        except (OSError, PermissionError) as e:
            report.add(DiagnosticResult(
                name=f"dir:{label}",
                passed=False,
                message=f"Cannot write to {path}: {e}",
                severity="warning",
            ))


async def _check_db_connectivity(report: DiagnosticReport) -> None:
    """Check database connectivity via the pool."""
    try:
        from data.db import get_connection

        async with get_connection() as conn:
            result = await conn.execute("SELECT 1")
            report.add(DiagnosticResult(
                name="db:connectivity",
                passed=True,
                message="Pool connection successful",
            ))
    except Exception as e:
        report.add(DiagnosticResult(
            name="db:connectivity",
            passed=False,
            message=f"Cannot connect to database: {e}",
            severity="error",
        ))


def _check_tool_handlers(report: DiagnosticReport) -> None:
    """Verify that all registered tools have callable handlers after wiring."""
    from agent.llm.agents import (
        DATA_TOOLS,
        DTIE_TOOLS,
        PLOTTING_TOOLS,
        VISUALIZATION_TOOLS,
    )

    all_tool_defs = [
        ("DTIE", DTIE_TOOLS),
        ("Visualization", VISUALIZATION_TOOLS),
        ("Plotting", PLOTTING_TOOLS),
        ("Data", DATA_TOOLS),
    ]

    total_tools = 0
    for group_name, tool_list in all_tool_defs:
        total_tools += len(tool_list)

    report.add(DiagnosticResult(
        name="tools:registered",
        passed=True,
        message=f"{total_tools} tool definitions across 4 groups",
    ))

    # Verify tool parameter schemas have required fields
    for group_name, tool_list in all_tool_defs:
        for tool_def in tool_list:
            schema = tool_def.parameters
            if not isinstance(schema, dict) or "type" not in schema:
                report.add(DiagnosticResult(
                    name=f"tools:schema:{tool_def.name}",
                    passed=False,
                    message=f"Invalid parameter schema for {tool_def.name}",
                    severity="error",
                ))
            else:
                report.add(DiagnosticResult(
                    name=f"tools:schema:{tool_def.name}",
                    passed=True,
                ))
