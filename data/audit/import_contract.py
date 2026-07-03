"""AST import-boundary checks for agent/compute separation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

AGENT_TOOLS_PREFIX = "agent/tools/"

AGENT_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "science.compute.runner_dispatch",
        "science.compute.pathway_executor",
        "science.compute.dispatch_helpers",
        "science.compute.scheduler",
    }
)

AGENT_NORMALIZER_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "agent/tools/data_tools.py",
        "agent/tools/hypothesis/tools.py",
        "agent/tools/hypothesis/contradiction.py",
    }
)

AGENT_COMPUTE_PERSIST_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Legacy on-demand paths — burn down in favor of ingest-only compute.
        "agent/tools/cryptic/scan_phase.py",
        "agent/tools/cryptic/smd_dispatch.py",
        "agent/tools/cryptic/tool.py",
    }
)

AGENT_FORBIDDEN_COMPUTE_PERSIST_PREFIXES: tuple[str, ...] = (
    "science.compute.persist",
    "science.compute.persist_",
)

RUNNERS_PREFIX = "science/compute/runners/"
RUNNERS_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "data.normalizer.core",
        "data.normalizer",
    }
)


@dataclass(frozen=True)
class ImportViolation:
    path: str
    line_no: int
    module: str
    rule: str


def _iter_py_files(prefix: str) -> list[Path]:
    base = ROOT / prefix
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if p.is_file())


def _module_name(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        if not node.names:
            return None
        return node.names[0].name
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module
    return None


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _scan_file(path: Path, rules: list) -> list[ImportViolation]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError:
        return []

    violations: list[ImportViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        module = _module_name(node)
        if module is None:
            continue
        for rule_name, predicate in rules:
            if predicate(rel, module):
                violations.append(
                    ImportViolation(
                        path=rel,
                        line_no=node.lineno,
                        module=module,
                        rule=rule_name,
                    )
                )
    return violations


def audit_agent_tools_imports() -> list[str]:
    errors: list[str] = []

    def _forbidden_compute(module: str) -> bool:
        return module in AGENT_FORBIDDEN_MODULES

    def _forbidden_normalizer(rel: str, module: str) -> bool:
        if module != "data.normalizer.core" and not module.startswith("data.normalizer."):
            return False
        return rel not in AGENT_NORMALIZER_IMPORT_ALLOWLIST

    def _forbidden_persist(rel: str, module: str) -> bool:
        if rel in AGENT_COMPUTE_PERSIST_ALLOWLIST:
            return False
        return any(_matches_prefix(module, prefix) for prefix in AGENT_FORBIDDEN_COMPUTE_PERSIST_PREFIXES)

    rules = [
        ("agent_compute_dispatch_forbidden", lambda rel, mod: _forbidden_compute(mod)),
        ("agent_normalizer_import_allowlist", _forbidden_normalizer),
        ("agent_compute_persist_forbidden", _forbidden_persist),
    ]

    for path in _iter_py_files(AGENT_TOOLS_PREFIX):
        for hit in _scan_file(path, rules):
            errors.append(f"{hit.path}:{hit.line_no} [{hit.rule}] imports {hit.module}")
    return errors


def audit_runner_imports() -> list[str]:
    errors: list[str] = []

    def _forbidden(rel: str, module: str) -> bool:
        return module in RUNNERS_FORBIDDEN_MODULES or any(
            module.startswith(f"{blocked}.") for blocked in RUNNERS_FORBIDDEN_MODULES
        )

    rules = [("runner_normalizer_import_forbidden", _forbidden)]

    for path in _iter_py_files(RUNNERS_PREFIX):
        for hit in _scan_file(path, rules):
            errors.append(f"{hit.path}:{hit.line_no} [{hit.rule}] imports {hit.module}")
    return errors


def audit_import_contracts() -> list[str]:
    return audit_agent_tools_imports() + audit_runner_imports()
