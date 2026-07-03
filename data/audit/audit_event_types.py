"""AST lint: pipeline audit event_type must use events.py constants."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCAN_PREFIXES: tuple[str, ...] = (
    "shared/audit/",
    "science/",
    "agent/coordinator/",
    "agent/tools/",
    "data/",
)

SKIP_FILES: frozenset[str] = frozenset(
    {
        "shared/audit/events.py",
        "shared/audit/bus.py",
    }
)


@dataclass(frozen=True)
class EventTypeViolation:
    path: str
    line_no: int
    detail: str


def _iter_py_files() -> list[Path]:
    paths: list[Path] = []
    for prefix in SCAN_PREFIXES:
        base = ROOT / prefix
        if not base.exists():
            continue
        paths.extend(sorted(p for p in base.rglob("*.py") if p.is_file()))
    return paths


def _is_string_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _scan_file(path: Path) -> list[EventTypeViolation]:
    rel = path.relative_to(ROOT).as_posix()
    if rel in SKIP_FILES:
        return []

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError:
        return []

    violations: list[EventTypeViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name) and func.id == "emit_audit_event":
            if node.args and _is_string_literal(node.args[0]):
                violations.append(
                    EventTypeViolation(
                        rel,
                        node.lineno,
                        f"emit_audit_event({node.args[0].value!r}, ...) — use EVENT_* from shared.audit.events",
                    )
                )
            continue

        if isinstance(func, ast.Name) and func.id == "AuditEvent":
            for kw in node.keywords:
                if kw.arg == "event_type" and _is_string_literal(kw.value):
                    violations.append(
                        EventTypeViolation(
                            rel,
                            node.lineno,
                            f"AuditEvent(event_type={kw.value.value!r}) — use EVENT_* constant",
                        )
                    )
    return violations


def audit_event_type_literals() -> list[str]:
    errors: list[str] = []
    for path in _iter_py_files():
        for hit in _scan_file(path):
            errors.append(f"{hit.path}:{hit.line_no} {hit.detail}")
    return errors
