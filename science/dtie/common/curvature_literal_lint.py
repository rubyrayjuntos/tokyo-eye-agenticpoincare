"""Static scan for hardcoded curvature literals (TS-002 fallback violations)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SCAN_ROOTS = ("agent", "science", "data", "visualizer")
SKIP_DIR_NAMES = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
SKIP_FILE_SUFFIXES = {".min.js", ".map"}

# Entire files exempt from the policy (legacy / diagnostic only).
SKIP_FILE_PREFIXES: tuple[str, ...] = (
    "science/dtie/v3/",
    "science/dtie/v4/",
    "science/dtie/common/curvature_literal_lint.py",
    "agent/coordinator/main_legacy.py",
    "agent/pipeline/",
    "experiments/",
    "tests/",
)

# Line-level math uses of 1.0 with curvature variables — not TS-002 defaults.
LINE_ALLOW_RE = re.compile(
    r"1\.0\s*/\s*math\.sqrt\(curvature|"
    r"1\.0\s*-\s*curvature_c\s*\*|"
    r"1\.0\s*\+\s*\(2\.0\s*\*\s*curvature_c|"
    r"curvature-lint:\s*allow",
    re.IGNORECASE,
)

VIOLATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcurvature(?:_c)?\s*=\s*1\.0\b"), "curvature literal assignment"),
    (re.compile(r"\bcurvature(?:_c)?\s*:\s*float\s*=\s*1\.0\b"), "curvature default parameter"),
    (
        re.compile(r"""\.get\(\s*['"]curvature(?:_c)?['"]\s*,\s*1\.0\s*\)"""),
        "curvature dict get fallback",
    ),
    (re.compile(r"\bcurvature(?:_c)?\s+or\s+1\.0\b"), "curvature or 1.0"),
    (
        re.compile(r"""\.get\(\s*['"]curvature(?:_c)?['"]\s*\)\s+or\s+1\.0"""),
        "get(curvature) or 1.0",
    ),
    (re.compile(r"\bcurvature(?:_c)?\s*\?\?\s*1\.0\b"), "curvature ?? 1.0"),
    (
        re.compile(r"""['"]curvature(?:_c)?['"]\s*:\s*[^,}\n]*\?\?\s*1\.0"""),
        "curvature property ?? 1.0",
    ),
)


@dataclass(frozen=True)
class CurvatureLiteralViolation:
    path: str
    line_no: int
    line: str
    rule: str


def _should_scan_file(rel_posix: str) -> bool:
    if any(rel_posix.startswith(prefix) for prefix in SKIP_FILE_PREFIXES):
        return False
    if rel_posix.startswith("tests/"):
        return False
    return rel_posix.endswith((".py", ".ts", ".tsx"))


def scan_text(rel_path: str, text: str) -> list[CurvatureLiteralViolation]:
    violations: list[CurvatureLiteralViolation] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if LINE_ALLOW_RE.search(line):
            continue
        for pattern, rule in VIOLATION_PATTERNS:
            if pattern.search(line):
                violations.append(
                    CurvatureLiteralViolation(
                        path=rel_path,
                        line_no=line_no,
                        line=line.strip(),
                        rule=rule,
                    )
                )
                break
    return violations


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        base = ROOT / root_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in (".py", ".ts", ".tsx") and path.suffix not in SKIP_FILE_SUFFIXES:
                rel = path.relative_to(ROOT).as_posix()
                if _should_scan_file(rel):
                    files.append(path)
    return sorted(files)


def scan_repository() -> list[CurvatureLiteralViolation]:
    violations: list[CurvatureLiteralViolation] = []
    for path in iter_scan_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        violations.extend(scan_text(rel, text))
    return violations
