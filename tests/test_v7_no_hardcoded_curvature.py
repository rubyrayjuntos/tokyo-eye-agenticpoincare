"""Guard: Tokyo Eye v7 paths must not hardcode curvature pins."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = [
    ROOT / "science" / "tokyo_eye",
    ROOT / "experiments" / "training" / "v7",
]

BANNED_SUBSTRINGS = (
    "CANONICAL_V6_CURVATURE",
    "CANONICAL_TS002_CURVATURE",
    "DEFAULT_CURVATURE",
)

# Literal c=1.0 / curvature=1.0 in call kwargs is banned on v7 paths
# (tests may still use synthetic c in unit tests under tests/).
BANNED_KWARG_RE = re.compile(
    r"\b(curvature|c)\s*=\s*(1\.0|1\.|0\.7026273608207703)\b"
)


def _py_files() -> list[Path]:
    files: list[Path] = []
    for scope in SCOPES:
        if not scope.is_dir():
            continue
        files.extend(sorted(scope.rglob("*.py")))
    return files


def test_no_canonical_curvature_pins_in_v7_scopes() -> None:
    offenders: list[str] = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_SUBSTRINGS:
            if banned in text:
                offenders.append(f"{path.relative_to(ROOT)}: {banned}")
        if BANNED_KWARG_RE.search(text):
            offenders.append(f"{path.relative_to(ROOT)}: hardcoded curvature kwarg")
    assert not offenders, "banned curvature literals on v7 paths:\n" + "\n".join(
        offenders
    )


def test_tokyo_eye_uses_learned_log_c() -> None:
    src = (ROOT / "science" / "tokyo_eye" / "TokyoEye.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found_log_c = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "log_c":
                    found_log_c = True
    assert found_log_c, "TokyoEye must define learnable self.log_c"


def test_hyp_mp_module_exists() -> None:
    assert (ROOT / "science" / "tokyo_eye" / "hyperbolic_mp.py").is_file()
