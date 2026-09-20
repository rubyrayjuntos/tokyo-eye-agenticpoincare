"""Static pure-hyp gate for tokyo_eye_equ_correct_start.

Fails if post-lift geometry substitutes remain in the EQU trunk forward:
  - _tangent_linear
  - TangentLinear / TangentPool class names
  - use_tangent_shortcut=True patterns
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ATTENTION = REPO / "science/tokyo_eye/v8/attention.py"
AFFINITY = REPO / "science/tokyo_eye/v8/affinity_head.py"


def _source(path: Path) -> str:
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _forbidden_hits(src: str) -> list[str]:
    hits: list[str] = []
    if "_tangent_linear" in src:
        hits.append("_tangent_linear")
    if "TangentLinear" in src:
        hits.append("TangentLinear")
    if "TangentPool" in src:
        hits.append("TangentPool")
    if "use_tangent_shortcut" in src and "use_tangent_shortcut = False" not in src:
        # presence of the flag name is suspicious unless hard-false only
        if "use_tangent_shortcut=True" in src or "use_tangent_shortcut = True" in src:
            hits.append("use_tangent_shortcut=True")
    # classic substitute pattern in source
    if "exp_map_zero" in src and "log_map_zero" in src and "nn.Linear" in src:
        # attention currently does lin(log_map_zero(...)) then exp — mark if call pattern exists
        if "lin(log_map_zero" in src or "lin(log_map_zero(" in src.replace(" ", ""):
            hits.append("lin(log_map_zero(...))")
        if "Linear" in src and "log_map_zero" in src and "exp_map_zero" in src:
            # tighter: method body of _tangent_linear
            if "def _tangent_linear" in src:
                hits.append("def _tangent_linear")
    return sorted(set(hits))


@pytest.mark.pure_hyp
def test_attention_has_no_tangent_geometry_substitutes():
    hits = _forbidden_hits(_source(ATTENTION))
    assert not hits, (
        "pure_hyp_pass FAIL — attention.py still contains forbidden post-lift "
        f"tangent geometry substitutes: {hits}. "
        "Implement on-manifold Q/K/V (plan §3) before train."
    )


@pytest.mark.pure_hyp
def test_affinity_head_has_no_tangent_pool_substitute():
    if not AFFINITY.is_file():
        pytest.skip("affinity_head.py absent")
    src = _source(AFFINITY)
    hits = []
    if "log_map_zero" in src and ("mean(" in src or "pool" in src.lower()):
        hits.append("tangent_pool_suspect")
    if "_tangent" in src:
        hits.append("_tangent*")
    # affinity may be parked; still Fail if used on trunk — static presence is Fail for strict card
    if "log_map_zero" in src and "exp_map_zero" in src:
        hits.append("log0_exp0_pair")
    assert not hits, (
        "pure_hyp_pass FAIL — affinity_head.py still looks like tangent geometry substitute: "
        f"{hits}. Park or rewrite before claiming pure_hyp_pass."
    )
