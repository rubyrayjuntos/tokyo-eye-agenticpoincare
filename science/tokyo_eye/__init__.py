"""Tokyo Eye GNN product package (v7+). Production entry: TokyoEye.

Heavy imports (torch, the model class) stay lazy so the agent container can
import governance/readiness helpers without a GPU science stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TokyoEye", "TokyoEyeRunner"]


def __getattr__(name: str) -> Any:
    if name == "TokyoEye":
        from science.tokyo_eye.TokyoEye import TokyoEye

        return TokyoEye
    if name == "TokyoEyeRunner":
        from science.tokyo_eye.runner import TokyoEyeRunner

        return TokyoEyeRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
