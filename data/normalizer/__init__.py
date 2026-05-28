# Migrated from: new (Phase 1 implementation) on 2026-05-27
"""Tokyo Eye Normalizer — Governed Write Path.

Deployment Model Decision (LOCKED):
    Library-first, service-optional.

    The Normalizer is implemented as a Python library that can be:
    1. Imported directly by science code running in the same process (low latency)
    2. Wrapped in a thin FastAPI service for remote callers (agent, external tools)

    This avoids the false choice between "in-process only" and "service only."
    Science code (v3/v4 pipelines) will typically use the library directly.
    The agent coordinator will call the FastAPI wrapper.

See: data/NORMALIZER_DESIGN.md for the full architecture.
"""

from data.normalizer.core import Normalizer, NormalizerError
from data.normalizer.registry import (
    ASSET_TYPE_REGISTRY,
    AssetCategory,
    get_asset_type,
    is_registered,
    list_asset_types,
    register_asset_type,
)

__all__ = [
    "Normalizer",
    "NormalizerError",
    "ASSET_TYPE_REGISTRY",
    "AssetCategory",
    "get_asset_type",
    "is_registered",
    "list_asset_types",
    "register_asset_type",
]
