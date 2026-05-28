# Migrated from: new (Phase 2 implementation) on 2026-05-27
"""Asset type registry for the Normalizer.

The Normalizer only accepts payloads for registered asset types.
This module manages the registry and provides the workflow for
adding new types as the platform evolves.

Schema Evolution Workflow:
1. Define a new Pydantic payload model in normalizer_payloads.py
2. Register the asset type here with its metadata
3. Add a normalize_* method to the Normalizer class
4. Create the corresponding fact table migration
5. Add tests

This ensures new data types are deliberate, documented, and governed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetCategory(str, Enum):
    """High-level categorization of governed assets."""

    GNN_EMBEDDING = "gnn_embedding"
    DTIE_PHASE = "dtie_phase"
    DEHYDRON = "dehydron"
    SITE = "site"
    ANNOTATION = "annotation"
    COMPUTED_PROPERTY = "computed_property"
    RAG = "rag"  # Future


@dataclass
class AssetTypeDefinition:
    """Definition of a registered asset type."""

    name: str
    category: AssetCategory
    description: str
    fact_table: str  # The primary fact table this type writes to
    payload_model: str  # Fully qualified name of the Pydantic model
    grain: str  # Primary grain: "residue", "site", "structure"
    supports_v3: bool = True
    supports_v4: bool = True
    migration: str | None = None  # Migration that created the fact table
    added_date: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry of all accepted asset types
# ---------------------------------------------------------------------------

ASSET_TYPE_REGISTRY: dict[str, AssetTypeDefinition] = {
    "gnn_node_embedding": AssetTypeDefinition(
        name="gnn_node_embedding",
        category=AssetCategory.GNN_EMBEDDING,
        description="Per-residue GNN node embedding (Euclidean or Hyperbolic)",
        fact_table="fact_gnn_node_embedding",
        payload_model="science.dtie.common.normalizer_payloads.GNNOutputPayload",
        grain="residue",
        supports_v3=True,
        supports_v4=True,
        migration="004_gnn_embeddings.sql",
        added_date="2026-05-27",
    ),
    "dtie_phase3_persistence": AssetTypeDefinition(
        name="dtie_phase3_persistence",
        category=AssetCategory.DTIE_PHASE,
        description="Phase 3 witness persistence / topological output",
        fact_table="fact_phase3_persistence",
        payload_model="science.dtie.common.normalizer_payloads.Phase3PersistencePayload",
        grain="structure",  # Structure-level with optional residue contributions
        supports_v3=True,
        supports_v4=True,
        migration="005_dtie_phase_outputs.sql",
        added_date="2026-05-27",
    ),
    "dtie_phase1_witness": AssetTypeDefinition(
        name="dtie_phase1_witness",
        category=AssetCategory.DTIE_PHASE,
        description="Phase 1 witness embedding complex",
        fact_table="fact_phase1_witness_embedding",
        payload_model="TBD",
        grain="structure",
        supports_v3=True,
        supports_v4=True,
        migration="005_dtie_phase_outputs.sql",
        added_date="2026-05-27",
        notes="Payload model to be defined when Phase 1 integration begins",
    ),
    "dehydron": AssetTypeDefinition(
        name="dehydron",
        category=AssetCategory.DEHYDRON,
        description="Dehydron (underwrapped hydrogen bond) detection",
        fact_table="fact_dehydron",
        payload_model="TBD",
        grain="residue",  # Donor + acceptor residues
        supports_v3=True,
        supports_v4=True,
        migration="005_dtie_phase_outputs.sql",
        added_date="2026-05-27",
        notes="Payload model to be defined when dehydron integration begins",
    ),
    "computed_property": AssetTypeDefinition(
        name="computed_property",
        category=AssetCategory.COMPUTED_PROPERTY,
        description="Flexible per-residue or per-site computed property",
        fact_table="fact_computed_property",
        payload_model="TBD",
        grain="residue",
        supports_v3=True,
        supports_v4=True,
        migration="014_flexible_computed_properties.sql",
        added_date="2026-05-27",
        notes="Extensibility escape hatch for rapid iteration",
    ),
    "residue_graph_features": AssetTypeDefinition(
        name="residue_graph_features",
        category=AssetCategory.GNN_EMBEDDING,
        description="Per-residue graph topology features (degree, clustering, etc.)",
        fact_table="fact_residue_graph_features",
        payload_model="TBD",
        grain="residue",
        supports_v3=True,
        supports_v4=True,
        migration="004_gnn_embeddings.sql",
        added_date="2026-05-27",
    ),
}


def get_asset_type(name: str) -> AssetTypeDefinition | None:
    """Look up an asset type definition by name."""
    return ASSET_TYPE_REGISTRY.get(name)


def is_registered(name: str) -> bool:
    """Check if an asset type is registered."""
    return name in ASSET_TYPE_REGISTRY


def list_asset_types(category: AssetCategory | None = None) -> list[AssetTypeDefinition]:
    """List all registered asset types, optionally filtered by category."""
    types = list(ASSET_TYPE_REGISTRY.values())
    if category:
        types = [t for t in types if t.category == category]
    return types


def register_asset_type(definition: AssetTypeDefinition) -> None:
    """Register a new asset type.

    This should be called during application startup or in migration
    scripts when new types are introduced.

    Raises:
        ValueError: If the type name is already registered.
    """
    if definition.name in ASSET_TYPE_REGISTRY:
        raise ValueError(f"Asset type '{definition.name}' is already registered")
    ASSET_TYPE_REGISTRY[definition.name] = definition
