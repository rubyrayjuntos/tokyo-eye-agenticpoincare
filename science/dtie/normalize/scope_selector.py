"""Scope selector — reads computation scope and applies normalization protocol.

Bridges between the stored structure_computation_scope and the protocol
registry to produce a NormalizedView for downstream consumers like GraphBuilder.

Requirements: 4.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol as TypingProtocol

from science.dtie.ingest.chain_scorer import ComputationScope, QualityFilters
from science.dtie.ingest.parser import ParsedStructure
from science.dtie.normalize.protocols.base import (
    NormalizedView,
    ProtocolName,
    get_protocol,
)

logger = logging.getLogger(__name__)


class ScopeDB(TypingProtocol):
    """Database protocol for fetching computation scope."""

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...


@dataclass
class StoredScope:
    """Computation scope as read from the database."""

    structure_id: str
    primary_chain_ids: list[str]
    reference_chain: str
    exclude_chain_ids: list[str]
    scope_source: str
    selection_reason: str
    normalization_protocol: str
    quality_filters: dict | None
    model_index: int


class ScopeSelector:
    """Reads scope from DB and applies the named protocol.

    Usage:
        selector = ScopeSelector(db=connection)
        view = await selector.select_and_apply("4obe", parsed_structure)
    """

    def __init__(self, db: ScopeDB):
        self._db = db

    async def fetch_scope(self, structure_id: str) -> StoredScope | None:
        """Fetch computation scope for a structure from the database.

        Args:
            structure_id: Canonical structure_id.

        Returns:
            StoredScope or None if no scope stored.
        """
        row = await self._db.fetch_one(
            """
            SELECT
                structure_id,
                primary_chain_ids,
                reference_chain,
                exclude_chain_ids,
                scope_source,
                selection_reason,
                normalization_protocol,
                quality_filters,
                model_index
            FROM structure_computation_scope
            WHERE structure_id = :structure_id
            """,
            {"structure_id": structure_id},
        )

        if row is None:
            return None

        return StoredScope(
            structure_id=row["structure_id"],
            primary_chain_ids=row["primary_chain_ids"] or [],
            reference_chain=row["reference_chain"],
            exclude_chain_ids=row["exclude_chain_ids"] or [],
            scope_source=row["scope_source"],
            selection_reason=row.get("selection_reason", ""),
            normalization_protocol=row["normalization_protocol"],
            quality_filters=row.get("quality_filters"),
            model_index=row.get("model_index", 1),
        )

    def scope_to_computation_scope(self, stored: StoredScope) -> ComputationScope:
        """Convert a StoredScope into a ComputationScope for protocol application."""
        quality_filters = None
        if stored.quality_filters:
            quality_filters = QualityFilters(
                max_b_factor_threshold=stored.quality_filters.get(
                    "max_b_factor_threshold", 100.0
                ),
                min_resolution=stored.quality_filters.get("min_resolution"),
            )

        return ComputationScope(
            primary_chain_ids=stored.primary_chain_ids,
            reference_chain=stored.reference_chain,
            exclude_chain_ids=stored.exclude_chain_ids,
            scope_source=stored.scope_source,
            selection_reason=stored.selection_reason,
            normalization_protocol=stored.normalization_protocol,
            quality_filters=quality_filters,
        )

    async def select_and_apply(
        self,
        structure_id: str,
        parsed: ParsedStructure,
        *,
        protocol_override: str | None = None,
    ) -> NormalizedView:
        """Fetch scope, resolve protocol, and produce a normalized view.

        Args:
            structure_id: Canonical structure_id.
            parsed: Raw ParsedStructure from BinaryCIF parser.
            protocol_override: Override the stored protocol name.

        Returns:
            NormalizedView produced by the protocol.

        Raises:
            ValueError: If no scope found for the structure.
            KeyError: If the protocol name is not registered.
        """
        stored_scope = await self.fetch_scope(structure_id)
        if stored_scope is None:
            raise ValueError(
                f"No computation scope found for structure '{structure_id}'. "
                "Run ingestion first."
            )

        scope = self.scope_to_computation_scope(stored_scope)

        protocol_name = protocol_override or stored_scope.normalization_protocol
        protocol = get_protocol(protocol_name)

        logger.info(
            "Applying protocol '%s' v%d to structure '%s' (scope_source=%s)",
            protocol.name.value,
            protocol.version,
            structure_id,
            stored_scope.scope_source,
        )

        view = protocol.apply(parsed, scope)
        return view

    def apply_protocol_direct(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
        protocol_name: str | None = None,
    ) -> NormalizedView:
        """Apply a protocol directly without DB lookup.

        Useful for testing or when scope is already in memory.

        Args:
            parsed: Raw ParsedStructure.
            scope: ComputationScope (already resolved).
            protocol_name: Protocol to apply. Uses scope's protocol if None.

        Returns:
            NormalizedView from the protocol.
        """
        name = protocol_name or scope.normalization_protocol
        protocol = get_protocol(name)
        return protocol.apply(parsed, scope)
