# Migrated from: new (Phase 1/2 implementation) on 2026-05-27
"""Materialized view refresh management.

Handles refresh strategies for the production materialized views:
- On-demand refresh (after pipeline runs complete)
- Scheduled refresh (periodic background)
- Selective refresh (only specific views)

The refresh strategy is critical for balancing data freshness against
query performance. Materialized views are the performance layer that
sits between the governed fact tables and the consumers (agent, visualizer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RefreshDB(Protocol):
    """Database protocol for view refresh operations."""

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None: ...
    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None: ...


class RefreshStrategy(str, Enum):
    """When to refresh a materialized view."""

    ON_DEMAND = "on_demand"       # After pipeline runs complete
    SCHEDULED = "scheduled"       # Periodic (e.g., every 5 minutes)
    MANUAL = "manual"             # Only when explicitly requested


@dataclass
class MaterializedViewConfig:
    """Configuration for a managed materialized view."""

    name: str
    strategy: RefreshStrategy
    description: str
    depends_on: list[str]  # Tables that, when written to, trigger refresh
    concurrent: bool = True  # Use CONCURRENTLY (no lock, requires unique index)
    priority: int = 1  # Lower = refreshed first


# Registry of all managed materialized views
MANAGED_VIEWS: list[MaterializedViewConfig] = [
    MaterializedViewConfig(
        name="mv_residue_latest_hyperbolic",
        strategy=RefreshStrategy.ON_DEMAND,
        description="Latest hyperbolic embedding per residue (for viewport + agent)",
        depends_on=["fact_gnn_node_embedding"],
        concurrent=True,
        priority=1,
    ),
    MaterializedViewConfig(
        name="mv_active_site_summary",
        strategy=RefreshStrategy.ON_DEMAND,
        description="Site summary with aggregated residue metrics",
        depends_on=["dim_site", "bridge_site_residue", "fact_gnn_node_embedding"],
        concurrent=True,
        priority=2,
    ),
    MaterializedViewConfig(
        name="mv_residue_current_state",
        strategy=RefreshStrategy.SCHEDULED,
        description="Broad residue state (embeddings + dehydrons + sites)",
        depends_on=["fact_gnn_node_embedding", "fact_dehydron", "dim_site"],
        concurrent=True,
        priority=3,
    ),
]


@dataclass
class RefreshResult:
    """Result of a view refresh operation."""

    view_name: str
    success: bool
    duration_ms: int
    refreshed_at: datetime
    error: str | None = None


class ViewRefresher:
    """Manages materialized view refresh operations.

    Usage:
        refresher = ViewRefresher(db=connection)

        # After a GNN pipeline run completes:
        results = await refresher.refresh_after_write("fact_gnn_node_embedding")

        # Scheduled refresh of all eligible views:
        results = await refresher.refresh_scheduled()

        # Manual refresh of a specific view:
        result = await refresher.refresh_view("mv_residue_latest_hyperbolic")
    """

    def __init__(self, db: RefreshDB):
        self._db = db

    async def refresh_after_write(self, table_name: str) -> list[RefreshResult]:
        """Refresh all on-demand views that depend on the given table.

        Called by the Normalizer after a successful write to trigger
        downstream view updates.

        Args:
            table_name: The fact table that was just written to.

        Returns:
            List of refresh results for affected views.
        """
        affected = [
            v for v in MANAGED_VIEWS
            if v.strategy == RefreshStrategy.ON_DEMAND
            and table_name in v.depends_on
        ]

        # Sort by priority (lower first)
        affected.sort(key=lambda v: v.priority)

        results = []
        for view_config in affected:
            result = await self.refresh_view(view_config.name, concurrent=view_config.concurrent)
            results.append(result)

        return results

    async def refresh_scheduled(self) -> list[RefreshResult]:
        """Refresh all views with SCHEDULED strategy.

        Intended to be called by a periodic background task.
        """
        scheduled = [
            v for v in MANAGED_VIEWS
            if v.strategy == RefreshStrategy.SCHEDULED
        ]
        scheduled.sort(key=lambda v: v.priority)

        results = []
        for view_config in scheduled:
            result = await self.refresh_view(view_config.name, concurrent=view_config.concurrent)
            results.append(result)

        return results

    async def refresh_view(
        self, view_name: str, concurrent: bool = True
    ) -> RefreshResult:
        """Refresh a single materialized view.

        Args:
            view_name: Name of the materialized view (must be in MANAGED_VIEWS registry).
            concurrent: If True, use CONCURRENTLY (no read lock).

        Returns:
            RefreshResult with timing and status.
        """
        import time

        # Validate view_name against the registry to prevent SQL injection
        valid_names = {v.name for v in MANAGED_VIEWS}
        if view_name not in valid_names:
            return RefreshResult(
                view_name=view_name,
                success=False,
                duration_ms=0,
                refreshed_at=datetime.now(timezone.utc),
                error=f"Unknown view: {view_name}. Must be one of: {valid_names}",
            )

        start = time.monotonic()
        concurrently = "CONCURRENTLY" if concurrent else ""

        try:
            await self._db.execute(
                f"REFRESH MATERIALIZED VIEW {concurrently} {view_name}"
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            logger.info(
                "Refreshed materialized view %s in %dms",
                view_name,
                duration_ms,
            )

            return RefreshResult(
                view_name=view_name,
                success=True,
                duration_ms=duration_ms,
                refreshed_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "Failed to refresh materialized view %s: %s",
                view_name,
                e,
            )
            return RefreshResult(
                view_name=view_name,
                success=False,
                duration_ms=duration_ms,
                refreshed_at=datetime.now(timezone.utc),
                error=str(e),
            )

    async def refresh_all(self) -> list[RefreshResult]:
        """Refresh ALL managed materialized views regardless of strategy.

        Use sparingly — intended for maintenance windows or initial setup.
        """
        all_views = sorted(MANAGED_VIEWS, key=lambda v: v.priority)
        results = []
        for view_config in all_views:
            result = await self.refresh_view(view_config.name, concurrent=view_config.concurrent)
            results.append(result)
        return results

    def get_view_config(self, view_name: str) -> MaterializedViewConfig | None:
        """Look up configuration for a specific view."""
        for v in MANAGED_VIEWS:
            if v.name == view_name:
                return v
        return None

    def list_views(self) -> list[MaterializedViewConfig]:
        """List all managed materialized views."""
        return MANAGED_VIEWS.copy()
