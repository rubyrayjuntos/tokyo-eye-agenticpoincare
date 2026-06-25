"""Topology cache for Poincaré disc computation results.

LRU cache keyed by (structure_id, run_id) to avoid recomputing
disc topology on every chat message.

Feature: poincare-visual-context
Requirements: 4.4, 4.5
"""

from __future__ import annotations

from collections import OrderedDict

from agent.tools.disc_topology import DiscTopologyResult


class TopologyCache:
    """LRU cache for computed disc topologies, keyed by (structure_id, run_id).

    Thread-safety note: this cache is designed for single-threaded async use
    within the agent coordinator process. If used across threads, external
    locking is required.
    """

    def __init__(self, max_size: int = 50):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str], DiscTopologyResult] = OrderedDict()

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, structure_id: str, run_id: str) -> DiscTopologyResult | None:
        """Retrieve cached topology result.

        Returns None on cache miss. Moves entry to most-recently-used on hit.
        """
        key = (structure_id, run_id)
        if key not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, result: DiscTopologyResult) -> None:
        """Store a topology result in the cache.

        If the cache is at capacity, evicts the least-recently-used entry.
        If the key already exists, updates the value and moves to most-recent.
        """
        key = (result.structure_id, result.run_id)
        if key in self._cache:
            # Update existing entry and move to end
            self._cache.move_to_end(key)
            self._cache[key] = result
        else:
            # Insert new entry
            self._cache[key] = result
            # Evict LRU if over capacity
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, structure_id: str) -> None:
        """Remove all cached entries for a given structure_id.

        Called when a new GNN run completes, so stale topology is not served.
        """
        keys_to_remove = [
            key for key in self._cache if key[0] == structure_id
        ]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()
