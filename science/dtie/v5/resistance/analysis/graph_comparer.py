"""Graph Comparer — topological evidence layer for resistance classification.

Provides structural physics evidence to supplement GNN-based classification:
1. Betweenness Centrality: detects if a mutation site is a "bridge" node
2. Edge Perturbation Clustering: quantifies local vs distributed edge changes
3. Hub Connectivity: measures if mutation disrupts paths to propagation hubs

This module operates on the raw protein contact graph (edge_index + edge_attr)
and does NOT require GNN inference. It provides "white box" structural evidence
that can break ties when the GNN classification is ambiguous (Hybrid).

Usage:
    comparer = GraphComparer(edge_index, edge_attr, residue_indices)
    evidence = comparer.analyze_mutation(mutation_site_idx, hub_indices)

    # evidence.bottleneck_disrupted → True if mutation site is high-centrality
    # evidence.edge_cluster_locality → ratio of local vs total edge changes
    # evidence.hub_path_severed → True if mutation increases path length to hubs
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopologicalEvidence:
    """Structural evidence from graph topology analysis."""

    # Betweenness centrality of the mutation site (0-1 normalized)
    site_betweenness: float

    # Percentile rank of site betweenness among all nodes (0-100)
    site_betweenness_percentile: float

    # Whether the mutation site is a topological bottleneck (top 10%)
    bottleneck_disrupted: bool

    # Fraction of perturbed edges that are within 2 hops of mutation site
    # High (>0.8) = steric/local; Low (<0.5) = allosteric/distributed
    edge_cluster_locality: float

    # Number of hub residues whose shortest path from mutation site
    # passes through high-centrality intermediates
    hub_paths_through_bridges: int

    # Total number of hubs analyzed
    total_hubs: int

    # Average betweenness of nodes on shortest paths to hubs
    avg_path_betweenness: float

    # Classification suggestion based purely on topology
    topology_suggestion: str  # "steric", "allosteric", or "ambiguous"

    # Confidence in the topology suggestion (0-1)
    topology_confidence: float


class GraphComparer:
    """Topological analysis of protein contact graphs for resistance evidence.

    Computes structural features that provide physics-based evidence for
    resistance mechanism classification, independent of GNN inference.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        num_nodes: int,
        residue_indices: list[int] | None = None,
    ):
        """Initialize with graph structure.

        Args:
            edge_index: [2, E] array of source/target node pairs.
            num_nodes: Total number of nodes in the graph.
            residue_indices: Optional mapping from node index to residue number.
        """
        self._edge_index = edge_index
        self._num_nodes = num_nodes
        self._residue_indices = residue_indices or list(range(num_nodes))

        # Build adjacency list (undirected)
        self._adj: list[set[int]] = [set() for _ in range(num_nodes)]
        num_edges = edge_index.shape[1]
        for e in range(num_edges):
            src = int(edge_index[0, e])
            tgt = int(edge_index[1, e])
            if src < num_nodes and tgt < num_nodes:
                self._adj[src].add(tgt)
                self._adj[tgt].add(src)

        # Lazily computed betweenness centrality
        self._betweenness: np.ndarray | None = None

    @property
    def betweenness(self) -> np.ndarray:
        """Compute or return cached betweenness centrality for all nodes."""
        if self._betweenness is None:
            self._betweenness = self._compute_betweenness()
        return self._betweenness

    def analyze_mutation(
        self,
        mutation_node_idx: int,
        hub_node_indices: list[int],
        perturbed_edge_mask: np.ndarray | None = None,
    ) -> TopologicalEvidence:
        """Analyze the topological significance of a mutation site.

        Args:
            mutation_node_idx: Graph index of the mutated residue.
            hub_node_indices: Graph indices of propagation hub residues.
            perturbed_edge_mask: Optional boolean mask [E] indicating which
                edges were perturbed by the mutation operator.

        Returns:
            TopologicalEvidence with structural analysis results.
        """
        bc = self.betweenness

        # 1. Site betweenness and percentile
        site_bc = bc[mutation_node_idx]
        percentile = float(np.sum(bc < site_bc) / max(len(bc) - 1, 1) * 100)
        bottleneck = percentile >= 90.0  # Top 10% = bottleneck

        # 2. Edge cluster locality
        if perturbed_edge_mask is not None:
            edge_cluster_locality = self._compute_edge_locality(
                mutation_node_idx, perturbed_edge_mask
            )
        else:
            # Default: compute based on 2-hop neighborhood size ratio
            two_hop = self._get_n_hop_neighborhood(mutation_node_idx, 2)
            edge_cluster_locality = len(two_hop) / max(self._num_nodes, 1)

        # 3. Hub path analysis
        hub_paths_through_bridges = 0
        path_betweenness_values: list[float] = []

        for hub_idx in hub_node_indices:
            path = self._bfs_path(mutation_node_idx, hub_idx)
            if path is None or len(path) < 3:
                continue

            # Check intermediate nodes on the path
            intermediates = path[1:-1]
            intermediate_bc = [bc[n] for n in intermediates]
            avg_bc = float(np.mean(intermediate_bc)) if intermediate_bc else 0.0
            path_betweenness_values.append(avg_bc)

            # Count paths that go through high-centrality nodes (top 25%)
            bc_threshold = float(np.percentile(bc, 75))
            if any(bc[n] >= bc_threshold for n in intermediates):
                hub_paths_through_bridges += 1

        avg_path_bc = float(np.mean(path_betweenness_values)) if path_betweenness_values else 0.0

        # 4. Classification suggestion
        topology_suggestion, topology_confidence = self._suggest_classification(
            bottleneck=bottleneck,
            edge_locality=edge_cluster_locality,
            hub_bridge_ratio=hub_paths_through_bridges / max(len(hub_node_indices), 1),
            site_percentile=percentile,
        )

        return TopologicalEvidence(
            site_betweenness=site_bc,
            site_betweenness_percentile=percentile,
            bottleneck_disrupted=bottleneck,
            edge_cluster_locality=edge_cluster_locality,
            hub_paths_through_bridges=hub_paths_through_bridges,
            total_hubs=len(hub_node_indices),
            avg_path_betweenness=avg_path_bc,
            topology_suggestion=topology_suggestion,
            topology_confidence=topology_confidence,
        )

    def _suggest_classification(
        self,
        bottleneck: bool,
        edge_locality: float,
        hub_bridge_ratio: float,
        site_percentile: float,
    ) -> tuple[str, float]:
        """Suggest a classification based purely on topology.

        Decision logic:
        - If site is a bottleneck AND paths to hubs go through bridges → allosteric
        - If site has high edge locality AND low hub bridge ratio → steric
        - Otherwise → ambiguous

        Returns:
            (suggestion, confidence) tuple.
        """
        allosteric_score = 0.0
        steric_score = 0.0

        # Bottleneck = allosteric signal (mutation breaks a bridge)
        if bottleneck:
            allosteric_score += 0.4
        elif site_percentile >= 70:
            allosteric_score += 0.2

        # High hub bridge ratio = allosteric (paths to hubs cross critical nodes)
        if hub_bridge_ratio >= 0.5:
            allosteric_score += 0.4
        elif hub_bridge_ratio >= 0.25:
            allosteric_score += 0.2

        # High edge locality = steric (changes are concentrated locally)
        if edge_locality >= 0.15:
            steric_score += 0.3
        elif edge_locality >= 0.08:
            steric_score += 0.15

        # Low hub bridge ratio = steric (no pathway disruption)
        if hub_bridge_ratio < 0.1:
            steric_score += 0.3

        # Low betweenness = steric (not a bridge node)
        if site_percentile < 50:
            steric_score += 0.2

        if allosteric_score > steric_score + 0.15:
            return "allosteric", min(allosteric_score, 1.0)
        elif steric_score > allosteric_score + 0.15:
            return "steric", min(steric_score, 1.0)
        else:
            return "ambiguous", max(allosteric_score, steric_score) * 0.5

    def _compute_betweenness(self) -> np.ndarray:
        """Compute approximate betweenness centrality using BFS from sampled nodes.

        For large graphs (>500 nodes), samples sqrt(N) source nodes for
        efficiency. For smaller graphs, computes exact betweenness.

        Returns:
            Array of shape [N] with normalized betweenness values.
        """
        n = self._num_nodes
        bc = np.zeros(n, dtype=np.float64)

        # Sample source nodes for large graphs
        if n > 500:
            num_samples = min(int(np.sqrt(n)) + 10, n)
            sources = np.random.choice(n, size=num_samples, replace=False)
        else:
            sources = np.arange(n)

        for s in sources:
            # BFS from source
            stack: list[int] = []
            pred: list[list[int]] = [[] for _ in range(n)]
            sigma = np.zeros(n, dtype=np.float64)
            dist = np.full(n, -1, dtype=np.int64)

            sigma[s] = 1.0
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in self._adj[v]:
                    # First visit
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # Shortest path via v
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # Back-propagation
            delta = np.zeros(n, dtype=np.float64)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    bc[w] += delta[w]

        # Normalize
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            if n > 500:
                # Adjust for sampling
                scale *= n / len(sources)
            bc *= scale

        return bc

    def _compute_edge_locality(
        self,
        mutation_node_idx: int,
        perturbed_edge_mask: np.ndarray,
    ) -> float:
        """Compute fraction of perturbed edges within 2 hops of mutation site.

        High locality (>0.8) suggests steric/local changes.
        Low locality (<0.5) suggests distributed/allosteric changes.
        """
        two_hop = self._get_n_hop_neighborhood(mutation_node_idx, 2)
        num_edges = self._edge_index.shape[1]

        local_perturbed = 0
        total_perturbed = 0

        for e in range(num_edges):
            if not perturbed_edge_mask[e]:
                continue
            total_perturbed += 1
            src = int(self._edge_index[0, e])
            tgt = int(self._edge_index[1, e])
            if src in two_hop or tgt in two_hop:
                local_perturbed += 1

        if total_perturbed == 0:
            return 1.0  # No perturbation = maximally local (trivial)

        return local_perturbed / total_perturbed

    def _get_n_hop_neighborhood(self, node: int, n_hops: int) -> set[int]:
        """Get all nodes within n hops of the given node."""
        visited = {node}
        frontier = {node}

        for _ in range(n_hops):
            next_frontier: set[int] = set()
            for v in frontier:
                for w in self._adj[v]:
                    if w not in visited:
                        visited.add(w)
                        next_frontier.add(w)
            frontier = next_frontier

        return visited

    def _bfs_path(self, start: int, end: int) -> list[int] | None:
        """BFS shortest path between two nodes."""
        if start == end:
            return [start]

        visited = [False] * self._num_nodes
        parent = [-1] * self._num_nodes
        visited[start] = True

        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self._adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    if neighbor == end:
                        path = []
                        node = end
                        while node != -1:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]
                    queue.append(neighbor)

        return None
