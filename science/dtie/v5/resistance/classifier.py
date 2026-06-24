"""Mechanism Classifier — resistance mechanism classification from WT/mutant comparison.

This module implements the classification decision tree that determines whether
a mutation causes:
- Type I (Steric/Binding Interference): localized uncertainty collapse at the
  mutation site with no propagation to distal hubs
- Type II (Allosteric Uncoupling): distributed perturbation along the drug's
  mechanical transmission pathways
- Hybrid: mixed steric + allosteric signatures

The classifier CONSUMES Phase 4 pathway data from the baseline to identify
coupling targets. It does NOT re-run spectral analysis on the mutant — the
perturbation is measured at the GNN embedding level (Δeps, Δdepth), not at
the graph Laplacian level.

Classification uses a combination of:
1. Noise floor filtering (Neutral detection)
2. Site perturbation magnitude (steric signal)
3. Hub delta magnitude (allosteric signal)
4. Propagation radius (broad distribution)
5. Path interception (narrow pathway allosteric transmission)

Thresholds are configurable via ClassifierConfig with a version string for
reproducibility across runs.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from science.dtie.common.interfaces import GNNNodeOutput
from science.dtie.v5.resistance.models import (
    ClassifierConfig,
    HubMetrics,
    MutationSpec,
    ResistanceReport,
)


class MechanismClassifier:
    """Classifies resistance mechanism from WT vs mutant comparison.

    Decision tree:
    1. Compute site_delta (Δeps at mutation site)
    2. Compute hub_deltas (Δeps at each hub)
    3. Compute propagation_radius (count of residues with |Δeps| > threshold)
    4. Classify:
       - If max(|hub_delta|) < HUB_THRESHOLD and site_delta significant → Type I
       - If max(|hub_delta|) >= HUB_THRESHOLD and radius > 5 → Type II
       - Otherwise → Hybrid
    """

    def __init__(self, config: ClassifierConfig | None = None):
        self._config = config or ClassifierConfig()

    @property
    def config(self) -> ClassifierConfig:
        """Return the current classifier configuration."""
        return self._config

    def classify(
        self,
        wt_nodes: list[GNNNodeOutput],
        mut_nodes: list[GNNNodeOutput],
        mutation: MutationSpec,
        hub_residues: list[tuple[str, int]],
        edge_index: np.ndarray | None = None,
    ) -> ResistanceReport:
        """Compare WT and mutant outputs, classify resistance mechanism.

        Args:
            wt_nodes: Per-node outputs from wild-type GNN inference.
            mut_nodes: Per-node outputs from mutant GNN inference.
            mutation: The mutation specification.
            hub_residues: List of (chain, residue_index) hub residues to monitor.
            edge_index: Optional [2, E] array of graph edges for path analysis.
                If provided, enables "path interception" detection for narrow
                allosteric pathways that don't produce high propagation radius.

        Returns:
            ResistanceReport with classification, confidence, and metrics.
        """
        config = self._config

        # Build lookup maps for WT and mutant nodes
        wt_map: dict[tuple[str, int], GNNNodeOutput] = {
            (n.chain_label, n.residue_index): n for n in wt_nodes
        }
        mut_map: dict[tuple[str, int], GNNNodeOutput] = {
            (n.chain_label, n.residue_index): n for n in mut_nodes
        }

        # Compute site delta (Δeps at mutation site)
        site_key = (mutation.chain, mutation.residue_index)
        wt_site = wt_map.get(site_key)
        mut_site = mut_map.get(site_key)

        if wt_site is None or mut_site is None:
            return ResistanceReport(
                variant=mutation.variant_name,
                mechanism_class="Unknown",
                confidence_score=0.0,
                metrics={"error": "Mutation site not found in results"},
                affected_pathways=[],
                structural_impact="Unable to compute — site not found",
                hub_details=[],
                error="Mutation site not found in WT or mutant results",
            )

        site_delta = mut_site.epistemic_uncertainty - wt_site.epistemic_uncertainty

        # Compute hub deltas
        hub_details: list[HubMetrics] = []
        for chain, res_idx in hub_residues:
            hub_key = (chain, res_idx)
            wt_hub = wt_map.get(hub_key)
            mut_hub = mut_map.get(hub_key)
            if wt_hub is None or mut_hub is None:
                continue

            delta_eps = mut_hub.epistemic_uncertainty - wt_hub.epistemic_uncertainty
            delta_depth = mut_hub.cone_depth - wt_hub.cone_depth

            hub_details.append(
                HubMetrics(
                    residue_index=res_idx,
                    chain=chain,
                    wt_epistemic=wt_hub.epistemic_uncertainty,
                    mut_epistemic=mut_hub.epistemic_uncertainty,
                    delta_epistemic=delta_eps,
                    wt_cone_depth=wt_hub.cone_depth,
                    mut_cone_depth=mut_hub.cone_depth,
                    delta_cone_depth=delta_depth,
                )
            )

        # Compute propagation radius: count of residues with |Δeps| > threshold
        propagation_count = 0
        for key in wt_map:
            mut_node = mut_map.get(key)
            if mut_node is None:
                continue
            wt_node = wt_map[key]
            delta = abs(mut_node.epistemic_uncertainty - wt_node.epistemic_uncertainty)
            if delta > abs(config.hub_allosteric_threshold):
                propagation_count += 1

        # Classification decision tree
        max_hub_delta = 0.0
        if hub_details:
            max_hub_delta = max(abs(h.delta_epistemic) for h in hub_details)

        # Step 1: NOISE FILTER — if both metrics are below the noise floor,
        # the mutation has no significant effect on the GNN topology
        noise_floor = config.noise_floor
        if abs(site_delta) < noise_floor and max_hub_delta < noise_floor:
            mechanism_class = "Neutral"
            structural_impact = (
                f"No significant perturbation detected for {mutation.variant_name}. "
                f"Site Δeps={site_delta:.4f}, max hub Δeps={max_hub_delta:.4f} "
                f"(below noise floor {noise_floor})."
            )
        else:
            # Step 2: CHEMICAL BOOST — amplify site_delta based on the
            # physicochemical severity of the amino acid substitution.
            # This catches mutations where the GNN signal is weak but the
            # chemical change is large (e.g., F→L in a tight binding pocket).
            # GUARD: Only boost when hub signal does NOT dominate (steric context).
            # If hub > site, the mutation is allosteric — don't boost toward steric.
            if max_hub_delta <= abs(site_delta):
                boosted_site_delta = self._apply_chemical_boost(
                    site_delta, mutation
                )
            else:
                boosted_site_delta = site_delta

            # Step 3: Compute significance flags using boosted site delta
            # Site significance uses absolute value — steric mutations can cause
            # uncertainty to increase OR decrease depending on the clash geometry
            site_significant = abs(boosted_site_delta) > abs(config.site_perturbation_threshold)
            hub_significant = max_hub_delta >= abs(config.hub_allosteric_threshold)
            radius_significant = (
                propagation_count > config.propagation_radius_threshold
            )

            # Step 3: EXCLUSIONARY classification
            # Steric: strong local perturbation WITHOUT allosteric propagation
            # (steric mutations can cause secondary ripples that are NOT allosteric)
            if site_significant and not hub_significant:
                mechanism_class = "Type_I_Steric"
                structural_impact = (
                    f"Localized steric perturbation at {mutation.variant_name}. "
                    f"Site Δeps={site_delta:.4f}, no significant hub disruption."
                )
            # Allosteric: hub disruption WITH distributed propagation
            elif hub_significant and radius_significant:
                mechanism_class = "Type_II_Allosteric"
                structural_impact = (
                    f"Distributed allosteric perturbation from {mutation.variant_name}. "
                    f"Max hub Δeps={max_hub_delta:.4f}, "
                    f"propagation radius={propagation_count} residues."
                )
            # Site significant + hub significant but radius NOT significant:
            # Check for PATH INTERCEPTION — if perturbed residues lie on the
            # shortest path between mutation site and a hub, this is narrow
            # allosteric transmission (not just local ripples)
            elif site_significant and hub_significant and not radius_significant:
                path_intercepts = self._check_path_interception(
                    wt_nodes=wt_nodes,
                    mut_nodes=mut_nodes,
                    mutation=mutation,
                    hub_residues=hub_residues,
                    edge_index=edge_index,
                    threshold=abs(config.hub_allosteric_threshold),
                )
                if path_intercepts:
                    mechanism_class = "Type_II_Allosteric"
                    structural_impact = (
                        f"Narrow allosteric pathway from {mutation.variant_name}. "
                        f"Perturbation intercepts hub pathway. "
                        f"Max hub Δeps={max_hub_delta:.4f}, radius={propagation_count}."
                    )
                else:
                    mechanism_class = "Type_I_Steric"
                    structural_impact = (
                        f"Steric perturbation at {mutation.variant_name} with local ripples. "
                        f"Site Δeps={site_delta:.4f}, hub Δeps={max_hub_delta:.4f}, "
                        f"radius={propagation_count} (no pathway interception)."
                    )
            # Hybrid: partial signals that don't cleanly fit either category
            else:
                mechanism_class = "Hybrid"
                structural_impact = (
                    f"Mixed steric/allosteric signature for {mutation.variant_name}. "
                    f"Site Δeps={site_delta:.4f}, max hub Δeps={max_hub_delta:.4f}, "
                    f"radius={propagation_count}."
                )

        # Compute confidence score
        confidence = self.compute_confidence(
            site_delta=site_delta,
            max_hub_delta=max_hub_delta,
            propagation_radius=propagation_count,
        )

        metrics = {
            "site_uncertainty_delta": site_delta,
            "max_hub_delta": max_hub_delta,
            "propagation_radius": propagation_count,
            "site_wt_epistemic": wt_site.epistemic_uncertainty,
            "site_mut_epistemic": mut_site.epistemic_uncertainty,
        }

        return ResistanceReport(
            variant=mutation.variant_name,
            mechanism_class=mechanism_class,
            confidence_score=confidence,
            metrics=metrics,
            affected_pathways=[],
            structural_impact=structural_impact,
            hub_details=hub_details,
        )

    def compute_confidence(
        self,
        site_delta: float,
        max_hub_delta: float,
        propagation_radius: int,
    ) -> float:
        """Compute classification confidence based on metric magnitudes.

        Confidence is higher when the distinguishing metrics are far from
        the classification thresholds (clear separation).

        Formula:
            site_conf = min(|site_delta| / |site_threshold|, 2.0) / 2.0
            hub_conf = min(max_hub_delta / |hub_threshold|, 3.0) / 3.0
            radius_conf = min(radius / radius_threshold, 3.0) / 3.0
            confidence = 0.4 * site_conf + 0.35 * hub_conf + 0.25 * radius_conf

        Returns:
            Float clamped to [0.0, 1.0].
        """
        config = self._config

        # Site confidence: how far below the threshold
        site_ratio = abs(site_delta) / abs(config.site_perturbation_threshold)
        site_conf = min(site_ratio, 2.0) / 2.0  # Saturates at 2x threshold

        # Hub confidence: how far from the hub threshold
        hub_ratio = max_hub_delta / abs(config.hub_allosteric_threshold)
        hub_conf = min(hub_ratio, 3.0) / 3.0  # Saturates at 3x threshold

        # Radius confidence: how far above the radius threshold
        radius_ratio = propagation_radius / max(
            config.propagation_radius_threshold, 1
        )
        radius_conf = min(radius_ratio, 3.0) / 3.0

        # Combined confidence: weighted average
        confidence = 0.4 * site_conf + 0.35 * hub_conf + 0.25 * radius_conf

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _apply_chemical_boost(
        self,
        site_delta: float,
        mutation: MutationSpec,
    ) -> float:
        """Amplify site_delta based on physicochemical severity of the mutation.

        When the GNN produces a weak signal but the amino acid substitution
        involves a large physicochemical change (volume, hydrophobicity, charge),
        the effective site_delta is boosted proportionally.

        This addresses the "chemically impoverished feature space" problem:
        the GNN sees backbone topology but not side-chain chemistry. The boost
        injects chemical knowledge as a post-GNN signal amplifier.

        Formula:
            severity = sqrt(ΔV_norm² + Δπ_norm² + Δq²)
            boost_factor = 1.0 + severity * boost_coefficient
            boosted_delta = site_delta * boost_factor

        The boost only amplifies — it never creates signal from nothing.
        If site_delta is zero, the boosted value is still zero.

        Args:
            site_delta: Raw GNN site perturbation delta.
            mutation: The mutation specification (wt/mut amino acids).

        Returns:
            Boosted site_delta value.
        """
        from science.dtie.v5.resistance.models import AMINO_ACID_PROPERTIES

        wt_props = AMINO_ACID_PROPERTIES.get(mutation.wild_type_aa)
        mut_props = AMINO_ACID_PROPERTIES.get(mutation.mutant_aa)

        if wt_props is None or mut_props is None:
            return site_delta  # Unknown AA, no boost

        # Normalized deltas (using empirical max ranges)
        # Max volume delta: W(227.8) - G(60.1) = 167.7
        # Max hydropathy delta: I(4.5) - R(-4.5) = 9.0
        # Max charge delta: 2 (e.g., E→K: -1 to +1)
        MAX_VOL_DELTA = 167.7
        MAX_HYD_DELTA = 9.0
        MAX_CHARGE_DELTA = 2.0

        delta_vol = abs(mut_props["volume"] - wt_props["volume"]) / MAX_VOL_DELTA
        delta_hyd = abs(mut_props["hydropathy"] - wt_props["hydropathy"]) / MAX_HYD_DELTA
        delta_charge = abs(mut_props["charge"] - wt_props["charge"]) / MAX_CHARGE_DELTA

        # Combined severity (L2 norm of normalized deltas)
        severity = (delta_vol**2 + delta_hyd**2 + delta_charge**2) ** 0.5

        # Boost coefficient: how much to amplify per unit severity
        # Calibrated so that F→L (severity ~0.15) gets ~1.5x boost
        # and E→K (severity ~0.55) gets ~2.6x boost
        BOOST_COEFFICIENT = 3.0

        boost_factor = 1.0 + severity * BOOST_COEFFICIENT

        return site_delta * boost_factor

    def _check_path_interception(
        self,
        wt_nodes: list[GNNNodeOutput],
        mut_nodes: list[GNNNodeOutput],
        mutation: MutationSpec,
        hub_residues: list[tuple[str, int]],
        edge_index: np.ndarray | None,
        threshold: float,
    ) -> bool:
        """Check if perturbed residues lie on shortest paths to hubs.

        This detects "narrow pathway" allosteric transmission: the mutation
        perturbs residues that sit on the critical signaling path between
        the mutation site and a propagation hub, even if the overall
        propagation radius is low.

        Algorithm:
        1. Build adjacency list from edge_index
        2. Find the graph index of the mutation site
        3. For each hub with significant delta, find shortest path from
           mutation site to hub using BFS
        4. Check if any residue on that path (excluding endpoints) has
           |Δeps| > threshold

        If edge_index is not provided, falls back to checking if any hub
        has a delta that is significantly larger than the site delta
        (amplification heuristic).

        Args:
            wt_nodes: Wild-type node outputs.
            mut_nodes: Mutant node outputs.
            mutation: The mutation specification.
            hub_residues: Hub residues to check paths to.
            edge_index: [2, E] graph connectivity array.
            threshold: Delta threshold for "perturbed" residues.

        Returns:
            True if path interception is detected (allosteric signal).
        """
        # Build delta map
        wt_map = {(n.chain_label, n.residue_index): n for n in wt_nodes}
        mut_map = {(n.chain_label, n.residue_index): n for n in mut_nodes}

        # If no edge_index, use amplification heuristic:
        # hub delta > 1.5x site delta suggests pathway amplification
        if edge_index is None:
            site_key = (mutation.chain, mutation.residue_index)
            wt_site = wt_map.get(site_key)
            mut_site = mut_map.get(site_key)
            if wt_site is None or mut_site is None:
                return False
            site_abs_delta = abs(mut_site.epistemic_uncertainty - wt_site.epistemic_uncertainty)
            for chain, res_idx in hub_residues:
                wt_hub = wt_map.get((chain, res_idx))
                mut_hub = mut_map.get((chain, res_idx))
                if wt_hub is None or mut_hub is None:
                    continue
                hub_abs_delta = abs(mut_hub.epistemic_uncertainty - wt_hub.epistemic_uncertainty)
                if hub_abs_delta > threshold and hub_abs_delta > 1.5 * site_abs_delta:
                    return True
            return False

        # Build node index mapping: (chain, residue_index) → graph index
        node_keys = [(n.chain_label, n.residue_index) for n in wt_nodes]
        key_to_idx = {k: i for i, k in enumerate(node_keys)}

        # Build adjacency list
        num_nodes = len(wt_nodes)
        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        num_edges = edge_index.shape[1]
        for e in range(num_edges):
            src = int(edge_index[0, e])
            tgt = int(edge_index[1, e])
            if src < num_nodes and tgt < num_nodes:
                adj[src].append(tgt)

        # Find mutation site index
        site_key = (mutation.chain, mutation.residue_index)
        site_idx = key_to_idx.get(site_key)
        if site_idx is None:
            return False

        # Compute per-node deltas
        node_deltas = []
        for i, key in enumerate(node_keys):
            wt_n = wt_map.get(key)
            mut_n = mut_map.get(key)
            if wt_n and mut_n:
                node_deltas.append(abs(mut_n.epistemic_uncertainty - wt_n.epistemic_uncertainty))
            else:
                node_deltas.append(0.0)

        # For each significant hub, BFS from mutation site and check path
        for chain, res_idx in hub_residues:
            hub_key = (chain, res_idx)
            hub_idx = key_to_idx.get(hub_key)
            if hub_idx is None:
                continue

            # Only check hubs with significant delta
            if node_deltas[hub_idx] < threshold:
                continue

            # BFS to find shortest path from site to hub
            path = self._bfs_shortest_path(adj, site_idx, hub_idx, num_nodes)
            if path is None:
                continue

            # Check if any intermediate node on the path is perturbed
            # (exclude the endpoints — mutation site and hub themselves)
            for node_idx in path[1:-1]:
                if node_deltas[node_idx] > threshold:
                    return True

        return False

    @staticmethod
    def _bfs_shortest_path(
        adj: list[list[int]],
        start: int,
        end: int,
        num_nodes: int,
    ) -> list[int] | None:
        """BFS shortest path between two nodes.

        Returns the path as a list of node indices, or None if unreachable.
        """
        if start == end:
            return [start]

        visited = [False] * num_nodes
        parent = [-1] * num_nodes
        visited[start] = True

        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    if neighbor == end:
                        # Reconstruct path
                        path = []
                        node = end
                        while node != -1:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]
                    queue.append(neighbor)

        return None  # Unreachable
