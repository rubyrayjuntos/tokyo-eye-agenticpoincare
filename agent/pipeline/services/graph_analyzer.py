"""
State Space Graph Analyzer

Builds FSM representations of protein conformational landscapes from dehydron data.
"""

import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Set
from scipy.spatial.distance import cdist
import numpy as np

from gosp.models.state_graph import (
    StateGraph,
    StateNode,
    StateEdge,
    StateNodeMetadata,
    DehydronCluster,
    HealthStatus,
    BioMoleculeType,
    StateMachineLens,
    StateGraphSearchFacets,
    StateGraphResponse,
)
from gosp.models.data_models import Dehydron


class StateSpaceGraph:
    """
    Builds graph representation of dehydron-driven state transitions.
    
    Treats protein folding as navigation through a Finite State Machine where:
    - Nodes = states defined by dehydron topology
    - Edges = thermodynamically feasible transitions
    """
    
    def __init__(self, dehydron_service=None, energy_calculator=None):
        self.dehydron_service = dehydron_service
        self.energy_calculator = energy_calculator
    
    def build_folding_pathway(
        self,
        pdb_id: str,
        protein_name: str,
        dehydrons: List[Dehydron],
        lens: StateMachineLens = StateMachineLens.FOLDING_PATHWAY,
    ) -> StateGraph:
        """
        Build complete state graph for a folding pathway.
        
        Steps:
        1. Cluster dehydrons spatially
        2. Define discrete states based on dehydron signature
        3. Calculate energy barriers between states
        4. Validate reachability (no infinite loops)
        5. Identify trap states and active sites
        """
        
        graph = StateGraph(
            pdb_id=pdb_id,
            protein_name=protein_name,
            lens=lens,
            total_dehydrons=len(dehydrons),
        )
        
        # Generate folding intermediates (simplified model)
        # In production, these would come from MD simulations or AlphaFold ensembles
        intermediates = self._generate_folding_intermediates(dehydrons)
        
        # Create nodes for each state
        native_state = None
        for i, intermediate in enumerate(intermediates):
            node = self._create_state_node(
                intermediate=intermediate,
                stage_order=i,
                pdb_id=pdb_id,
                protein_name=protein_name,
                lens=lens,
            )
            graph.nodes.append(node)
            
            if node.metadata.is_native:
                native_state = node
        
        graph.native_state = native_state
        
        # Create edges between consecutive states
        for i in range(len(graph.nodes) - 1):
            source = graph.nodes[i]
            target = graph.nodes[i + 1]
            
            edge = self._create_transition_edge(source, target)
            graph.edges.append(edge)
        
        # Validate reachability
        self._validate_reachability(graph)
        
        # Count trap states and healthy paths
        graph.trap_states = len([n for n in graph.nodes if n.metadata.is_trap_state])
        graph.healthy_paths = len([e for e in graph.edges if e.is_reachable and e.pathway_type == "standard"])
        graph.dead_ends = len([e for e in graph.edges if not e.is_reachable])
        
        return graph
    
    def _generate_folding_intermediates(self, dehydrons: List[Dehydron]) -> List[Dict]:
        """
        Generate intermediate states along folding pathway.
        
        Simplified model: assume 5 linear stages from unfolded to native.
        In production, use MD or ensemble methods.
        
        Returns list of {dehydron_fraction, dehydrons_subset}
        """
        total = len(dehydrons)
        stages = [
            {'dehydron_fraction': 1.0, 'label': 'Unfolded'},        # 100% dehydrons
            {'dehydron_fraction': 0.8, 'label': 'SEARCH'},          # 80%
            {'dehydron_fraction': 0.5, 'label': 'ALIGN'},           # 50%
            {'dehydron_fraction': 0.3, 'label': 'LOCK-partial'},    # 30%
            {'dehydron_fraction': 0.15, 'label': 'Native'},         # 15% (final)
        ]
        
        intermediates = []
        for stage in stages:
            # Select subset of dehydrons for this stage
            frac = stage['dehydron_fraction']
            count = int(total * frac)
            subset = dehydrons[:count]  # Simplified: just take first N
            
            intermediates.append({
                'dehydron_fraction': frac,
                'dehydrons': subset,
                'stage_name': stage['label'],
                'is_native': frac < 0.2,  # Native state has <20% dehydrons
            })
        
        return intermediates
    
    def _create_state_node(
        self,
        intermediate: Dict,
        stage_order: int,
        pdb_id: str,
        protein_name: str,
        lens: StateMachineLens,
    ) -> StateNode:
        """Create a StateNode from an intermediate state."""
        
        dehydrons = intermediate['dehydrons']
        
        # Cluster dehydrons spatially
        clusters = self._cluster_dehydrons(dehydrons)
        
        # Compute deterministic signature
        signature = self._compute_dehydron_signature(dehydrons)
        
        # Determine health status
        health_status = self._classify_health_status(
            dehydron_fraction=intermediate['dehydron_fraction'],
            is_native=intermediate['is_native'],
            dehydron_count=len(dehydrons),
        )
        
        # Compute Gibbs free energy (relative to native)
        gibbs_free_energy = self._estimate_gibbs_free_energy(
            dehydron_fraction=intermediate['dehydron_fraction'],
        )
        
        metadata = StateNodeMetadata(
            lens_category=lens,
            biomolecule_type=BioMoleculeType.WILD_TYPE,
            pdb_id=pdb_id,
            protein_name=protein_name,
            dehydron_clusters=clusters,
            primary_cluster_id=clusters[0].cluster_id if clusters else "",
            total_dehydron_count=len(dehydrons),
            dehydron_fraction=intermediate['dehydron_fraction'],
            gibbs_free_energy=gibbs_free_energy,
            is_native=intermediate['is_native'],
            is_trap_state=self._is_trap_state(gibbs_free_energy, len(dehydrons)),
            health_status=health_status,
            node_signature=signature,
            stage_name=intermediate['stage_name'],
            stage_order=stage_order,
        )
        
        return StateNode(
            node_id=str(uuid.uuid4()),
            metadata=metadata,
        )
    
    def _cluster_dehydrons(self, dehydrons: List[Dehydron]) -> List[DehydronCluster]:
        """
        Group dehydrons spatially into clusters.
        
        Uses 8Å radius for clustering.
        """
        if not dehydrons:
            return []
        
        # Extract coordinates
        coords = np.array([[d.midpoint[0], d.midpoint[1], d.midpoint[2]] for d in dehydrons])
        
        # Simple clustering: group by residue range
        clusters_by_residue = {}
        for i, dehydron in enumerate(dehydrons):
            # Bin by residue (rounded to nearest 5)
            residue_bin = (dehydron.donor_res_id // 5) * 5
            if residue_bin not in clusters_by_residue:
                clusters_by_residue[residue_bin] = []
            clusters_by_residue[residue_bin].append((i, dehydron))
        
        # Create cluster objects
        clusters = []
        for residue_bin, items in sorted(clusters_by_residue.items()):
            indices = [i for i, _ in items]
            dehydron_objs = [d for _, d in items]
            
            start_res = min(d.donor_res_id for d in dehydron_objs)
            end_res = max(d.acceptor_res_id for d in dehydron_objs)
            center_res = (start_res + end_res) // 2
            
            cluster_id = f"region-{residue_bin}-{center_res}"
            
            cluster = DehydronCluster(
                cluster_id=cluster_id,
                location=f"{start_res}-{end_res}",
                center_residue=center_res,
                dehydron_indices=indices,
                dehydron_count=len(dehydron_objs),
                density=len(dehydron_objs) / len(dehydrons),
            )
            clusters.append(cluster)
        
        return sorted(clusters, key=lambda c: c.density, reverse=True)
    
    def _compute_dehydron_signature(self, dehydrons: List[Dehydron]) -> str:
        """
        Create deterministic hash from dehydron topology.
        
        Two states with identical dehydron signatures are considered the same state.
        """
        if not dehydrons:
            return "empty"
        
        # Sort by residue ID for determinism
        sorted_dehydrons = sorted(
            dehydrons,
            key=lambda d: (d.donor_res_id, d.acceptor_res_id),
        )
        
        # Create signature from residue pairs
        signature_parts = [
            f"{d.donor_res_id}-{d.acceptor_res_id}" for d in sorted_dehydrons
        ]
        signature_str = "|".join(signature_parts)
        
        # Hash it
        return hashlib.sha256(signature_str.encode()).hexdigest()[:12]
    
    def _classify_health_status(
        self,
        dehydron_fraction: float,
        is_native: bool,
        dehydron_count: int,
    ) -> HealthStatus:
        """Classify a state as healthy, trap, etc."""
        if is_native:
            return HealthStatus.NATIVE
        if dehydron_fraction > 0.7:
            return HealthStatus.HEALTHY  # Unfolded is healthy (no misfolding)
        if dehydron_fraction < 0.25:
            return HealthStatus.THERAPEUTIC_TARGET  # Partially folded, targetable
        return HealthStatus.HEALTHY
    
    def _estimate_gibbs_free_energy(self, dehydron_fraction: float) -> float:
        """
        Rough estimate of ΔG based on dehydron fraction.
        
        More dehydrons = higher energy (less stable).
        This is a simplification; in production, use actual MD or calc.
        """
        # Assume ~-1 kcal/mol per missing dehydron in native state
        native_fraction = 0.15
        excess = (dehydron_fraction - native_fraction) * 100
        delta_g = excess * 0.05  # Rough coefficient
        return delta_g
    
    def _is_trap_state(self, gibbs_free_energy: float, dehydron_count: int) -> bool:
        """
        Identify high-energy trap states.
        
        Traps have high ΔG but don't transition easily.
        """
        return gibbs_free_energy > 2.0 and dehydron_count > 10
    
    def _create_transition_edge(self, source: StateNode, target: StateNode) -> StateEdge:
        """
        Create edge representing feasible transition.
        """
        # Calculate energy barrier
        dg_source = source.metadata.gibbs_free_energy
        dg_target = target.metadata.gibbs_free_energy
        barrier = max(0, dg_source - dg_target + 1.5)  # +1.5 for crossing barrier
        
        # Estimate transition time from barrier height
        # Assume Kramers rate: t ~ 1/k, where k ~ exp(-Ea/RT)
        # At 300K: RT = 0.596 kcal/mol
        transition_time_ms = 0.1 * np.exp(barrier / 0.596)
        
        return StateEdge(
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            transition_barrier=barrier,
            transition_time_ms=transition_time_ms,
            is_reachable=barrier < 8.0,  # Assume >8 kcal/mol is unreachable
            pathway_type="standard",
        )
    
    def _validate_reachability(self, graph: StateGraph) -> None:
        """
        Validate that native state is reachable from all unfolded states.
        
        Proof: no infinite loops or dead ends assuming deterministic folding.
        """
        if not graph.native_state:
            return
        
        for node in graph.nodes:
            if node.metadata.is_native:
                continue
            
            # Check if native is reachable from this node
            is_reachable = graph.is_reachable_from(node.node_id, graph.native_state.node_id)
            
            # Mark edges accordingly
            for edge in graph.get_edges_from(node.node_id):
                if not is_reachable:
                    edge.is_reachable = False
                    edge.pathway_type = "dead-end"
    
    def build_graph_with_filters(
        self,
        graph: StateGraph,
        filter_health: Optional[str] = None,
        filter_cluster: Optional[str] = None,
        search_mutation: Optional[str] = None,
    ) -> StateGraph:
        """
        Apply filters to graph for progressive drill-down.
        """
        filtered_nodes = graph.nodes
        
        if filter_health:
            filtered_nodes = [
                n for n in filtered_nodes
                if n.metadata.health_status.value == filter_health
            ]
        
        if filter_cluster:
            filtered_nodes = [
                n for n in filtered_nodes
                if any(c.cluster_id == filter_cluster for c in n.metadata.dehydron_clusters)
            ]
        
        if search_mutation:
            filtered_nodes = [
                n for n in filtered_nodes
                if n.metadata.mutation == search_mutation
            ]
        
        # Filter edges to only include those between filtered nodes
        filtered_node_ids = {n.node_id for n in filtered_nodes}
        filtered_edges = [
            e for e in graph.edges
            if e.source_node_id in filtered_node_ids and e.target_node_id in filtered_node_ids
        ]
        
        # Create filtered graph
        filtered_graph = StateGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            native_state=graph.native_state if graph.native_state and graph.native_state.node_id in filtered_node_ids else None,
            pdb_id=graph.pdb_id,
            protein_name=graph.protein_name,
            lens=graph.lens,
            total_dehydrons=graph.total_dehydrons,
        )
        
        return filtered_graph
    
    def extract_search_facets(self, graph: StateGraph) -> StateGraphSearchFacets:
        """
        Extract faceted search options from full graph.
        """
        health_statuses = sorted(set(
            n.metadata.health_status.value for n in graph.nodes
        ))
        
        cluster_ids = sorted(set(
            c.cluster_id
            for n in graph.nodes
            for c in n.metadata.dehydron_clusters
        ))
        
        mutations = sorted(set(
            n.metadata.mutation
            for n in graph.nodes
            if n.metadata.mutation
        ))
        
        stage_names = sorted(set(
            n.metadata.stage_name
            for n in graph.nodes
            if n.metadata.stage_name
        ))
        
        disease_assoc = sorted(set(
            n.metadata.disease_association
            for n in graph.nodes
            if n.metadata.disease_association
        ))
        
        return StateGraphSearchFacets(
            health_statuses=health_statuses,
            cluster_ids=cluster_ids,
            available_mutations=mutations,
            stage_names=stage_names,
            disease_associations=disease_assoc,
        )
