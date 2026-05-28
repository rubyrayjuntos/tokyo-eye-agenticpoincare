"""
State Space Graph Models

Represents a protein's conformational landscape as a Finite State Machine (FSM),
where nodes are dehydron-defined states and edges are thermodynamically feasible transitions.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import hashlib


class HealthStatus(str, Enum):
    """Classification of a state's relationship to native conformation."""
    HEALTHY = "healthy"
    MISFOLDING = "misfolding"
    TRAP = "trap"
    THERAPEUTIC_TARGET = "therapeutic-target"
    NATIVE = "native"


class BioMoleculeType(str, Enum):
    """Type of biomolecule state."""
    WILD_TYPE = "wild-type"
    MUTANT = "mutant"
    COMPLEX = "complex"
    TRANSITION_TRAP = "transition-trap"


class StateMachineLens(str, Enum):
    """Different projections of the state space."""
    FOLDING_PATHWAY = "folding-pathway"
    DEHYDRON_TOPOLOGY = "dehydron-topology"
    ENERGY_LANDSCAPE = "energy-landscape"
    MUTATION_IMPACT = "mutation-impact"
    THERAPEUTIC_TARGETS = "therapeutic-targets"


@dataclass
class DehydronCluster:
    """
    Spatially-grouped dehydron defects that define a structural region.
    
    Acts as a searchable entity and primary signature component.
    """
    cluster_id: str  # e.g., 'alpha-helix-4', 'loop-beta-sheet-2'
    location: str    # Residue range: '60-76'
    center_residue: int  # Average residue for clustering
    dehydron_indices: List[int] = field(default_factory=list)  # Indices into full dehydron list
    dehydron_count: int = 0
    density: float = 0.0  # 0.0-1.0, relative to state's total dehydrons
    is_active_site: bool = False  # Predictive: does this cluster drive transitions?
    is_water_exposed: bool = False  # Is this cluster likely to flood with water?


@dataclass
class StateNodeMetadata:
    """
    Full searchable metadata for each state node.
    Supports faceted navigation (Level 2 and 3 of the taxonomy).
    """
    # Level 1: System State (Lens-aware)
    lens_category: StateMachineLens
    
    # Level 2: Structural Granularity
    biomolecule_type: BioMoleculeType
    pdb_id: str
    protein_name: str
    mutation: Optional[str] = None  # e.g., 'G12D'
    
    # Level 3: Dehydron Signature (Searchable)
    dehydron_clusters: List[DehydronCluster] = field(default_factory=list)
    primary_cluster_id: str = ""  # Which cluster dominates THIS state's identity
    total_dehydron_count: int = 0
    dehydron_fraction: float = 0.0  # 0.0-1.0
    
    # Thermodynamic properties (sortable/filterable)
    gibbs_free_energy: float = 0.0  # ΔG relative to native state (kcal/mol)
    is_native: bool = False
    is_trap_state: bool = False  # High ΔG, difficult to escape
    is_active_site: bool = False  # Can bind ligands/drugs
    
    # Clinical/therapeutic metadata
    health_status: HealthStatus = HealthStatus.HEALTHY
    disease_association: Optional[str] = None  # e.g., 'cancer', 'neurodegeneration'
    
    # Deterministic Node Signature (for de-duplication and search)
    node_signature: str = ""  # Hash of dehydron topology
    
    # Visualization metadata
    stage_name: Optional[str] = None  # Human-readable: 'SEARCH', 'ALIGN', 'LOCK'
    stage_order: Optional[int] = None  # Position in folding timeline
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['lens_category'] = self.lens_category.value
        data['biomolecule_type'] = self.biomolecule_type.value
        data['health_status'] = self.health_status.value
        data['dehydron_clusters'] = [asdict(c) for c in self.dehydron_clusters]
        return data


@dataclass
class StateNode:
    """
    A single point in the protein's conformational landscape.
    
    Identified by dehydron topology and energy state.
    """
    node_id: str  # UUID
    metadata: StateNodeMetadata
    position_3d: Optional[Tuple[float, float, float]] = None  # For force-directed layout
    
    def __hash__(self):
        """Deterministic identity based on dehydron signature."""
        return hash(self.metadata.node_signature)
    
    def __eq__(self, other):
        """Two nodes are equal if they have identical signatures."""
        if not isinstance(other, StateNode):
            return False
        return self.metadata.node_signature == other.metadata.node_signature


@dataclass
class StateEdge:
    """
    A transition between two states.
    
    Represents a thermodynamically feasible folding pathway.
    """
    source_node_id: str
    target_node_id: str
    transition_barrier: float  # ΔG‡ in kcal/mol
    transition_time_ms: float  # Expected transition kinetics (at 300K)
    is_reachable: bool = True  # Formal verification: can this path be taken?
    pathway_type: str = "standard"  # 'standard', 'shortcut', 'dead-end'
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StateGraph:
    """
    Complete FSM representation of a protein's conformational landscape.
    
    Nodes represent energy minima; edges represent folding pathways.
    """
    nodes: List[StateNode] = field(default_factory=list)
    edges: List[StateEdge] = field(default_factory=list)
    native_state: Optional[StateNode] = None
    
    # Metadata
    pdb_id: str = ""
    protein_name: str = ""
    lens: StateMachineLens = StateMachineLens.FOLDING_PATHWAY
    
    # Statistics
    total_dehydrons: int = 0
    healthy_paths: int = 0
    trap_states: int = 0
    dead_ends: int = 0
    
    def get_node(self, node_id: str) -> Optional[StateNode]:
        """Retrieve node by ID."""
        return next((n for n in self.nodes if n.node_id == node_id), None)
    
    def get_edges_from(self, source_node_id: str) -> List[StateEdge]:
        """Get all outgoing transitions from a node."""
        return [e for e in self.edges if e.source_node_id == source_node_id]
    
    def get_edges_to(self, target_node_id: str) -> List[StateEdge]:
        """Get all incoming transitions to a node."""
        return [e for e in self.edges if e.target_node_id == target_node_id]
    
    def is_reachable_from(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if target is reachable from source via feasible transitions."""
        visited = set()
        queue = [source_node_id]
        
        while queue:
            current = queue.pop(0)
            if current == target_node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.get_edges_from(current):
                if edge.is_reachable:
                    queue.append(edge.target_node_id)
        
        return False
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'nodes': [
                {
                    'node_id': n.node_id,
                    'metadata': n.metadata.to_dict(),
                    'position_3d': n.position_3d,
                }
                for n in self.nodes
            ],
            'edges': [e.to_dict() for e in self.edges],
            'native_state': self.native_state.node_id if self.native_state else None,
            'pdb_id': self.pdb_id,
            'protein_name': self.protein_name,
            'lens': self.lens.value,
            'statistics': {
                'total_dehydrons': self.total_dehydrons,
                'healthy_paths': self.healthy_paths,
                'trap_states': self.trap_states,
                'dead_ends': self.dead_ends,
            }
        }


@dataclass
class StateGraphSearchFacets:
    """
    Faceted search results for progressive drill-down.
    """
    health_statuses: List[str]
    cluster_ids: List[str]
    available_mutations: List[str]
    stage_names: List[str]
    disease_associations: List[str]


@dataclass
class StateGraphResponse:
    """
    API response combining graph data and search facets.
    """
    graph: StateGraph
    search_facets: StateGraphSearchFacets
    
    def to_dict(self):
        return {
            'graph': self.graph.to_dict(),
            'search_facets': asdict(self.search_facets),
        }
