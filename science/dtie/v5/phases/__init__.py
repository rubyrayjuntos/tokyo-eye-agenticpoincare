"""V5-native phase implementations.

These operate directly from GNN embeddings stored in the governed layer
(fact_gnn_node_embedding) and the graph topology, bypassing the v3
dataclass contracts entirely. Each phase queries the DB for what it needs.
"""
