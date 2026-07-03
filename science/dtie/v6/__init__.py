"""V6 GNN module — GOSPConeMapperV6 (Topologically-Routed MoE Specialization)."""

# Keep package init lightweight: agent container mounts science/ but has no torch.
# Import submodules directly (e.g. science.dtie.v6.gnn.runner), not via loss re-exports.
