"""GNN channel profiles for binding-site seed generation.

v5 and v6 checkpoints emit different scales for epistemic uncertainty and
shell geometry. The scan phase must not apply v5 absolute thresholds to v6
outputs (e.g. epistemic ~1.5–1.6 vs ~9.5+ on legacy checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.tools.cryptic.seed_generator import GNNNodeOutput


@dataclass(frozen=True)
class GNNChannelProfile:
    """Thresholds and normalization for seed-generation filters."""

    name: str
    shell_field: str  # "cone_depth" | "disc_r"
    epistemic_threshold: float | None = None
    shell_threshold: float | None = None
    epistemic_percentile: float | None = None
    shell_percentile: float | None = None
    epistemic_norm_max: float = 30.0
    shell_norm_max: float = 20.0


V5_LEGACY = GNNChannelProfile(
    name="v5_legacy",
    shell_field="cone_depth",
    epistemic_threshold=9.5,
    shell_threshold=6.0,
    epistemic_norm_max=30.0,
    shell_norm_max=20.0,
)

# lever_a / v6: percentile gates on structure; disc_r = ‖hyp_projections_2d‖
V6_LEVER_A = GNNChannelProfile(
    name="v6_lever_a",
    shell_field="disc_r",
    epistemic_percentile=75.0,
    shell_percentile=50.0,
    epistemic_norm_max=2.0,
    shell_norm_max=1.0,
)


def profile_for_model_version(model_version: str) -> GNNChannelProfile:
    """Pick seed-generation profile from provenance model_version string."""
    v = (model_version or "").lower()
    if any(tok in v for tok in ("v6", "lever_a", "moe", "hyperbolic_v6")):
        return V6_LEVER_A
    return V5_LEGACY


def _shell_value(node: GNNNodeOutput, profile: GNNChannelProfile) -> float:
    if profile.shell_field == "disc_r":
        if node.disc_r is None:
            return 0.0
        return float(node.disc_r)
    return float(node.cone_depth)


def resolve_profile_thresholds(
    nodes: list[GNNNodeOutput],
    profile: GNNChannelProfile,
) -> tuple[float, float]:
    """Resolve absolute epistemic and shell thresholds for a profile."""
    if not nodes:
        return (profile.epistemic_threshold or 0.0, profile.shell_threshold or 0.0)

    epi_vals = np.array([n.epistemic_uncertainty for n in nodes], dtype=float)
    shell_vals = np.array([_shell_value(n, profile) for n in nodes], dtype=float)

    if profile.epistemic_threshold is not None:
        epi_thr = profile.epistemic_threshold
    elif profile.epistemic_percentile is not None:
        epi_thr = float(np.percentile(epi_vals, profile.epistemic_percentile))
    else:
        epi_thr = 0.0

    if profile.shell_threshold is not None:
        shell_thr = profile.shell_threshold
    elif profile.shell_percentile is not None:
        shell_thr = float(np.percentile(shell_vals, profile.shell_percentile))
    else:
        shell_thr = 0.0

    return (epi_thr, shell_thr)


def filter_qualifying_residues_profile(
    nodes: list[GNNNodeOutput],
    profile: GNNChannelProfile,
) -> list[GNNNodeOutput]:
    """Filter residues using a channel profile (absolute or percentile thresholds)."""
    epi_thr, shell_thr = resolve_profile_thresholds(nodes, profile)
    return [
        node
        for node in nodes
        if node.epistemic_uncertainty >= epi_thr
        and _shell_value(node, profile) >= shell_thr
    ]


def composite_score_profile(
    node: GNNNodeOutput,
    profile: GNNChannelProfile,
) -> float:
    """Per-residue composite in [0, 1] for cluster scoring."""
    norm_epi = min(node.epistemic_uncertainty / profile.epistemic_norm_max, 1.0)
    norm_shell = min(_shell_value(node, profile) / profile.shell_norm_max, 1.0)
    return (norm_epi + norm_shell) / 2.0
