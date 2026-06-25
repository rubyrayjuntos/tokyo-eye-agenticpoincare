"""Stability Score computation for the SDRP module.

The stability score measures the minimum margin to the nearest decision
boundary that would flip the classification. This is analogous to
|log_odds(predicted) - log_odds(runner_up)| — it directly measures how
much noise you can add before the call changes.

For each mechanism class, the relevant thresholds are identified and the
distance from the observed metrics to those thresholds is computed. The
minimum distance (weakest link) determines stability.

A stability of 0.0 means the metrics are exactly at a decision boundary.
A stability of 1.0 means the metrics are far from any boundary (saturated).
"""

from __future__ import annotations

from science.dtie.v5.resistance.models import ClassifierConfig


def compute_stability_score(
    site_delta: float,
    max_hub_delta: float,
    propagation_radius: int,
    mechanism_class: str,
    config: ClassifierConfig | None = None,
) -> float:
    """Compute margin to nearest decision boundary that would flip the call.

    For each classification threshold relevant to the current mechanism_class,
    compute the normalized distance from the observed metric to that threshold.
    The minimum distance (weakest link) determines stability.

    Args:
        site_delta: Absolute site uncertainty delta |Δε_site|.
        max_hub_delta: Maximum absolute hub delta max|Δε_hub|.
        propagation_radius: Count of perturbed residues.
        mechanism_class: Current classification result.
        config: Classifier configuration with thresholds.

    Returns:
        Stability score clamped to [0.0, 1.0].
    """
    if config is None:
        config = ClassifierConfig()

    site_threshold = abs(config.site_perturbation_threshold)
    hub_threshold = abs(config.hub_allosteric_threshold)
    radius_threshold = config.propagation_radius_threshold

    if mechanism_class == "Type_I_Steric":
        # Type I requires: site above threshold AND hub below threshold
        # Margin = min(how far site is above site_threshold,
        #              how far hub is below hub_threshold)
        site_margin = (abs(site_delta) - site_threshold) / site_threshold
        hub_margin = (hub_threshold - max_hub_delta) / hub_threshold
        margin = min(site_margin, hub_margin)

    elif mechanism_class == "Type_II_Allosteric":
        # Type II requires: hub above threshold AND radius above threshold
        # Margin = min(how far hub is above hub_threshold,
        #              how far radius is above radius_threshold)
        hub_margin = (max_hub_delta - hub_threshold) / hub_threshold
        radius_margin = (propagation_radius - radius_threshold) / max(radius_threshold, 1)
        margin = min(hub_margin, radius_margin)

    elif mechanism_class == "Hybrid":
        # Hybrid: doesn't cleanly fit either category.
        # Stability = distance to nearest clear-class boundary.
        # Compute how close we are to becoming Type I or Type II.
        # The farther from both, the more stable the Hybrid call.
        dist_to_type_i = _distance_to_type_i(
            site_delta, max_hub_delta, site_threshold, hub_threshold
        )
        dist_to_type_ii = _distance_to_type_ii(
            max_hub_delta, propagation_radius, hub_threshold, radius_threshold
        )
        margin = min(dist_to_type_i, dist_to_type_ii)

    elif mechanism_class == "Neutral":
        # Neutral requires: both site and hub below noise floor.
        # Stability = how far both are below the noise floor.
        noise_floor = config.noise_floor
        site_margin = (noise_floor - abs(site_delta)) / noise_floor if noise_floor > 0 else 1.0
        hub_margin = (noise_floor - max_hub_delta) / noise_floor if noise_floor > 0 else 1.0
        margin = min(site_margin, hub_margin)

    else:
        # Unknown class — return 0 stability
        margin = 0.0

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, margin))


def _distance_to_type_i(
    site_delta: float,
    max_hub_delta: float,
    site_threshold: float,
    hub_threshold: float,
) -> float:
    """Compute normalized distance from current metrics to Type I boundary.

    Type I requires: site significant AND hub NOT significant.
    Distance = min of how far site would need to rise to cross threshold,
    or how far hub would need to drop below threshold.
    We measure how close the current state is to satisfying Type I conditions.
    """
    # How far is site from being significant (positive = already significant)
    site_gap = abs(abs(site_delta) - site_threshold) / site_threshold
    # How far is hub from being non-significant (positive = already non-significant)
    hub_gap = abs(hub_threshold - max_hub_delta) / hub_threshold
    return min(site_gap, hub_gap)


def _distance_to_type_ii(
    max_hub_delta: float,
    propagation_radius: int,
    hub_threshold: float,
    radius_threshold: int,
) -> float:
    """Compute normalized distance from current metrics to Type II boundary.

    Type II requires: hub significant AND radius significant.
    Distance = min of how far hub is from threshold, how far radius is from threshold.
    """
    hub_gap = abs(max_hub_delta - hub_threshold) / hub_threshold
    radius_gap = abs(propagation_radius - radius_threshold) / max(radius_threshold, 1)
    return min(hub_gap, radius_gap)
