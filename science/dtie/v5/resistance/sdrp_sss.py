"""State-Sensitivity Score (SSS) computation via Jensen-Shannon Divergence.

The SSS quantifies how much a mutation's resistance mechanism varies across
conformational states. It is computed as the JSD of per-state classification
probability vectors, normalized to [0, 1] by dividing by log₂(|CLASS_SPACE|).

Key design decisions:
- CLASS_SPACE is fixed at 4 categories → normalization factor = log₂(4) = 2.0
- Probability vectors use confidence as weight on the classified category,
  with (1 - confidence) distributed uniformly across remaining categories
- None entries (from sparse KinaseStateSet) are skipped in SSS computation
"""

from __future__ import annotations

import numpy as np

from .sdrp_models import StateProfileEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_SPACE = ["Type_I_Steric", "Type_II_Allosteric", "Hybrid", "Neutral"]

# Fixed normalization denominator: log₂(|CLASS_SPACE|) = log₂(4) = 2.0
# Ensures SSS ∈ [0, 1] regardless of ensemble size.
SSS_NORMALIZATION_FACTOR = np.log2(len(CLASS_SPACE))  # 2.0


# ---------------------------------------------------------------------------
# Probability Vector Construction
# ---------------------------------------------------------------------------


def mechanism_to_probability_vector(
    mechanism_class: str,
    confidence_score: float,
) -> np.ndarray:
    """Convert a classification to a probability vector over CLASS_SPACE.

    The classified category gets weight = confidence_score.
    Remaining (1 - confidence) is distributed uniformly across other classes.

    Args:
        mechanism_class: One of CLASS_SPACE values.
        confidence_score: Classifier confidence in [0, 1].

    Returns:
        np.ndarray of shape (4,) summing to 1.0.

    Raises:
        ValueError: If mechanism_class is not in CLASS_SPACE.
    """
    if mechanism_class not in CLASS_SPACE:
        raise ValueError(
            f"Unknown mechanism_class '{mechanism_class}'. "
            f"Must be one of {CLASS_SPACE}"
        )

    n_classes = len(CLASS_SPACE)
    remainder = (1.0 - confidence_score) / (n_classes - 1)

    vec = np.full(n_classes, remainder, dtype=np.float64)
    idx = CLASS_SPACE.index(mechanism_class)
    vec[idx] = confidence_score

    return vec


# ---------------------------------------------------------------------------
# SSS Computation
# ---------------------------------------------------------------------------


def _shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy H(p) using log base 2.

    Handles zero probabilities gracefully (0 * log(0) = 0).
    """
    mask = p > 0
    return -float(np.sum(p[mask] * np.log2(p[mask])))


def _jsd(distributions: list[np.ndarray]) -> float:
    """Compute Jensen-Shannon Divergence of multiple distributions.

    JSD(P₁, ..., Pₙ) = H(M) - (1/N) * Σ H(Pᵢ)
    where M = (1/N) * Σ Pᵢ (the mixture distribution).

    Returns:
        JSD value in [0, log₂(|CLASS_SPACE|)].
    """
    n = len(distributions)
    if n == 0:
        return 0.0

    # Mixture distribution
    mixture = np.mean(distributions, axis=0)

    # H(M) - mean(H(Pᵢ))
    h_mixture = _shannon_entropy(mixture)
    h_components = sum(_shannon_entropy(p) for p in distributions) / n

    return h_mixture - h_components


def compute_sss(state_entries: list[StateProfileEntry | None]) -> float:
    """Compute State-Sensitivity Score via Jensen-Shannon Divergence.

    JSD is computed over the probability vectors of all non-None state entries.
    Normalized by dividing by log₂(|CLASS_SPACE|) = 2.0 to produce
    a fixed [0.0, 1.0] range independent of batch size.

    None entries (from sparse KinaseStateSet) are SKIPPED — SSS is
    computed only over populated states.

    Args:
        state_entries: List of StateProfileEntry (or None for sparse states).

    Returns:
        SSS in [0.0, 1.0]. 0 = all states agree, 1 = maximum divergence.
        Returns 0.0 if fewer than 2 non-None entries exist.
    """
    # Filter out None entries
    valid_entries = [e for e in state_entries if e is not None]

    if len(valid_entries) < 2:
        return 0.0

    # Build probability vectors
    distributions = [
        mechanism_to_probability_vector(e.mechanism_class, e.confidence_score)
        for e in valid_entries
    ]

    # Compute JSD and normalize
    jsd_value = _jsd(distributions)
    sss = jsd_value / SSS_NORMALIZATION_FACTOR

    # Clamp to [0, 1] for floating-point safety
    return float(np.clip(sss, 0.0, 1.0))
