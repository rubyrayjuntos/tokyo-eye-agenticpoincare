"""Categorization of mutations based on State-Sensitivity Score (SSS).

Classifies mutations as Conformational_Switch, Static_Disruptor, or Intermediate
based on SSS thresholds, and generates clinical relevance summaries.

Thresholds:
- SSS > 0.5 → Conformational_Switch
- SSS < 0.3 → Static_Disruptor
- 0.3 ≤ SSS ≤ 0.5 → Intermediate
"""

from __future__ import annotations

from .sdrp_models import MechanismShift, StateProfileEntry

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

SSS_SWITCH_THRESHOLD = 0.5
SSS_STATIC_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


def categorize(sss_score: float) -> str:
    """Categorize a mutation based on its SSS value.

    Args:
        sss_score: State-Sensitivity Score in [0.0, 1.0].

    Returns:
        "Conformational_Switch" if SSS > 0.5,
        "Static_Disruptor" if SSS < 0.3,
        "Intermediate" otherwise.
    """
    if sss_score > SSS_SWITCH_THRESHOLD:
        return "Conformational_Switch"
    elif sss_score < SSS_STATIC_THRESHOLD:
        return "Static_Disruptor"
    else:
        return "Intermediate"


# ---------------------------------------------------------------------------
# Clinical Relevance
# ---------------------------------------------------------------------------


def generate_clinical_relevance(
    category: str,
    variant: str,
    state_profile: dict[str, StateProfileEntry | None],
    mechanism_shifts: list[MechanismShift],
) -> str:
    """Generate a clinical relevance summary string.

    Produces a non-empty human-readable summary describing the drug-design
    implications of the categorization.

    Args:
        category: One of "Conformational_Switch", "Static_Disruptor", "Intermediate".
        variant: Mutation identifier (e.g., "T315I").
        state_profile: Mapping from binding context label to StateProfileEntry.
        mechanism_shifts: List of detected mechanism shifts.

    Returns:
        Non-empty summary string.
    """
    # Collect distinct mechanism classes from populated entries
    classes = set()
    for entry in state_profile.values():
        if entry is not None:
            classes.add(entry.mechanism_class)

    n_states = sum(1 for e in state_profile.values() if e is not None)
    n_shifts = len(mechanism_shifts)

    if category == "Conformational_Switch":
        shift_desc = ""
        if mechanism_shifts:
            first = mechanism_shifts[0]
            shift_desc = (
                f" Primary shift: {first.source_class} ({first.source_state}) → "
                f"{first.target_class} ({first.target_state})."
            )
        return (
            f"{variant} is a Conformational Switch exhibiting state-dependent "
            f"resistance across {n_states} conformational states with "
            f"{n_shifts} mechanism shift(s). Drug design must account for "
            f"multiple resistance mechanisms ({', '.join(sorted(classes))})."
            f"{shift_desc}"
        )

    elif category == "Static_Disruptor":
        dominant = sorted(classes)[0] if classes else "Unknown"
        return (
            f"{variant} is a Static Disruptor with consistent {dominant} "
            f"resistance across {n_states} conformational states. "
            f"A single targeted strategy addressing {dominant} mechanism "
            f"is expected to be effective across binding contexts."
        )

    else:  # Intermediate
        return (
            f"{variant} shows borderline conformational plasticity "
            f"(Intermediate category) across {n_states} states with "
            f"{n_shifts} mechanism shift(s). Further structural analysis "
            f"recommended to determine if state-aware drug design is warranted. "
            f"Observed mechanisms: {', '.join(sorted(classes)) if classes else 'none'}."
        )
