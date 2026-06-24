"""Mechanism shift detection between conformational states.

Detects all pairwise mechanism shifts — cases where a mutation's resistance
classification changes between two conformational states. Each shift is
annotated with the delta in site_uncertainty and max_hub_delta between states.
"""

from __future__ import annotations

from itertools import combinations

from .sdrp_models import MechanismShift, StateProfileEntry


def detect_mechanism_shifts(
    state_profile: dict[str, StateProfileEntry | None],
) -> list[MechanismShift]:
    """Detect all pairwise mechanism shifts between states.

    A shift is recorded when two states have different mechanism_class values.
    Only unique pairs are recorded (A→B, not also B→A). Pairs are ordered
    lexicographically by state label to ensure deterministic output.

    None entries (sparse states) are skipped.

    Args:
        state_profile: Mapping from binding context label to StateProfileEntry.

    Returns:
        List of MechanismShift for all unique pairs with different mechanism_class.
    """
    # Filter to populated entries only
    populated: list[tuple[str, StateProfileEntry]] = [
        (label, entry)
        for label, entry in state_profile.items()
        if entry is not None
    ]

    shifts: list[MechanismShift] = []

    for (label_a, entry_a), (label_b, entry_b) in combinations(populated, 2):
        if entry_a.mechanism_class != entry_b.mechanism_class:
            # Order lexicographically by label for determinism
            if label_a <= label_b:
                source_label, source_entry = label_a, entry_a
                target_label, target_entry = label_b, entry_b
            else:
                source_label, source_entry = label_b, entry_b
                target_label, target_entry = label_a, entry_a

            shifts.append(
                MechanismShift(
                    source_state=source_label,
                    target_state=target_label,
                    source_class=source_entry.mechanism_class,
                    target_class=target_entry.mechanism_class,
                    delta_site_uncertainty=(
                        target_entry.site_uncertainty_delta
                        - source_entry.site_uncertainty_delta
                    ),
                    delta_max_hub=(
                        target_entry.max_hub_delta - source_entry.max_hub_delta
                    ),
                )
            )

    return shifts
