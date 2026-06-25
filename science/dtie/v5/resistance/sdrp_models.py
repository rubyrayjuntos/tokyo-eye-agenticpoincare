"""Data models for the State-Dependent Resistance Profile (SDRP) module.

Defines:
- BindingContext: Metadata describing the conformational state of a PDB structure
- StructureEntry: A structure to profile against, with its binding context
- KinaseStateSet: Canonical conformational states for a kinase family
- StateProfileEntry: Per-state classification result within an SDRP
- MechanismShift: Records a mechanism change between two conformational states
- EnsembleProfile: Complete SDRP output for a single variant
- ensemble_profile_to_dict: JSON serialization with numpy→native conversion
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BindingContext:
    """Metadata describing the conformational state of a PDB structure."""

    label: str  # Human-readable label, e.g. "imatinib_bound"
    drug: str | None  # Drug name if ligand-bound, None for apo
    binding_mode: str  # "type_i", "type_ii", "allosteric", "apo"
    pdb_id: str  # Source PDB identifier


@dataclass
class StructureEntry:
    """A structure to profile against, with its binding context."""

    structure_id: str
    binding_context: BindingContext


@dataclass
class KinaseStateSet:
    """Canonical conformational states for a kinase family.

    Optional constraint that defines the expected states for structured
    downstream ML. Sparse states (not provided) get None entries in the
    profile, enabling consistent feature vectors across variants.
    """

    kinase_family: str  # e.g., "ABL1"
    canonical_states: list[str]  # e.g., ["apo", "imatinib_bound", "dasatinib_bound"]


@dataclass
class StateProfileEntry:
    """Per-state classification result within an SDRP.

    Preserves raw metrics for post-hoc debugging alongside the classification.
    """

    mechanism_class: str  # "Type_I_Steric", "Type_II_Allosteric", "Hybrid", "Neutral"
    stability_score: float  # 0.0–1.0, margin to nearest decision boundary
    confidence_score: float  # Original classifier confidence
    site_uncertainty_delta: float  # Raw Δε_site
    max_hub_delta: float  # Raw max |Δε_hub|
    propagation_radius: int  # Count of perturbed residues


@dataclass
class MechanismShift:
    """Records a mechanism change between two conformational states.

    This is an observational record — it captures WHAT changed, not WHY.
    The optional structural_basis field is an extension point for future
    causal analysis (backbone RMSD, DFG state, contact network changes).
    """

    source_state: str  # BindingContext label
    target_state: str  # BindingContext label
    source_class: str
    target_class: str
    delta_site_uncertainty: float  # target.site_delta - source.site_delta
    delta_max_hub: float  # target.max_hub - source.max_hub
    structural_basis: dict | None = None  # Future: RMSD, DFG state, contact Jaccard


@dataclass
class EnsembleProfile:
    """Complete SDRP output for a single variant."""

    variant: str
    state_profile: dict[str, StateProfileEntry | None]  # binding_context.label → entry
    conformational_sensitivity: float  # Alias for sss_score (0.0–1.0)
    sss_score: float  # Jensen-Shannon Divergence / log₂(4)
    category: str  # "Conformational_Switch", "Static_Disruptor", "Intermediate"
    clinical_relevance: str  # Summary string
    mechanism_shifts: list[MechanismShift]
    structure_entries: list[StructureEntry]  # Input structures for provenance
    per_structure_run_ids: list[str] = field(default_factory=list)  # Provenance linkage
    error_structures: dict[str, str] = field(default_factory=dict)  # structure_id → error msg


# ---------------------------------------------------------------------------
# JSON Serialization (numpy → native Python types)
# ---------------------------------------------------------------------------


def _convert_value(val: object) -> object:
    """Convert a single value from numpy to native Python type."""
    try:
        import numpy as np

        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.bool_):
            return bool(val)
    except ImportError:
        pass
    return val


def _binding_context_to_dict(ctx: BindingContext) -> dict:
    """Serialize a BindingContext to a JSON-safe dict."""
    return {
        "label": ctx.label,
        "drug": ctx.drug,
        "binding_mode": ctx.binding_mode,
        "pdb_id": ctx.pdb_id,
    }


def _structure_entry_to_dict(entry: StructureEntry) -> dict:
    """Serialize a StructureEntry to a JSON-safe dict."""
    return {
        "structure_id": entry.structure_id,
        "binding_context": _binding_context_to_dict(entry.binding_context),
    }


def _state_profile_entry_to_dict(entry: StateProfileEntry) -> dict:
    """Serialize a StateProfileEntry to a JSON-safe dict."""
    return {
        "mechanism_class": entry.mechanism_class,
        "stability_score": _convert_value(entry.stability_score),
        "confidence_score": _convert_value(entry.confidence_score),
        "site_uncertainty_delta": _convert_value(entry.site_uncertainty_delta),
        "max_hub_delta": _convert_value(entry.max_hub_delta),
        "propagation_radius": _convert_value(entry.propagation_radius),
    }


def _mechanism_shift_to_dict(shift: MechanismShift) -> dict:
    """Serialize a MechanismShift to a JSON-safe dict."""
    return {
        "source_state": shift.source_state,
        "target_state": shift.target_state,
        "source_class": shift.source_class,
        "target_class": shift.target_class,
        "delta_site_uncertainty": _convert_value(shift.delta_site_uncertainty),
        "delta_max_hub": _convert_value(shift.delta_max_hub),
        "structural_basis": shift.structural_basis,
    }


def ensemble_profile_to_dict(profile: EnsembleProfile) -> dict:
    """Serialize an EnsembleProfile to a JSON-safe dict.

    Converts numpy values to native Python types and includes
    all required fields. None entries in state_profile are preserved
    as null in the JSON output.
    """
    state_profile_dict: dict[str, dict | None] = {}
    for label, entry in profile.state_profile.items():
        if entry is None:
            state_profile_dict[label] = None
        else:
            state_profile_dict[label] = _state_profile_entry_to_dict(entry)

    return {
        "variant": profile.variant,
        "state_profile": state_profile_dict,
        "conformational_sensitivity": _convert_value(profile.conformational_sensitivity),
        "sss_score": _convert_value(profile.sss_score),
        "category": profile.category,
        "clinical_relevance": profile.clinical_relevance,
        "mechanism_shifts": [_mechanism_shift_to_dict(s) for s in profile.mechanism_shifts],
        "structure_entries": [_structure_entry_to_dict(e) for e in profile.structure_entries],
        "per_structure_run_ids": profile.per_structure_run_ids,
        "error_structures": profile.error_structures,
    }
