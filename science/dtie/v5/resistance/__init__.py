"""Resistance Profiler — virtual mutation scanning and mechanism classification.

This module automates drug resistance mechanism classification using the
DTIE v5 hyperbolic GNN pipeline. It performs virtual mutations on protein
graphs, runs GNN inference on the perturbed graphs, and classifies the
resulting topological signatures into resistance mechanism types:
- Type I: Steric/Binding Interference (localized uncertainty collapse)
- Type II: Allosteric Uncoupling (distributed hub disruption)
- Hybrid: Mixed steric + allosteric signatures

Key components:
- MutationOperator: Applies physicochemically-grounded virtual mutations
- MechanismClassifier: Decision-tree classification from WT/mutant comparison
- ResistanceProfiler: Orchestrator with baseline caching and batch scanning
"""

from science.dtie.v5.resistance.classifier import MechanismClassifier
from science.dtie.v5.resistance.models import (
    AMINO_ACID_PROPERTIES,
    BatchReport,
    ClassifierConfig,
    HubMetrics,
    MutationSpec,
    ResistanceContribution,
    ResistanceReport,
    batch_report_to_dict,
    compute_perturbation_factor,
    compute_residue_contributions,
    mutant_structure_id,
    resistance_report_to_dict,
)
from science.dtie.v5.resistance.operator import MutationOperator
from science.dtie.v5.resistance.profiler import ResistanceProfiler

__all__ = [
    "AMINO_ACID_PROPERTIES",
    "BatchReport",
    "ClassifierConfig",
    "HubMetrics",
    "MechanismClassifier",
    "MutationOperator",
    "MutationSpec",
    "ResistanceContribution",
    "ResistanceProfiler",
    "ResistanceReport",
    "batch_report_to_dict",
    "compute_perturbation_factor",
    "compute_residue_contributions",
    "mutant_structure_id",
    "resistance_report_to_dict",
]
