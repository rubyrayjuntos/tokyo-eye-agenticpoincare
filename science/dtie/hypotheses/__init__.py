"""Hypothesis models and evaluators for the DTIE atlas.

Currently contains the IRK C-terminal lid entropy-trap hypothesis.
"""

from .irk_lid_trap import (  # noqa: F401
    # IRK lid-trap
    IRKLidHypothesis,
    IRKLidHypothesisInput,
    IRKLidHypothesisResult,
    IRKLidBaseline,
    LidMetrics,
    LidResidueSet,
    TrapMode,
    IRKStateLabel,
    load_lid_metrics,
    evaluate_lid_hypothesis,
    # Abl L'-set (parallel, defined from IRK-neighbor recurring residues)
    AblResidueSet,
    AblLidBaseline,
    AblLidHypothesisInput,
    AblLidHypothesisResult,
    AblLidHypothesis,
    evaluate_abl_lid_hypothesis,
)