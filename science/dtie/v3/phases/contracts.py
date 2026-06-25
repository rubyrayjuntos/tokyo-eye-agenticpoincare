# Re-export from common.contracts for backward compatibility.
# The v3 phase files were migrated from a flat directory where contracts.py
# was a sibling. This shim makes both `from .contracts import ...` and
# `from contracts import ...` (fallback) work when the phases are imported
# as a package.
from science.dtie.common.contracts import *  # noqa: F401,F403
from science.dtie.common.contracts import (
    Doorway,
    LiftedSite,
    Phase1Input,
    Phase1Output,
    Phase2Input,
    Phase2Output,
    Phase3Input,
    Phase3Output,
    Phase35Input,
    Phase35Output,
    Phase4Input,
    Phase4Output,
    Phase5Input,
    Phase5Output,
    Phase6aInput,
    Phase6aOutput,
    Phase6bInput,
    Phase6bOutput,
    Phase6cInput,
    Phase6cOutput,
    Phase6dInput,
    Phase6dOutput,
    Pharmacophore,
    ScreeningHit,
    SpectralAnalysis,
)
