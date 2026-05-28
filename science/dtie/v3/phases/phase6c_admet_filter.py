"""
## CONTRACT

### Reads
- phase6b_result                   <- written by phase6b_binding_affinity.py
  - docking_results               : list of hit dicts with smiles, delta_G, gtp_delta_G

### Writes
- phase6c_result (dict)
  - passed                        : list of hit dicts that cleared all ADMET filters
  - failed                        : list of hit dicts with admet_fail_reasons
  - total_input                   : int       count of input hits
  - filters                       : list[str] filter names applied

### GNN fields used directly
  - None (operates on Phase 6b docking results)

### What this phase adds
  - Ames mutagenicity SMARTS structural alerts (13 patterns)
  - Aronov hERG liability model (basic N + logP + MW)
  - Multi-violation Lipinski filter (>1 Ro5 violation)
  - CYP3A4 poly-aromatic flag (≥3 aromatic rings, low TPSA, high logP)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger_p6c = logging.getLogger("DTIE_Phase6c")

# ---------------------------------------------------------------------------
# Pre-compiled SMARTS structural alerts
# ---------------------------------------------------------------------------

# Ames mutagenicity alerts (Derek/DEREK-derived subset)
_AMES_SMARTS_DEFS: Dict[str, str] = {
    "nitro_aromatic": "[a][N+](=O)[O-]",
    "nitroso": "[#6][N;X2]=[O;X1]",
    "n_nitrosamine": "[#7][N;X2]=[O;X1]",
    "aromatic_amine": "[a][NH2]",
    "azo": "[#6][N;X2]=[N;X2][#6]",
    "epoxide": "[C;r3][O;r3]",
    "alkyl_halide_alpha": "[F,Cl,Br,I][CX4][CX4]",
    "acylating_agent": "[CX3](=[OX1])[F,Cl,Br,I]",
    "michael_acceptor": "[CX3]=[CX3][C,S](=[OX1])",
    "hydrazine": "[NX3][NX3]",
    "diazo": "[#6][N;X2]=[N;X2+]=[N;X1-]",
    "beta_lactam_alert": "[C;R][C;R](=O)[N;R][C;R](=O)",
    "quinone": "[#6]1(=O)[#6]~[#6][#6](=O)[#6]~[#6]1",
}

_HERG_BASIC_N: str = "[N;+0;!$(NC=O);!$(NS(=O));!$(NC#N)]"
_CYP3A4_ALERT: str = "[a]1[a][a][a][a][a]1.[a]1[a][a][a][a][a]1.[a]1[a][a][a][a][a]1"

# Compile once at import time
_AMES_PATTERNS: Dict[str, Chem.Mol] = {
    name: Chem.MolFromSmarts(sma) for name, sma in _AMES_SMARTS_DEFS.items()
}
assert all(
    v is not None for v in _AMES_PATTERNS.values()
), "One or more Ames SMARTS failed to compile — check patterns"

_HERG_N_PAT = Chem.MolFromSmarts(_HERG_BASIC_N)
assert _HERG_N_PAT is not None

# ---------------------------------------------------------------------------
# Individual filter predicates
# ---------------------------------------------------------------------------


def _check_ames(mol: Chem.Mol) -> List[str]:
    """Return list of triggered Ames alert names (empty = clean)."""
    return [name for name, pat in _AMES_PATTERNS.items() if mol.HasSubstructMatch(pat)]


def _check_herg_aronov(mol: Chem.Mol) -> bool:
    """
    Aronov hERG risk model:
    flag if: basic_N_count >= 1  AND  logP >= 3.7  AND  MW >= 300
    """
    n_basic = len(mol.GetSubstructMatches(_HERG_N_PAT))
    if n_basic < 1:
        return False
    logp = Descriptors.MolLogP(mol)
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    return logp >= 3.7 and mw >= 300.0


def _check_lipinski_multi(mol: Chem.Mol, max_violations: int = 1) -> bool:
    """Return True (flag) if more than max_violations Lipinski rules are broken."""
    violations = 0
    if rdMolDescriptors.CalcExactMolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if rdMolDescriptors.CalcNumHBD(mol) > 5:
        violations += 1
    if rdMolDescriptors.CalcNumHBA(mol) > 10:
        violations += 1
    return violations > max_violations


def _check_cyp3a4(mol: Chem.Mol) -> bool:
    """Flag highly lipophilic poly-aromatic compounds (crude CYP3A4 concern)."""
    n_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    return n_rings >= 3 and tpsa < 25 and logp > 4.0


# ---------------------------------------------------------------------------
# Master ADMET evaluator
# ---------------------------------------------------------------------------


def _admet_evaluate(smiles: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ["invalid_smiles"]

    # Ames structural alerts
    ames_hits = _check_ames(mol)
    if ames_hits:
        reasons.extend(f"ames:{h}" for h in ames_hits)

    # hERG (Aronov)
    if _check_herg_aronov(mol):
        reasons.append("herg_aronov")

    # Lipinski multi-violation (>1 violation)
    if _check_lipinski_multi(mol):
        reasons.append("lipinski_multi_violation")

    # CYP3A4 poly-aromatic
    if _check_cyp3a4(mol):
        reasons.append("cyp3a4_polyaromatic")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Phase 6c entry point
# ---------------------------------------------------------------------------


def execute_phase_6c_admet_filter(
    phase_input: "Phase6cInput",
) -> Dict:
    """
    Phase 6c: ADMET Filtration.

    Applies:
      - Ames mutagenicity SMARTS alerts (pre-compiled)
      - Aronov hERG liability model
      - Multi-violation Lipinski (>1 violation)
      - CYP3A4 poly-aromatic flag
    """
    affinity_ranked_hits = phase_input.phase6b_result.docking_results
    filters = phase_input.filters
    if filters is None:
        filters = [
            "ames",
            "herg_aronov",
            "lipinski_multi_violation",
            "cyp3a4_polyaromatic",
        ]

    passed = []
    failed = []

    for hit in affinity_ranked_hits:
        smiles = hit.get("smiles", "")
        ok, reasons = _admet_evaluate(smiles)
        if ok:
            passed.append(hit)
        else:
            failed.append({**hit, "admet_fail_reasons": reasons})

    logger_p6c.info(
        f"Phase 6c: {len(passed)}/{len(affinity_ranked_hits)} hits passed ADMET "
        f"({len(failed)} flagged)"
    )
    return {
        "passed": passed,
        "failed": failed,
        "total_input": len(affinity_ranked_hits),
        "filters": filters,
    }
