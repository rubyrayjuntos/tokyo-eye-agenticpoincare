"""
## CONTRACT

### Reads
- phase6c_result                   <- written by phase6c_admet_filter.py
  - passed                        : list of ADMET-clean hit dicts with delta_G, gtp_delta_G, mol, pharmacophore_center
- GTP receptor PDB                 <- user-supplied alternate-state structure
- GDP receptor PDB                 <- user-supplied protein structure (for alignment reference)

### Writes
- phase6d_result (dict)
  - selective                     : list of hit dicts with selectivity_ratio > threshold
  - non_selective                 : list of hit dicts with selectivity_ratio ≤ threshold
  - undetermined                  : list of hit dicts where ratio could not be computed
  - selectivity_threshold_used    : float     threshold applied

### GNN fields used directly
  - None (operates on Phase 6c ADMET-filtered hits + receptor PDBs)

### What this phase adds
  - GDP vs GTP conformational-state selectivity scoring
  - Selectivity ratio: abs(GDP_delta_G) / abs(GTP_delta_G)
  - Reuses Phase 6b GTP docking results when available (avoids re-docking)
  - Falls back to Vina docking against GTP receptor when gtp_delta_G is absent
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

try:
    from .phase6b_binding_affinity import run_vina_docking
except ImportError:
    from science.dtie.v3.phases.phase6b_binding_affinity import run_vina_docking

logger_p6d = logging.getLogger("DTIE_Phase6d")


def execute_phase_6d_state_selectivity_check(
    phase_input: "Phase6dInput",
) -> Dict:
    """
    Phase 6d: Conformational-State Selectivity Check.

    Docks each ADMET-passed hit against the GTP-bound receptor state using
    AutoDock Vina (same engine as Phase 6b).  The selectivity ratio is:

        ratio = abs(GDP_delta_G) / abs(GTP_delta_G)

    ratio > selectivity_threshold → GDP-state selective.
    (abs(GDP) > abs(GTP) means stronger binding to inactive GDP state.)

    A threshold of 1.3 means GDP binding must be ≥30% stronger than GTP binding.

    Handles edge cases:
      - GDP delta_G is None (Vina not available in Phase 6b): undetermined
      - GTP docking fails:                                    undetermined
      - GDP delta_G is near zero (≤ 1e-6 kcal/mol):          undetermined
    """
    admet_filtered_hits = phase_input.passed_admet.passed
    gtp_protein_pdb_path = phase_input.gtp_pdb
    selectivity_threshold = phase_input.selectivity_threshold
    gdp_protein_pdb_path = phase_input.gdp_protein_pdb_path
    selective = []
    non_selective = []
    undetermined = []

    for hit in admet_filtered_hits:
        smiles = hit.get("smiles", "")
        mol = hit.get("mol")
        center_xyz = hit.get("pharmacophore_center", [0.0, 0.0, 0.0])
        gdp_dg = hit.get("delta_G")  # may be None

        # ---- GTP-state docking -----------------------------------------------
        # If Phase 6b already docked against the GTP structure (gtp_delta_G
        # present in hit), reuse that value — no second Vina call needed.
        gtp_dg: Optional[float] = hit.get("gtp_delta_G")
        if gtp_dg is None:
            gtp_dg = run_vina_docking(
                mol,
                gtp_protein_pdb_path,
                center_xyz,
                reference_pdb=gdp_protein_pdb_path,
            )

        # ---- Ratio guard ------------------------------------------------------
        ratio: Optional[float] = None

        if gdp_dg is None or gtp_dg is None:
            logger_p6d.debug(
                f"Selectivity undetermined for {smiles[:40]}: "
                f"gdp_dG={gdp_dg}, gtp_dG={gtp_dg}"
            )
        elif abs(gdp_dg) <= 1e-6:
            logger_p6d.debug(f"GDP delta_G near-zero for {smiles[:40]}, skipping ratio")
        else:
            # Both affinities are negative (favorable binding); take magnitudes
            ratio = abs(gdp_dg) / max(abs(gtp_dg), 1e-8)

        enriched = {
            **hit,
            "gtp_delta_G": gtp_dg,
            "selectivity_ratio": ratio,
        }

        if ratio is None:
            undetermined.append(enriched)
        elif ratio > selectivity_threshold:
            selective.append(enriched)
        else:
            non_selective.append(enriched)

    selective.sort(key=lambda x: x["selectivity_ratio"])

    logger_p6d.info(
        f"Phase 6d: {len(selective)} selective | "
        f"{len(non_selective)} non-selective | "
        f"{len(undetermined)} undetermined "
        f"(threshold={selectivity_threshold})"
    )
    return {
        "selective": selective,
        "non_selective": non_selective,
        "undetermined": undetermined,
        "selectivity_threshold_used": selectivity_threshold,
    }
