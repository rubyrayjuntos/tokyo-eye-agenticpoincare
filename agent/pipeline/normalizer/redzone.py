"""Normalize Red Zone screening results into the immunogenicity and metabolism tables.

Write path:
    RedZoneFlag (flag_type='immunogenicity')  →  fact_immunogenicity_run  +  fact_immunogenic_epitope
    RedZoneFlag (flag_type='metabolism')      →  fact_metabolism_run      +  fact_metabolism_site

There is no standalone fact_red_zone table. The v_red_zone_flags view in the
schema is a UNION over fact_immunogenic_epitope (where flagged=TRUE) and
fact_metabolism_site (where is_exposed=TRUE), joined back through their
respective run tables to dim_structure.

Each call writes exactly one run row and one detail row per RedZoneFlag.

atom_coords (optional, migration 007):
    Callers that have 3D coordinates for each vulnerable atom (e.g. from
    SiteOfMetabolism.coords) can pass a mapping of {atom_idx: (x, y, z)}.
    When present, redzone will resolve the dim_atom FK at write time by
    matching coordinates against dim_atom for the parent structure.
    Rows written without coordinates carry atom_id = NULL and are eligible
    for the coordinate-based backfill in migration 007.
"""
import uuid
from typing import Dict, List, Optional, Tuple

import sqlalchemy as sa

from gosp.models.data_models import RedZoneFlag


# ---------------------------------------------------------------------------
# SQL — immunogenicity
# ---------------------------------------------------------------------------

_INSERT_IMMUNO_RUN = sa.text("""
    INSERT INTO fact_immunogenicity_run
        (immuno_run_id, structure_id,
         alleles, ic50_threshold, sasa_threshold, peptide_length,
         total_peptides, flagged_count, warning,
         source_type)
    VALUES
        (:immuno_run_id, :structure_id,
         :alleles, :ic50_threshold, :sasa_threshold, :peptide_length,
         :total_peptides, :flagged_count, :warning,
         :source_type)
    ON CONFLICT (immuno_run_id) DO NOTHING
""")

_INSERT_IMMUNO_EPITOPE = sa.text("""
    INSERT INTO fact_immunogenic_epitope
        (epitope_id, immuno_run_id,
         sequence, start_res, end_res,
         allele, ic50, sasa_percent, flagged)
    VALUES
        (:epitope_id, :immuno_run_id,
         :sequence, :start_res, :end_res,
         :allele, :ic50, :sasa_percent, :flagged)
    ON CONFLICT (epitope_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# SQL — metabolism
# ---------------------------------------------------------------------------

_INSERT_METABOLISM_RUN = sa.text("""
    INSERT INTO fact_metabolism_run
        (metabolism_run_id, structure_id,
         isoform, sasa_threshold, flag_threshold,
         som_count, flagged, warning,
         source_type)
    VALUES
        (:metabolism_run_id, :structure_id,
         :isoform, :sasa_threshold, :flag_threshold,
         :som_count, :flagged, :warning,
         :source_type)
    ON CONFLICT (metabolism_run_id) DO NOTHING
""")

_INSERT_METABOLISM_SITE = sa.text("""
    INSERT INTO fact_metabolism_site
        (som_id, metabolism_run_id,
         atom_idx, atom_id, atom_type, pattern,
         coord_x, coord_y, coord_z,
         sasa, is_exposed)
    VALUES
        (:som_id, :metabolism_run_id,
         :atom_idx, :atom_id, :atom_type, :pattern,
         :coord_x, :coord_y, :coord_z,
         :sasa, :is_exposed)
    ON CONFLICT (som_id) DO NOTHING
""")

# Resolve dim_atom.atom_id by matching coordinates within a structure.
# Rounds to 3 dp to match PDB coordinate precision.
_RESOLVE_ATOM_ID = sa.text("""
    SELECT da.atom_id
    FROM dim_chain  dc
    JOIN dim_residue dr ON dr.chain_id   = dc.chain_id
    JOIN dim_atom    da ON da.residue_id = dr.residue_id
                       AND ROUND(da.x::numeric, 3) = ROUND(:x::numeric, 3)
                       AND ROUND(da.y::numeric, 3) = ROUND(:y::numeric, 3)
                       AND ROUND(da.z::numeric, 3) = ROUND(:z::numeric, 3)
    WHERE dc.structure_id = :structure_id
    LIMIT 1
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_redzone_flags(
    conn,
    structure_id: str,
    flags: List[RedZoneFlag],
    immuno_source_type: str = "external",
    metabolism_source_type: str = "deterministic",
    atom_coords: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> int:
    """Write Red Zone screening flags to their respective fact tables.

    Each ``RedZoneFlag`` produces one run row and one detail row:

    * ``flag_type='immunogenicity'``  →  ``fact_immunogenicity_run`` +
      ``fact_immunogenic_epitope``
    * ``flag_type='metabolism'``      →  ``fact_metabolism_run`` +
      ``fact_metabolism_site``

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    flags:
        List of :class:`~gosp.models.data_models.RedZoneFlag` objects
        produced by the immunogenicity or metabolic liability screening
        services.
    immuno_source_type:
        Provenance tag for ``fact_immunogenicity_run`` rows.  Defaults to
        ``"external"`` because immunogenicity prediction uses NetMHCIIpan,
        an external predictor.
    metabolism_source_type:
        Provenance tag for ``fact_metabolism_run`` rows.  Defaults to
        ``"deterministic"`` because metabolic liability screening is a
        rule-based, deterministic computation.
    atom_coords:
        Optional mapping of ``{biotite_atom_idx: (x, y, z)}`` for the
        structure being written.  When provided, each metabolism site row
        is enriched with the resolved ``dim_atom.atom_id`` FK by matching
        coordinates against the database.  Callers that do not have
        coordinates available (legacy path) omit this argument; affected
        rows carry ``atom_id = NULL`` and are eligible for the coordinate-
        based backfill in migration 007.

    Returns
    -------
    int
        Total number of *detail* rows inserted (one per flag).
    """
    count = 0
    for flag in flags:
        details = flag.details

        if flag.flag_type == "immunogenicity":
            run_id = str(uuid.uuid4())
            await conn.execute(_INSERT_IMMUNO_RUN, {
                "immuno_run_id":  run_id,
                "structure_id":   structure_id,
                "alleles":        ([details.allele] if details.allele else None),
                "ic50_threshold": details.ic50,
                "sasa_threshold": None,
                "peptide_length": None,
                "total_peptides": None,
                "flagged_count":  1,
                "warning":        None,
                "source_type":    immuno_source_type,
            })
            await conn.execute(_INSERT_IMMUNO_EPITOPE, {
                "epitope_id":    str(uuid.uuid4()),
                "immuno_run_id": run_id,
                "sequence":      details.sequence,
                "start_res":     details.start_res,
                "end_res":       details.end_res,
                "allele":        details.allele,
                "ic50":          details.ic50,
                "sasa_percent":  None,
                "flagged":       True,
            })

        elif flag.flag_type == "metabolism":
            run_id = str(uuid.uuid4())
            som_count = details.som_count or (
                len(details.vulnerable_atoms) if details.vulnerable_atoms else 0
            )
            await conn.execute(_INSERT_METABOLISM_RUN, {
                "metabolism_run_id": run_id,
                "structure_id":      structure_id,
                "isoform":           details.isoform or "unknown",
                "sasa_threshold":    None,
                "flag_threshold":    None,
                "som_count":         som_count,
                "flagged":           True,
                "warning":           None,
                "source_type":       metabolism_source_type,
            })
            # Write one metabolism_site row per vulnerable atom index.
            # Falls back to a single placeholder row when no atom list is given.
            atom_indices = details.vulnerable_atoms or [0]
            for atom_idx in atom_indices:
                coords = atom_coords.get(atom_idx) if atom_coords else None
                coord_x, coord_y, coord_z = coords if coords else (None, None, None)

                # Resolve dim_atom FK when coordinates are available.
                resolved_atom_id: Optional[str] = None
                if coords is not None:
                    row = await conn.execute(_RESOLVE_ATOM_ID, {
                        "structure_id": structure_id,
                        "x": coord_x,
                        "y": coord_y,
                        "z": coord_z,
                    })
                    result = row.fetchone()
                    if result:
                        resolved_atom_id = result[0]

                await conn.execute(_INSERT_METABOLISM_SITE, {
                    "som_id":              str(uuid.uuid4()),
                    "metabolism_run_id":   run_id,
                    "atom_idx":            atom_idx,
                    "atom_id":             resolved_atom_id,
                    "atom_type":           None,
                    "pattern":             None,
                    "coord_x":             coord_x,
                    "coord_y":             coord_y,
                    "coord_z":             coord_z,
                    "sasa":                None,
                    "is_exposed":          True,
                })

        count += 1
    return count
