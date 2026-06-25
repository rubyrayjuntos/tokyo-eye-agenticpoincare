"""SDRP API router — REST endpoints for State-Dependent Resistance Profiles.

Exposes endpoints for:
- Profiling a single variant across multiple structures
- Batch profiling multiple variants
- Querying persisted ensemble profiles from the database
- Listing available structures for ensemble profiling
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sdrp", tags=["sdrp"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class BindingContextRequest(BaseModel):
    label: str = Field(..., description="Human-readable label, e.g. 'imatinib_bound'")
    drug: str | None = Field(None, description="Drug name if ligand-bound, None for apo")
    binding_mode: str = Field("apo", description="type_i, type_ii, allosteric, apo")
    pdb_id: str = Field(..., description="Source PDB identifier")


class StructureEntryRequest(BaseModel):
    structure_id: str = Field(..., description="Structure identifier")
    binding_context: BindingContextRequest


class ProfileVariantRequest(BaseModel):
    variant: str = Field(..., description="Mutation spec, e.g. 'T315I'")
    chain: str = Field("A", description="Chain identifier")
    position: int = Field(..., description="Residue position")
    wt_residue: str = Field(..., description="Wild-type residue one-letter code")
    mut_residue: str = Field(..., description="Mutant residue one-letter code")
    structures: list[StructureEntryRequest] = Field(
        ..., min_length=2, description="At least 2 structures for ensemble profiling"
    )
    hub_residues: list[list[Any]] | None = Field(
        None, description="Optional hub residues as [[chain, position], ...]"
    )


class BatchProfileRequest(BaseModel):
    variants: list[ProfileVariantRequest] = Field(
        ..., min_length=1, description="List of variants to profile"
    )
    structures: list[StructureEntryRequest] = Field(
        ..., min_length=2, description="Shared structure ensemble"
    )
    hub_residues: list[list[Any]] | None = Field(
        None, description="Optional hub residues"
    )


# ---------------------------------------------------------------------------
# POST /api/sdrp/profile — Profile a single variant
# ---------------------------------------------------------------------------


@router.post("/profile")
async def profile_variant(request: ProfileVariantRequest, db=Depends(get_db)):
    """Profile a single variant across multiple structures.

    Runs the SDRP Engine to compute an EnsembleProfile with:
    - Per-state classifications and stability scores
    - SSS (State-Sensitivity Score) via Jensen-Shannon Divergence
    - Mechanism shift detection
    - Categorization (Conformational_Switch / Static_Disruptor / Intermediate)
    - Clinical relevance summary
    """
    try:
        from science.dtie.v5.resistance.models import MutationSpec
        from science.dtie.v5.resistance.sdrp_engine import SDRPEngine
        from science.dtie.v5.resistance.sdrp_models import (
            BindingContext,
            StructureEntry,
            ensemble_profile_to_dict,
        )

        # Build structure entries
        structures = [
            StructureEntry(
                structure_id=s.structure_id,
                binding_context=BindingContext(
                    label=s.binding_context.label,
                    drug=s.binding_context.drug,
                    binding_mode=s.binding_context.binding_mode,
                    pdb_id=s.binding_context.pdb_id,
                ),
            )
            for s in request.structures
        ]

        # Build mutation spec
        mutation = MutationSpec(
            variant_name=request.variant,
            chain=request.chain,
            position=request.position,
            wt=request.wt_residue,
            mut=request.mut_residue,
        )

        # Build hub residues
        hub_residues = None
        if request.hub_residues:
            hub_residues = [(h[0], h[1]) for h in request.hub_residues]

        # Run the engine
        engine = SDRPEngine(db=db)
        profile = await engine.profile_variant(
            variant=mutation,
            structures=structures,
            hub_residues=hub_residues,
        )

        return ensemble_profile_to_dict(profile)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("SDRP profile_variant failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profiling failed: {e}")


# ---------------------------------------------------------------------------
# POST /api/sdrp/batch — Batch profile multiple variants
# ---------------------------------------------------------------------------


@router.post("/batch")
async def batch_profile(request: BatchProfileRequest, db=Depends(get_db)):
    """Profile multiple variants across the same structure ensemble.

    Returns a list of EnsembleProfiles in the same order as input variants.
    Baselines are computed once per structure and shared across all variants.
    """
    try:
        from science.dtie.v5.resistance.models import MutationSpec
        from science.dtie.v5.resistance.sdrp_engine import SDRPEngine
        from science.dtie.v5.resistance.sdrp_models import (
            BindingContext,
            StructureEntry,
            ensemble_profile_to_dict,
        )

        # Build shared structure entries
        structures = [
            StructureEntry(
                structure_id=s.structure_id,
                binding_context=BindingContext(
                    label=s.binding_context.label,
                    drug=s.binding_context.drug,
                    binding_mode=s.binding_context.binding_mode,
                    pdb_id=s.binding_context.pdb_id,
                ),
            )
            for s in request.structures
        ]

        # Build mutation specs
        mutations = [
            MutationSpec(
                variant_name=v.variant,
                chain=v.chain,
                position=v.position,
                wt=v.wt_residue,
                mut=v.mut_residue,
            )
            for v in request.variants
        ]

        # Build hub residues
        hub_residues = None
        if request.hub_residues:
            hub_residues = [(h[0], h[1]) for h in request.hub_residues]

        # Run the engine
        engine = SDRPEngine(db=db)
        profiles = await engine.profile_batch(
            variants=mutations,
            structures=structures,
            hub_residues=hub_residues,
        )

        return [ensemble_profile_to_dict(p) for p in profiles]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("SDRP batch_profile failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch profiling failed: {e}")


# ---------------------------------------------------------------------------
# GET /api/sdrp/profiles — Query persisted ensemble profiles
# ---------------------------------------------------------------------------


@router.get("/profiles")
async def list_profiles(
    variant: str | None = None,
    category: str | None = None,
    min_sss: float | None = None,
    max_sss: float | None = None,
    limit: int = 100,
    offset: int = 0,
    db=Depends(get_db),
):
    """Query persisted ensemble profiles from the database.

    Supports filtering by variant name, category, and SSS score range.
    Returns profiles ordered by computed_at descending.
    """
    try:
        conditions = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if variant:
            conditions.append("variant = :variant")
            params["variant"] = variant
        if category:
            conditions.append("category = :category")
            params["category"] = category
        if min_sss is not None:
            conditions.append("sss_score >= :min_sss")
            params["min_sss"] = min_sss
        if max_sss is not None:
            conditions.append("sss_score <= :max_sss")
            params["max_sss"] = max_sss

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT ensemble_id, variant, sss_score, category,
                   state_profile, mechanism_shifts, clinical_relevance,
                   structure_ids, run_ids, computed_at
            FROM fact_ensemble_resistance_profile
            {where_clause}
            ORDER BY computed_at DESC
            LIMIT :limit OFFSET :offset
        """

        rows = await db.fetch_all(query, params)

        results = []
        for row in rows:
            import json as json_mod
            results.append({
                "ensemble_id": row["ensemble_id"],
                "variant": row["variant"],
                "sss_score": float(row["sss_score"]),
                "category": row["category"],
                "state_profile": json_mod.loads(row["state_profile"]) if isinstance(row["state_profile"], str) else row["state_profile"],
                "mechanism_shifts": json_mod.loads(row["mechanism_shifts"]) if isinstance(row["mechanism_shifts"], str) else row["mechanism_shifts"],
                "clinical_relevance": row["clinical_relevance"],
                "structure_ids": row["structure_ids"],
                "run_ids": row["run_ids"],
                "computed_at": str(row["computed_at"]),
            })

        return {"profiles": results, "total": len(results), "limit": limit, "offset": offset}

    except Exception as e:
        logger.error("SDRP list_profiles failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ---------------------------------------------------------------------------
# GET /api/sdrp/profiles/{variant} — Get a specific variant's profile
# ---------------------------------------------------------------------------


@router.get("/profiles/{variant}")
async def get_profile(variant: str, db=Depends(get_db)):
    """Get the most recent ensemble profile for a specific variant."""
    try:
        query = """
            SELECT ensemble_id, variant, sss_score, category,
                   state_profile, mechanism_shifts, clinical_relevance,
                   structure_ids, run_ids, computed_at
            FROM fact_ensemble_resistance_profile
            WHERE variant = :variant
            ORDER BY computed_at DESC
            LIMIT 1
        """
        row = await db.fetch_one(query, {"variant": variant})

        if not row:
            raise HTTPException(status_code=404, detail=f"No profile found for variant {variant}")

        import json as json_mod
        return {
            "ensemble_id": row["ensemble_id"],
            "variant": row["variant"],
            "sss_score": float(row["sss_score"]),
            "category": row["category"],
            "state_profile": json_mod.loads(row["state_profile"]) if isinstance(row["state_profile"], str) else row["state_profile"],
            "mechanism_shifts": json_mod.loads(row["mechanism_shifts"]) if isinstance(row["mechanism_shifts"], str) else row["mechanism_shifts"],
            "clinical_relevance": row["clinical_relevance"],
            "structure_ids": row["structure_ids"],
            "run_ids": row["run_ids"],
            "computed_at": str(row["computed_at"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("SDRP get_profile failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
