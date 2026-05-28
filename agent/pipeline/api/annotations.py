"""Annotation API endpoints."""

from fastapi import APIRouter, HTTPException

from gosp.models.data_models import CddAnnotation
from gosp.services.cdd_annotations import fetch_cdd_annotations

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


@router.get("/cdd", response_model=CddAnnotation)
async def get_cdd_annotations(pdb_id: str, chain_id: str) -> CddAnnotation:
    """Return CDD annotations for a PDB chain."""
    try:
        return await fetch_cdd_annotations(pdb_id, chain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="CDD annotation fetch failed") from exc
