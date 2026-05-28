from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/session", tags=["session"])

_store: dict = {"last": None}


class SessionResponse(BaseModel):
    structure_id: Optional[str]


class SessionUpdate(BaseModel):
    structure_id: str


@router.get("/last", response_model=SessionResponse)
async def get_last_session() -> SessionResponse:
    return SessionResponse(structure_id=_store["last"])


@router.put("/last", response_model=SessionResponse)
async def update_last_session(update: SessionUpdate) -> SessionResponse:
    _store["last"] = update.structure_id
    return SessionResponse(structure_id=_store["last"])
