"""Cloud Tasks push endpoint — dispatches Tier 2 jobs."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskPayload(BaseModel):
    job_type: str           # void | cdd | immunogenicity | metabolism | lerp | energy | validation_mining
    structure_id: str
    job_status_id: Optional[str] = None


@router.post("/run")
async def run_task(payload: TaskPayload, background_tasks: BackgroundTasks):
    """
    Cloud Tasks push endpoint. Receives a Tier 2 job request and runs it.

    Cloud Tasks sends POST with JSON body. This endpoint dispatches to the
    correct Tier 2 handler based on job_type.

    Returns 200 immediately; Cloud Tasks retries on non-2xx.
    """
    from gosp.jobs import tier2

    handler_map = {
        "void": tier2.run_void_job,
        "cdd": tier2.run_cdd_job,
        "immunogenicity": tier2.run_immunogenicity_job,
        "metabolism": tier2.run_metabolism_job,
        "lerp": tier2.run_lerp_job,
        "energy": tier2.run_energy_job,
        "validation_mining": tier2.run_validation_mining_job,
    }

    handler = handler_map.get(payload.job_type)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unknown job_type: {payload.job_type}")

    # Run in background to return 200 quickly to Cloud Tasks.
    # The job itself updates fact_job_status on completion/failure.
    background_tasks.add_task(_run_job_with_conn, handler, payload.structure_id)
    return {
        "status": "dispatched",
        "job_type": payload.job_type,
        "structure_id": payload.structure_id,
    }


async def _run_job_with_conn(handler, structure_id: str):
    """Open a DB connection and run the handler."""
    from gosp.db.connection import create_engine_from_env
    from gosp.services.structure_loader import load_structure_from_db
    from sqlalchemy.ext.asyncio import AsyncSession

    engine = create_engine_from_env()
    async with AsyncSession(engine) as session:
        async with session.begin():
            structure = await load_structure_from_db(session, structure_id)
            await handler(conn=session, structure_id=structure_id, structure=structure)
