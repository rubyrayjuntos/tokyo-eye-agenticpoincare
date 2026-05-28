"""
Signed URL generation for GCS-backed binary assets.

Assets are large files stored in GCS and referenced by URI in AlloyDB.
The client receives the signed URL and fetches directly from GCS,
avoiding proxying large binaries through Cloud Run.

Supported asset types (extensible):
  folding  — fact_folding_path.gcs_uri  (.npy LERP frame array)

Future:
  persistence  — fact_persistence_diagram.gcs_uri  (TDA barcode arrays)
  gnn_payload  — GCS payload uploaded pre gnn-ready event
"""
from __future__ import annotations

import datetime
import os

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncConnection

from gosp.api_v2.structures import get_db_conn

router = APIRouter(prefix="/structures", tags=["assets"])

_TTL = 3600  # signed URL lifetime in seconds


class SignedUrlResponse(BaseModel):
    signed_url: str
    gcs_uri: str
    expires_in: int


def _sign(gcs_uri: str) -> str:
    """Return a v4 signed GET URL for a gs:// URI.

    Falls back to the raw GCS URI in local dev (GCP_PROJECT_ID unset).
    """
    if not os.environ.get("GCP_PROJECT_ID"):
        return gcs_uri  # local dev / CI — caller downloads via gsutil or skips

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    rest = gcs_uri[5:]
    bucket_name, _, blob_path = rest.partition("/")
    if not blob_path:
        raise ValueError(f"No blob path in GCS URI: {gcs_uri}")

    try:
        from google.cloud import storage  # type: ignore[import]

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=_TTL),
            method="GET",
            version="v4",
        )
    except Exception as exc:
        raise RuntimeError(f"GCS signing failed: {exc}") from exc


@router.get(
    "/{structure_id}/assets/folding/{path_id}/signed-url",
    response_model=SignedUrlResponse,
)
async def get_folding_signed_url(
    structure_id: str,
    path_id: str,
    conn: AsyncConnection = Depends(get_db_conn),
) -> SignedUrlResponse:
    """Return a time-limited signed URL for a LERP folding-path asset.

    The client fetches the .npy frame array directly from GCS using this URL,
    bypassing Cloud Run and avoiding large binary proxying.

    Ownership is verified: path_id must belong to structure_id.
    """
    row = (
        await conn.execute(
            sa.text("""
                SELECT gcs_uri
                FROM   fact_folding_path
                WHERE  path_id = :path_id
                  AND  structure_id = :sid
            """),
            {"path_id": path_id, "sid": structure_id},
        )
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Folding path '{path_id}' not found for structure '{structure_id}'",
        )

    try:
        signed_url = _sign(row.gcs_uri)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SignedUrlResponse(
        signed_url=signed_url,
        gcs_uri=row.gcs_uri,
        expires_in=_TTL,
    )
