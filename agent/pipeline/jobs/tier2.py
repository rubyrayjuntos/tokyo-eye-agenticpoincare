"""
Tier 2 parallel async job handlers.

Each job is enqueued independently via Cloud Tasks after Tier 1 completes.
Jobs run concurrently and do not block the GNN or viewport.

Handlers
--------
run_void_job              -- detect voids, write fact_void + bridge_void_dehydron
run_cdd_job               -- fetch CDD domain annotations, write fact_cdd_annotation
run_immunogenicity_job    -- run NetMHCIIpan immunogenicity screening
run_metabolism_job        -- run CYP450 metabolic liability screening
run_lerp_job              -- compute LERP folding path, upload to GCS, write fact_folding_path
run_energy_job            -- compute physics energy, write fact_energy_calculation
run_validation_mining_job -- validation mining using precomputed dehydrons/voids
"""
from __future__ import annotations

import os

import sqlalchemy as sa

from gosp.normalizer.events import NormalizerEvent, publish_event
from gosp.normalizer.jobs import write_job_status


# ---------------------------------------------------------------------------
# Private DB helpers
# ---------------------------------------------------------------------------

_FETCH_DEHYDRON_IDS = sa.text("""
    SELECT dehydron_id
    FROM fact_dehydron
    WHERE structure_id = :structure_id
    ORDER BY donor_residue_index, acceptor_residue_index
""")

_FETCH_DEHYDRONS_FULL = sa.text("""
    SELECT dehydron_id,
           donor_chain, donor_residue_index,
           acceptor_chain, acceptor_residue_index,
           midpoint_x, midpoint_y, midpoint_z,
           distance, wrapping_count,
           is_dehydron, is_interchain
    FROM fact_dehydron
    WHERE structure_id = :structure_id
    ORDER BY donor_residue_index, acceptor_residue_index
""")

_FETCH_VOIDS_FULL = sa.text("""
    SELECT void_id,
           center_x, center_y, center_z,
           volume, point_count
    FROM fact_void
    WHERE structure_id = :structure_id
""")


async def _fetch_dehydron_id_map(conn, structure_id: str) -> dict:
    """Return a 0-based index -> dehydron_id UUID mapping from fact_dehydron.

    The ordering mirrors the insertion order used by normalize_dehydrons
    (sorted by donor_residue_index, acceptor_residue_index), so index 0
    corresponds to the first dehydron written by Tier 1.
    """
    rows = await conn.execute(_FETCH_DEHYDRON_IDS, {"structure_id": structure_id})
    return {i: str(row.dehydron_id) for i, row in enumerate(rows.fetchall())}


async def _fetch_precomputed_dehydrons(conn, structure_id: str) -> list:
    """Reconstruct Dehydron objects from fact_dehydron for the given structure."""
    from gosp.models.data_models import Dehydron

    rows = await conn.execute(_FETCH_DEHYDRONS_FULL, {"structure_id": structure_id})
    dehydrons = []
    for row in rows.fetchall():
        dehydrons.append(Dehydron(
            donor_res_id=row.donor_residue_index,
            acceptor_res_id=row.acceptor_residue_index,
            wrapping_count=row.wrapping_count,
            midpoint=(row.midpoint_x, row.midpoint_y, row.midpoint_z),
            distance=row.distance,
            is_dehydron=row.is_dehydron,
            donor_chain_id=row.donor_chain,
            acceptor_chain_id=row.acceptor_chain,
            is_interchain=row.is_interchain,
        ))
    return dehydrons


async def _fetch_precomputed_voids(conn, structure_id: str) -> list:
    """Reconstruct Void objects from fact_void for the given structure."""
    from gosp.models.data_models import Void

    rows = await conn.execute(_FETCH_VOIDS_FULL, {"structure_id": structure_id})
    voids = []
    for i, row in enumerate(rows.fetchall()):
        voids.append(Void(
            void_id=i + 1,
            center=(row.center_x, row.center_y, row.center_z),
            volume=row.volume,
            point_count=row.point_count,
            nearby_dehydrons=[],
        ))
    return voids


async def _upload_lerp_frames(structure_id: str, frames_bytes: bytes) -> str:
    """Upload LERP frame bytes to GCS and return the GCS URI."""
    from google.cloud import storage  # type: ignore[import]

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    blob_name = f"folding-paths/{structure_id}/lerp_frames.npy"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(frames_bytes, content_type="application/octet-stream")

    return f"gs://{bucket_name}/{blob_name}"


# ---------------------------------------------------------------------------
# Public handlers
# ---------------------------------------------------------------------------

async def run_void_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Detect voids, write to fact_void + bridge_void_dehydron.

    Requires that Tier 1 has already written dehydrons to fact_dehydron for
    this structure_id so the bridge table FK references can be resolved.
    """
    from gosp.services.void_detection import detect_voids
    from gosp.normalizer.void import normalize_voids

    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="void", status="running",
    )
    try:
        dehydron_id_map = await _fetch_dehydron_id_map(conn, structure_id)
        dehydrons = await _fetch_precomputed_dehydrons(conn, structure_id)

        result = detect_voids(structure, dehydrons=dehydrons)
        count = await normalize_voids(
            conn=conn,
            structure_id=structure_id,
            voids=result.voids,
            dehydron_id_map=dehydron_id_map,
        )

        await write_job_status(
            conn, structure_id=structure_id, job_type="void",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="void", status="ready",
        ))
        return {"void_count": count}

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="void",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_cdd_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Fetch CDD domain annotations, write to fact_cdd_annotation.

    If the cdd_annotations service is unavailable (e.g. NCBI network not
    reachable in test environments), the job skips gracefully and returns
    ``{"skipped": True}``.
    """
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="cdd", status="running",
    )
    try:
        try:
            from gosp.services.cdd_annotations import fetch_cdd_annotations
        except ImportError:
            await write_job_status(
                conn, structure_id=structure_id, job_type="cdd",
                status="complete", job_status_id=jid,
                error_message="cdd_annotations service not available — skipped",
            )
            return {"skipped": True}

        from gosp.normalizer.cdd import normalize_cdd_annotations

        count = 0
        for chain in structure.chains:
            annotation = await fetch_cdd_annotations(structure.pdb_id, chain.chain_id)
            count += await normalize_cdd_annotations(
                conn=conn,
                structure_id=structure_id,
                annotations=annotation.domains,
                chain_label=chain.chain_id,
            )

        await write_job_status(
            conn, structure_id=structure_id, job_type="cdd",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="cdd_annotation", status="ready",
        ))
        return {"annotation_count": count}

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="cdd",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_immunogenicity_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Run NetMHCIIpan immunogenicity screening.

    Returns a summary dict and publishes a Pub/Sub event. No dedicated fact
    table exists yet for epitope rows.
    """
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="immunogenicity", status="running",
    )
    try:
        from gosp.services.immunogenicity_screening import screen_immunogenicity

        try:
            result = screen_immunogenicity(structure)
        except Exception as exc:
            await write_job_status(
                conn, structure_id=structure_id, job_type="immunogenicity",
                status="complete", job_status_id=jid, error_message=str(exc),
            )
            publish_event(NormalizerEvent(
                structure_id=structure_id, data_type="immunogenicity", status="ready",
                extra={
                    "flagged_count": 0,
                    "total_peptides": 0,
                    "warning": str(exc),
                },
            ))
            return {"skipped": True, "warning": str(exc)}

        await write_job_status(
            conn, structure_id=structure_id, job_type="immunogenicity",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="immunogenicity", status="ready",
            extra={
                "flagged_count": result.flagged_count,
                "total_peptides": result.total_peptides,
            },
        ))
        return {
            "flagged_count": result.flagged_count,
            "total_peptides": result.total_peptides,
        }

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="immunogenicity",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_metabolism_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Run CYP450 metabolic liability screening.

    Returns a summary dict with SOM count and flagged status.
    """
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="metabolism", status="running",
    )
    try:
        from gosp.services.metabolic_liability_screening import screen_metabolic_liability

        try:
            result = screen_metabolic_liability(structure)
        except Exception as exc:
            await write_job_status(
                conn, structure_id=structure_id, job_type="metabolism",
                status="complete", job_status_id=jid, error_message=str(exc),
            )
            publish_event(NormalizerEvent(
                structure_id=structure_id, data_type="metabolism", status="ready",
                extra={"som_count": 0, "flagged": False, "warning": str(exc)},
            ))
            return {"skipped": True, "warning": str(exc)}

        await write_job_status(
            conn, structure_id=structure_id, job_type="metabolism",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="metabolism", status="ready",
            extra={"som_count": result.som_count, "flagged": result.flagged},
        ))
        return {"som_count": result.som_count, "flagged": result.flagged}

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="metabolism",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_energy_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Run physics energy calculation and persist fact_energy_calculation."""
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="energy", status="running",
    )
    try:
        from gosp.services.physics_kernel import calculate_energy
        from gosp.normalizer.energy import normalize_energy

        result = calculate_energy(structure)
        count = await normalize_energy(
            conn=conn,
            structure_id=structure_id,
            energy_result=result,
            source_type="deterministic",
        )

        await write_job_status(
            conn, structure_id=structure_id, job_type="energy",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="energy", status="ready",
            extra={"row_count": count},
        ))
        return {"row_count": count}

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="energy",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_lerp_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Compute LERP folding path, upload to GCS, write fact_folding_path.

    The LERP service returns a numpy array of shape (num_frames, num_atoms, 3).
    Raw frame bytes are uploaded to GCS; only metadata is persisted to Cloud SQL
    via normalize_folding_path.
    """
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="lerp", status="running",
    )
    try:
        try:
            from gosp.services.physics_kernel import compute_lerp_path
        except ImportError:
            await write_job_status(
                conn, structure_id=structure_id, job_type="lerp",
                status="complete", job_status_id=jid,
                error_message="compute_lerp_path not available — skipped",
            )
            return {"skipped": True}

        import numpy as np
        from gosp.normalizer.folding import normalize_folding_path

        frames = compute_lerp_path(structure)  # ndarray (num_frames, num_atoms, 3)
        num_frames = frames.shape[0]
        num_atoms = frames.shape[1]

        frames_bytes = frames.astype(np.float32).tobytes()
        gcs_uri = await _upload_lerp_frames(structure_id, frames_bytes)

        path_id = await normalize_folding_path(
            conn=conn,
            structure_id=structure_id,
            gcs_uri=gcs_uri,
            num_frames=num_frames,
            num_atoms=num_atoms,
            method="lerp",
        )

        await write_job_status(
            conn, structure_id=structure_id, job_type="lerp",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="folding_path", status="ready",
            extra={"path_id": path_id, "gcs_uri": gcs_uri},
        ))
        return {"path_id": path_id, "num_frames": num_frames, "gcs_uri": gcs_uri}

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="lerp",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise


async def run_validation_mining_job(conn, structure_id: str, structure) -> dict:
    """Tier 2: Run validation mining using precomputed dehydrons and voids.

    Reads dehydrons and voids already written by prior Tier 2 jobs from the DB
    rather than recomputing them, then delegates to
    ValidationMiningPipeline.run_with_precomputed().
    """
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="validation_mining", status="running",
    )
    try:
        from gosp.services.validation_mining_pipeline import ValidationMiningPipeline
        from gosp.normalizer.validation import normalize_validation_run

        precomputed_dehydrons = await _fetch_precomputed_dehydrons(conn, structure_id)
        precomputed_voids = await _fetch_precomputed_voids(conn, structure_id)

        pdb_id = getattr(structure, "pdb_id", structure_id)
        pipeline = ValidationMiningPipeline()
        response = await pipeline.run_with_precomputed(
            pdb_id=pdb_id,
            structure=structure,
            precomputed_dehydrons=precomputed_dehydrons,
            precomputed_voids=precomputed_voids,
            precomputed_cdd_annotations=[],
        )

        run_id = await normalize_validation_run(
            conn=conn,
            structure_id=structure_id,
            response=response,
        )

        await write_job_status(
            conn, structure_id=structure_id, job_type="validation_mining",
            status="complete", job_status_id=jid,
        )
        publish_event(NormalizerEvent(
            structure_id=structure_id, data_type="validation_mining", status="ready",
            extra={"run_id": run_id},
        ))
        return {
            "run_id": run_id,
            "status": response.status,
            "dehydron_count": len(precomputed_dehydrons),
            "void_count": len(precomputed_voids),
        }

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="validation_mining",
            status="failed", job_status_id=jid, error_message=str(exc),
        )
        raise
