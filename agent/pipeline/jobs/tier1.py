"""
Tier 1 fast-path job handler.

Sequence (sequential, ~10-15s total):
1. Calculate per-residue SASA  →  UPDATE dim_residue.sasa
2. Detect dehydrons            →  normalize to fact_dehydron
3. Assemble GNN payload from DB
4. Serialize and upload to GCS
5. Publish gnn-ready event
6. Track job status throughout
"""
from typing import Optional

import sqlalchemy as sa

from gosp.normalizer.events import publish_event, NormalizerEvent
from gosp.normalizer.jobs import write_job_status
from gosp.normalizer.dehydron import normalize_dehydrons
from gosp.normalizer.gnn_payload import assemble_gnn_payload, serialize_and_upload
from gosp.services.physics_kernel import calculate_per_residue_sasa
from gosp.services.dehydron_detection import detect_dehydrons

async def _enqueue_validation_mining(structure_id: str) -> None:
    """Enqueue the validation_mining Cloud Task after gnn-ready fires."""
    import json
    import os

    project = os.environ.get("GCP_PROJECT_ID", "")
    tasks_url = os.environ.get("CLOUD_TASKS_HANDLER_URL", "")
    if tasks_url.endswith("/tasks/run"):
        tasks_url = tasks_url[: -len("/tasks/run")]

    if not project or not tasks_url:
        return  # local dev / CI — skip

    try:
        from google.cloud import tasks_v2  # type: ignore[import]

        client = tasks_v2.CloudTasksClient()
        location = os.environ.get("CLOUD_TASKS_LOCATION", "us-central1")
        queue = os.environ.get("CLOUD_TASKS_QUEUE", "tier2-jobs")
        parent = client.queue_path(project, location, queue)
        sa_email = os.environ.get(
            "CLOUD_TASKS_SA_EMAIL",
            "578264867059-compute@developer.gserviceaccount.com",
        )
        http_request = tasks_v2.HttpRequest(
            http_method=tasks_v2.HttpMethod.POST,
            url=f"{tasks_url}/tasks/run",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {"job_type": "validation_mining", "structure_id": structure_id}
            ).encode("utf-8"),
            oidc_token=tasks_v2.OidcToken(
                service_account_email=sa_email,
                audience=tasks_url,
            ),
        )
        client.create_task(
            request={"parent": parent, "task": tasks_v2.Task(http_request=http_request)}
        )
    except Exception as exc:
        import logging
        logging.getLogger("gosp.tier1").warning(
            "Failed to enqueue validation_mining for %s: %s", structure_id, exc
        )


_UPDATE_SASA = sa.text("""
    UPDATE dim_residue
    SET sasa = :sasa
    WHERE residue_id = :residue_id
""")


async def run_tier1(
    conn,
    structure_id: str,
    structure,
    job_status_id: Optional[str] = None,
) -> dict:
    """Execute Tier 1 fast-path chain. Returns summary dict."""

    # Mark job running
    jid = await write_job_status(
        conn, structure_id=structure_id, job_type="fast_path",
        status="running", job_status_id=job_status_id,
    )

    try:
        # Step 1: Per-residue SASA (synchronous)
        sasa_map = calculate_per_residue_sasa(structure)

        # Step 2: Resolve residue_index -> residue_id from DB, then UPDATE sasa
        rows = await conn.execute(
            sa.text("""
                SELECT r.residue_id, r.residue_index
                FROM dim_residue r
                JOIN dim_chain c ON r.chain_id = c.chain_id
                WHERE c.structure_id = :structure_id
            """),
            {"structure_id": structure_id}
        )
        for row in rows.fetchall():
            sasa_val = sasa_map.get(row.residue_index)
            if sasa_val is not None:
                await conn.execute(_UPDATE_SASA, {
                    "sasa": sasa_val,
                    "residue_id": row.residue_id,
                })

        # Step 3: Detect dehydrons (synchronous).
        # detect_dehydrons returns a DehydronDetectionResult; .dehydrons is the list.
        detection_result = detect_dehydrons(structure)
        dehydrons = detection_result.dehydrons

        # Step 4: Write dehydrons to fact_dehydron
        await normalize_dehydrons(
            conn=conn,
            structure_id=structure_id,
            dehydrons=dehydrons,
            source_type="deterministic",
        )

        # Step 5: Assemble GNN payload from DB (reads dim_residue + fact_dehydron)
        payload = await assemble_gnn_payload(conn=conn, structure_id=structure_id)

        # Step 6: Serialize and upload to GCS
        gcs_uri = await serialize_and_upload(payload)

        # Step 7: Publish gnn-ready event
        publish_event(NormalizerEvent(
            structure_id=structure_id,
            data_type="gnn",
            status="ready",
            extra={"gcs_uri": gcs_uri},
        ))

        # Step 8: Enqueue validation_mining now that dehydrons are in the DB.
        # This fires AFTER gnn-ready so the pipeline has everything it needs.
        await _enqueue_validation_mining(structure_id)

        # Mark complete
        await write_job_status(
            conn, structure_id=structure_id, job_type="fast_path",
            status="complete", job_status_id=jid,
        )

        return {
            "structure_id": structure_id,
            "dehydron_count": detection_result.dehydron_count,
            "node_count": len(payload.node_features),
            "gcs_uri": gcs_uri,
        }

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id, job_type="fast_path",
            status="failed", job_status_id=jid,
            error_message=str(exc),
        )
        raise
