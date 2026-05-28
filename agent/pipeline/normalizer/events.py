"""Publish Pub/Sub events after every successful normalizer write."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("gosp.events")


@dataclass
class NormalizerEvent:
    structure_id: str
    data_type: str           # dehydron | void | gnn | etc.
    status: str              # ready | failed
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def publish_event(event: NormalizerEvent) -> None:
    """Publish a normalizer event to the canonical Pub/Sub topic for data_type.

    Gracefully skips in local dev when GCP_PROJECT_ID is not set.
    """
    project_id = os.environ.get("GCP_PROJECT_ID", "")
    if not project_id:
        logger.debug(
            "Pub/Sub skipped (no GCP_PROJECT_ID): %s/%s/%s",
            event.structure_id, event.data_type, event.status,
        )
        return

    from google.cloud import pubsub_v1

    topic_name = f"{event.data_type.replace('_', '-')}-ready"
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)
    payload = json.dumps({
        "structure_id": event.structure_id,
        "data_type": event.data_type,
        "status": event.status,
        "error": event.error,
        **event.extra,
    }).encode("utf-8")
    future = publisher.publish(
        topic_path,
        data=payload,
        structure_id=event.structure_id,
        data_type=event.data_type,
        status=event.status,
    )
    future.result()  # Block until confirmed
