"""Pub/Sub push endpoint — receives normalizer events."""
import base64
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/pubsub", tags=["pubsub"])


class PubSubMessage(BaseModel):
    data: str           # base64-encoded JSON
    messageId: str
    attributes: Optional[dict] = None


class PubSubPushPayload(BaseModel):
    message: PubSubMessage
    subscription: str


@router.post("/push")
async def receive_pubsub_push(payload: PubSubPushPayload):
    """
    Pub/Sub push endpoint.

    Pub/Sub sends base64-encoded JSON in message.data.
    Decode and log the event. Downstream consumers (viewport, GNN)
    use WebSocket for real-time delivery (see session.py).

    Returns 200 to acknowledge. Non-2xx would trigger redelivery.
    """
    try:
        raw = base64.b64decode(payload.message.data).decode("utf-8")
        event_data = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid message data: {exc}")

    structure_id = event_data.get("structure_id")
    data_type = event_data.get("data_type")
    status = event_data.get("status")

    # TODO: broadcast to WebSocket sessions in session.py
    # For now, just acknowledge
    return {
        "ack": True,
        "structure_id": structure_id,
        "data_type": data_type,
        "status": status,
    }
