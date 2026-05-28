import asyncio
import base64
import json
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{structure_id}")
async def websocket_endpoint(websocket: WebSocket, structure_id: str):
    await websocket.accept()
    await websocket.send_json({"event": "connected", "structure_id": structure_id})

    project = os.environ.get("GCP_PROJECT_ID", "")
    subscription = os.environ.get(
        "PUBSUB_SUBSCRIPTION",
        f"projects/{project}/subscriptions/gosp-events" if project else "",
    )

    if not project or not subscription:
        # Local dev: echo mode
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(data)
        except WebSocketDisconnect:
            return

    # Cloud mode: poll Pub/Sub
    try:
        from google.cloud import pubsub_v1  # type: ignore[import]
        subscriber = pubsub_v1.SubscriberClient()
    except ImportError:
        # Fallback to echo mode
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(data)
        except WebSocketDisconnect:
            return
        return

    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: subscriber.pull(
                        request={"subscription": subscription, "max_messages": 10},
                        timeout=2.0,
                    ),
                )
            except Exception:
                await asyncio.sleep(1)
                continue

            ack_ids = []
            for msg in response.received_messages:
                try:
                    raw = base64.b64decode(msg.message.data).decode("utf-8")
                    event_data = json.loads(raw)
                except Exception:
                    ack_ids.append(msg.ack_id)
                    continue

                if event_data.get("structure_id") == structure_id:
                    await websocket.send_json(event_data)
                ack_ids.append(msg.ack_id)

            if ack_ids:
                await loop.run_in_executor(
                    None,
                    lambda: subscriber.acknowledge(
                        request={"subscription": subscription, "ack_ids": ack_ids}
                    ),
                )

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        subscriber.close()
