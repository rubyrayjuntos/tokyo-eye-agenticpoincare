"""WebSocket Viewport Protocol — manages viewer connections and directives."""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import WebSocket, WebSocketDisconnect, status

from agent.coordinator.auth import session_store, verify_token

logger = logging.getLogger(__name__)


class ViewportConnection:
    """A single WebSocket connection with its registered viewports."""

    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.viewport_ids: list[str] = []
        self.connected_at = time.time()


class ViewportConnectionManager:
    """Manages all active WebSocket viewport connections."""

    def __init__(self):
        self._connections: dict[str, ViewportConnection] = {}
        self._by_session: dict[str, list[str]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> ViewportConnection:
        await websocket.accept()
        conn_id = str(uuid.uuid4())
        conn = ViewportConnection(websocket, session_id)
        self._connections[conn_id] = conn
        self._by_session.setdefault(session_id, []).append(conn_id)
        return conn

    async def disconnect(self, conn: ViewportConnection) -> None:
        for conn_id, c in list(self._connections.items()):
            if c is conn:
                del self._connections[conn_id]
                session_conns = self._by_session.get(conn.session_id, [])
                if conn_id in session_conns:
                    session_conns.remove(conn_id)
                break

    async def send_to_session(self, session_id: str, message: dict) -> None:
        """Send a message to all connections for a session."""
        conn_ids = self._by_session.get(session_id, [])
        for conn_id in conn_ids:
            conn = self._connections.get(conn_id)
            if conn:
                try:
                    await conn.websocket.send_json(message)
                except Exception:
                    pass

    async def broadcast_directive(self, directive: dict) -> None:
        """Broadcast a viewport directive to ALL connected frontends."""
        for conn in self._connections.values():
            try:
                await conn.websocket.send_json(directive)
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Return connection stats for telemetry."""
        return {
            "active_connections": len(self._connections),
            "sessions": len(self._by_session),
        }


# Singleton manager
viewport_manager = ViewportConnectionManager()


async def websocket_viewport(websocket: WebSocket):
    """WebSocket endpoint for the unified Viewport Protocol.

    Lifecycle:
    1. Client connects with ?token= for authentication
    2. Server validates JWT and accepts
    3. Client sends viewport_register for each viewer
    4. Bidirectional event/command flow
    5. On disconnect, session state preserved for reconnection
    """
    token = websocket.query_params.get("token", "")
    payload = verify_token(token) if token else None

    # Auth required by default; opt-out only for local dev
    if payload is None and os.getenv("REQUIRE_AUTH", "true").lower() != "false":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    session_id = (payload or {}).get("session_id", str(uuid.uuid4()))
    conn = await viewport_manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "viewport_register":
                viewport_id = data.get("viewport_id", str(uuid.uuid4()))
                viewer_type = data.get("viewer_type", "poincare_disc")
                capabilities = data.get("capabilities", [])

                conn.viewport_ids.append(viewport_id)

                session_store.update(session_id, f"viewport_{viewport_id}", {
                    "viewer_type": viewer_type,
                    "capabilities": capabilities,
                    "registered_at": time.time(),
                    "state": data.get("initial_state", {}),
                })

                await websocket.send_json({
                    "type": "register_ack",
                    "viewport_id": viewport_id,
                    "session_id": session_id,
                    "capabilities": capabilities,
                })

            elif msg_type == "viewport_event":
                viewport_id = data.get("viewport_id", "")
                event_type = data.get("event_type", "")
                payload_data = data.get("payload", {})

                vp_state = session_store.get(session_id).get(f"viewport_{viewport_id}", {})
                if event_type == "selection":
                    vp_state["selected_residues"] = payload_data.get("residue_ids", [])
                elif event_type == "hover":
                    vp_state["hovered_residue"] = payload_data.get("residue_id")
                elif event_type == "metric_change":
                    vp_state["current_metric"] = payload_data.get("metric")

                session_store.update(session_id, f"viewport_{viewport_id}", vp_state)

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (session preserved): %s", session_id)
    except Exception:
        logger.exception("WebSocket error for session %s", session_id)
    finally:
        await viewport_manager.disconnect(conn)
