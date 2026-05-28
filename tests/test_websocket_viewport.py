"""Tests for the WebSocket viewport protocol."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.coordinator.viewport import ViewportConnectionManager, ViewportConnection


class TestViewportConnectionManager:
    @pytest.fixture
    def manager(self):
        return ViewportConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager, mock_websocket):
        conn = await manager.connect(mock_websocket, "session_1")
        mock_websocket.accept.assert_called_once()
        assert conn.session_id == "session_1"
        assert conn.websocket is mock_websocket

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self, manager, mock_websocket):
        conn = await manager.connect(mock_websocket, "session_1")
        await manager.disconnect(conn)

        # send_to_session should not reach the disconnected websocket
        await manager.send_to_session("session_1", {"type": "test"})
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_to_session_delivers_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "session_1")
        await manager.send_to_session("session_1", {"type": "directive", "data": "test"})
        mock_websocket.send_json.assert_called_once_with({"type": "directive", "data": "test"})

    @pytest.mark.asyncio
    async def test_send_to_session_ignores_other_sessions(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "session_1")
        await manager.send_to_session("session_2", {"type": "test"})
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self, manager):
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "session_1")
        await manager.connect(ws2, "session_2")

        await manager.broadcast_directive({"action": "clear"})
        ws1.send_json.assert_called_once_with({"action": "clear"})
        ws2.send_json.assert_called_once_with({"action": "clear"})

    @pytest.mark.asyncio
    async def test_multiple_connections_per_session(self, manager):
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "session_1")
        await manager.connect(ws2, "session_1")

        await manager.send_to_session("session_1", {"type": "update"})
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_handles_broken_connection_gracefully(self, manager):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("Connection closed"))

        await manager.connect(ws, "session_1")
        # Should not raise
        await manager.send_to_session("session_1", {"type": "test"})


class TestViewportConnection:
    def test_connection_tracks_viewport_ids(self):
        ws = MagicMock()
        conn = ViewportConnection(ws, "session_1")
        assert conn.viewport_ids == []
        conn.viewport_ids.append("viewer_1")
        assert "viewer_1" in conn.viewport_ids

    def test_connection_records_timestamp(self):
        import time
        ws = MagicMock()
        before = time.time()
        conn = ViewportConnection(ws, "session_1")
        after = time.time()
        assert before <= conn.connected_at <= after
