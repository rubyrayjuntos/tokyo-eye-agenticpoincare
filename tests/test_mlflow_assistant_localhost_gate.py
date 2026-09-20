"""Unit tests for the Docker-aware MLflow Assistant localhost gate."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_GATE = Path(__file__).resolve().parents[1] / "scripts" / "mlflow_assistant_localhost_gate.py"
_spec = importlib.util.spec_from_file_location("mlflow_assistant_localhost_gate", _GATE)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
require_localhost_or_docker_private = _mod.require_localhost_or_docker_private


def _request(host: str | None):
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(client=client)


def test_loopback_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT", raising=False)
    asyncio.run(require_localhost_or_docker_private(_request("127.0.0.1")))


def test_private_blocked_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT", raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(require_localhost_or_docker_private(_request("172.19.0.1")))
    assert exc.value.status_code == 403


def test_private_allowed_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT", "1")
    asyncio.run(require_localhost_or_docker_private(_request("172.19.0.1")))


def test_public_blocked_even_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT", "1")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(require_localhost_or_docker_private(_request("8.8.8.8")))
    assert exc.value.status_code == 403
