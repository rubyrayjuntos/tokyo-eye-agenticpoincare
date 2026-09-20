"""Allow MLflow Assistant API through Docker published localhost ports.

Upstream `_require_localhost` inspects `request.client.host`. With
`127.0.0.1:5000:5000` publish, the browser is on the host loopback, but the
container sees the Docker bridge/gateway IP → false 403 and the UI shows
"You do not have permission to access this resource."

When MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT is truthy, also accept private
client IPs (still gated by host-only port bind in compose).
"""

from __future__ import annotations

import ipaddress
import os

from fastapi import HTTPException, Request

_BLOCK_REMOTE_ACCESS_ERROR_MSG = (
    "Assistant API is only accessible from the same host where the MLflow server is running."
)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


async def require_localhost_or_docker_private(request: Request) -> None:
    client_host = request.client.host if request.client else None
    if not client_host:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG) from exc
    if ip.is_loopback:
        return
    if _env_truthy("MLFLOW_ASSISTANT_ALLOW_PRIVATE_CLIENT") and ip.is_private:
        return
    raise HTTPException(status_code=403, detail=_BLOCK_REMOTE_ACCESS_ERROR_MSG)
