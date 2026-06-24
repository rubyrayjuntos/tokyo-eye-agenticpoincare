"""Typed async HTTP client for the Science Container API.

Replaces docker-shell dispatch with direct HTTP calls over the internal
Docker network. The science container runs FastAPI on port 8001.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SCIENCE_BASE_URL = os.getenv("SCIENCE_API_URL", "http://science:8001")


class ScienceTimeoutError(Exception):
    """Raised when a Science API call exceeds its configured timeout."""

    def __init__(self, endpoint: str, timeout: float):
        self.endpoint = endpoint
        self.timeout = timeout
        super().__init__(f"Science API timeout: {endpoint} after {timeout}s")


class ScienceComputeError(Exception):
    """Raised when the Science API returns a 4xx/5xx error response."""

    def __init__(self, endpoint: str, status: int, detail: str):
        self.endpoint = endpoint
        self.status = status
        self.detail = detail
        super().__init__(f"Science API error: {endpoint} → {status}: {detail}")


class ScienceClient:
    """Async HTTP client for the Science Container API.

    Provides typed methods for each compute endpoint with configurable
    timeouts. Raises ScienceTimeoutError or ScienceComputeError on failure.
    """

    def __init__(
        self,
        base_url: str = SCIENCE_BASE_URL,
        pipeline_timeout: float = 600.0,
        health_timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._pipeline_timeout = pipeline_timeout
        self._health_timeout = health_timeout

    async def health(self) -> dict[str, Any]:
        """Check science container health status."""
        return await self._get("/health", timeout=self._health_timeout)

    async def run_gnn(self, structure_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run GNN inference on a structure."""
        return await self._post(
            "/compute/gnn",
            {"structure_id": structure_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def run_pipeline(self, structure_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run the full DTIE pipeline on a structure."""
        return await self._post(
            "/compute/pipeline",
            {"structure_id": structure_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def run_cryptic_scan(self, structure_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run cryptic binding site scan on a structure."""
        return await self._post(
            "/compute/cryptic-scan",
            {"structure_id": structure_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def run_motif_analysis(self, structure_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run hyperbolic motif analysis on a structure."""
        return await self._post(
            "/compute/motif-analysis",
            {"structure_id": structure_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def run_graph_topology(self, structure_id: str, **kwargs: Any) -> dict[str, Any]:
        """Compute graph topology metrics for a structure (independent of pipeline)."""
        return await self._post(
            "/compute/graph-topology",
            {"structure_id": structure_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def run_md_validate(
        self, structure_id: str, site_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Run MD validation on a cryptic site."""
        return await self._post(
            "/compute/md-validate",
            {"structure_id": structure_id, "site_id": site_id, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def ingest_structure(
        self, pdb_id: str, force_reingest: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Ingest a structure via the full BinaryCIF pipeline.

        Calls POST /compute/ingest-full on the science container.
        Returns structure_id, counts, scope, alignment_status.
        """
        return await self._post(
            "/compute/ingest-full",
            {"pdb_id": pdb_id, "force_reingest": force_reingest, **kwargs},
            timeout=self._pipeline_timeout,
        )

    async def _get(self, path: str, timeout: float) -> dict[str, Any]:
        """Execute a GET request against the Science API."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self._base_url}{path}")
            if resp.status_code >= 400:
                detail = self._extract_detail(resp)
                raise ScienceComputeError(path, resp.status_code, detail)
            return resp.json()
        except httpx.TimeoutException:
            raise ScienceTimeoutError(path, timeout)

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Execute a POST request against the Science API."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self._base_url}{path}", json=payload)
            if resp.status_code >= 400:
                detail = self._extract_detail(resp)
                raise ScienceComputeError(path, resp.status_code, detail)
            return resp.json()
        except httpx.TimeoutException:
            raise ScienceTimeoutError(path, timeout)

    @staticmethod
    def _extract_detail(resp: httpx.Response) -> str:
        """Extract error detail from a response body."""
        try:
            body = resp.json()
            if isinstance(body, dict):
                return str(body.get("detail", resp.text))
            return resp.text
        except Exception:
            return resp.text
