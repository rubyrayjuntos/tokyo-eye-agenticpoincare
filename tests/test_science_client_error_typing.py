"""Property-based test for ScienceClient error typing.

Feature: science-container-api, Property 6: ScienceClient error typing

For any Science_API timeout, the ScienceClient SHALL raise ScienceTimeoutError.
For any 4xx/5xx response, it SHALL raise ScienceComputeError with the correct
status code and detail message.

**Validates: Requirements 7.3, 7.4**
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.tools.science_client import (
    ScienceClient,
    ScienceComputeError,
    ScienceTimeoutError,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

error_status_codes = st.sampled_from([400, 401, 403, 404, 409, 422, 500, 502, 503])

detail_messages = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)

endpoints = st.sampled_from([
    "/compute/cryptic-scan",
    "/compute/motif-analysis",
    "/compute/md-validate",
    "/compute/jobs/gnn_inference",
    "/health",
])


# ---------------------------------------------------------------------------
# Mock transport for httpx
# ---------------------------------------------------------------------------


class ErrorTransport(httpx.AsyncBaseTransport):
    """Transport that returns a fixed error response."""

    def __init__(self, status_code: int, detail: str):
        self._status_code = status_code
        self._detail = detail

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        body = json.dumps({"detail": self._detail}).encode()
        return httpx.Response(
            status_code=self._status_code,
            content=body,
            headers={"content-type": "application/json"},
        )


class TimeoutTransport(httpx.AsyncBaseTransport):
    """Transport that always raises a timeout."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestScienceClientErrorTyping:
    """Property-based test for ScienceClient error typing.

    # Feature: science-container-api, Property 6: ScienceClient error typing
    """

    @settings(max_examples=100)
    @given(status_code=error_status_codes, detail=detail_messages)
    def test_property_6_error_responses_raise_compute_error(
        self, status_code: int, detail: str
    ):
        """For any 4xx/5xx response from the Science API, the ScienceClient
        SHALL raise ScienceComputeError with the correct status code and detail.

        **Validates: Requirements 7.4**
        """

        async def run():
            transport = ErrorTransport(status_code, detail)
            # Patch httpx.AsyncClient to use our transport
            original_init = httpx.AsyncClient.__init__

            def patched_init(self_client, **kwargs):
                kwargs["transport"] = transport
                kwargs.pop("timeout", None)
                original_init(self_client, timeout=30.0, **kwargs)

            httpx.AsyncClient.__init__ = patched_init  # type: ignore[assignment]
            try:
                client = ScienceClient(base_url="http://test:8001")
                with pytest.raises(ScienceComputeError) as exc_info:
                    await client.run_cryptic_scan("test_structure")

                assert exc_info.value.status == status_code
                assert exc_info.value.detail == detail
                assert exc_info.value.endpoint == "/compute/cryptic-scan"
            finally:
                httpx.AsyncClient.__init__ = original_init  # type: ignore[assignment]

        asyncio.run(run())

    @settings(max_examples=100)
    @given(endpoint=endpoints)
    def test_property_6_timeouts_raise_timeout_error(self, endpoint: str):
        """For any Science API timeout, the ScienceClient SHALL raise
        ScienceTimeoutError with the endpoint and timeout duration.

        **Validates: Requirements 7.3**
        """

        async def run():
            transport = TimeoutTransport()
            original_init = httpx.AsyncClient.__init__

            def patched_init(self_client, **kwargs):
                kwargs["transport"] = transport
                kwargs.pop("timeout", None)
                original_init(self_client, timeout=30.0, **kwargs)

            httpx.AsyncClient.__init__ = patched_init  # type: ignore[assignment]
            try:
                client = ScienceClient(
                    base_url="http://test:8001",
                    pipeline_timeout=5.0,
                    health_timeout=2.0,
                )

                with pytest.raises(ScienceTimeoutError) as exc_info:
                    if endpoint == "/health":
                        await client.health()
                    elif endpoint == "/compute/cryptic-scan":
                        await client.run_cryptic_scan("test_structure")
                    elif endpoint == "/compute/motif-analysis":
                        await client.run_motif_analysis("test_structure")
                    elif endpoint == "/compute/md-validate":
                        await client.run_md_validate("test_structure", "site_1")
                    elif endpoint == "/compute/jobs/gnn_inference":
                        await client.run_compute_job("gnn_inference", "test_structure")

                assert exc_info.value.endpoint == endpoint
                if endpoint == "/health":
                    assert exc_info.value.timeout == 2.0
                else:
                    assert exc_info.value.timeout == 5.0
            finally:
                httpx.AsyncClient.__init__ = original_init  # type: ignore[assignment]

        asyncio.run(run())
