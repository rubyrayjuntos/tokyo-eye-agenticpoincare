"""LLM provider implementations.

Supports:
- AWS Bedrock (Claude) — primary for production
- Anthropic API — for direct access
- Mock provider — for testing without API keys
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from agent.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

# Default timeout for LLM calls (seconds)
_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the configured timeout."""
    pass


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider (Claude models).

    Uses boto3 to call Bedrock's converse API with tool use.
    Requires AWS credentials configured (env vars, IAM role, or profile).

    Includes:
    - Exponential backoff via boto3's built-in retry config
    - Configurable timeout per call
    """

    @property
    def supports_vision(self) -> bool:
        """Bedrock Claude models support vision/multimodal input."""
        return True

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int = 4096,
        timeout: int = _LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Built-in retry with exponential backoff for throttling
            retry_config = Config(
                retries={
                    "max_attempts": self.max_retries,
                    "mode": "adaptive",  # Adaptive retry with token bucket
                },
                read_timeout=self.timeout,
                connect_timeout=10,
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=retry_config,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert to Bedrock format
        bedrock_messages = self._format_messages(messages)
        tool_config = self._format_tools(tools) if tools else None

        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        if tool_config:
            kwargs["toolConfig"] = tool_config

        # Run synchronous boto3 call in thread with timeout
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(client.converse, **kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Bedrock call timed out after {self.timeout}s (model={self.model_id})"
            )

        return self._parse_response(response)

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.role == "user":
                content = [{"text": msg.content}]
                # Include image content block for multimodal (Requirements 3.2)
                if msg.image:
                    image_data = msg.image
                    # Strip data URI prefix if present
                    if image_data.startswith("data:"):
                        parts = image_data.split(",", 1)
                        if len(parts) == 2:
                            image_data = parts[1]
                    content.append({
                        "image": {
                            "format": "png",
                            "source": {"bytes": image_data},
                        }
                    })
                result.append({"role": "user", "content": content})
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        })
                result.append({"role": "assistant", "content": content})
            elif msg.role == "tool_result":
                if msg.tool_results:
                    content = []
                    for tr in msg.tool_results:
                        content.append({
                            "toolResult": {
                                "toolUseId": tr.tool_call_id,
                                "content": [{"text": tr.content}],
                                "status": "error" if tr.is_error else "success",
                            }
                        })
                    result.append({"role": "user", "content": content})
        return result

    def _format_tools(self, tools: list[ToolDefinition]) -> dict:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.parameters},
                    }
                }
                for t in tools
            ]
        }

    def _parse_response(self, response: dict) -> LLMResponse:
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=tu.get("input", {}),
                ))

        # Extract token usage from Bedrock response
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        return LLMResponse(
            message=Message(
                role="assistant",
                content="\n".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class AnthropicProvider(LLMProvider):
    """Direct Anthropic API provider.

    Uses the anthropic Python SDK. Requires ANTHROPIC_API_KEY env var.

    Includes:
    - Configurable timeout per call
    - The anthropic SDK handles retries internally
    """

    @property
    def supports_vision(self) -> bool:
        """Anthropic Claude models support vision/multimodal input."""
        return True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        timeout: int = _LLM_TIMEOUT,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            # The anthropic SDK accepts a timeout and handles retries
            self._client = anthropic.AsyncAnthropic(
                timeout=float(self.timeout),
                max_retries=3,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert messages
        api_messages = []
        for msg in messages:
            if msg.role == "user":
                # Build multimodal content if image present (Requirements 3.2)
                if msg.image:
                    image_data = msg.image
                    if image_data.startswith("data:"):
                        parts = image_data.split(",", 1)
                        if len(parts) == 2:
                            image_data = parts[1]
                    content = [
                        {"type": "text", "text": msg.content},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                    ]
                    api_messages.append({"role": "user", "content": content})
                else:
                    api_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                api_messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool_result" and msg.tool_results:
                content = []
                for tr in msg.tool_results:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    })
                api_messages.append({"role": "user", "content": content})

        # Convert tools
        api_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ] if tools else None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools

        # The SDK timeout handles hung connections; wrap with asyncio
        # timeout as a secondary safeguard
        try:
            response = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=self.timeout + 10,  # SDK timeout + grace period
            )
        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Anthropic call timed out after {self.timeout}s (model={self.model})"
            )

        # Parse response
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        # Extract token usage from Anthropic response
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        return LLMResponse(
            message=Message(
                role="assistant",
                content="\n".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class MockProvider(LLMProvider):
    """Mock provider for testing without API keys.

    Returns canned responses and simulates tool calling based on keywords.
    Reports synthetic token counts for testing budget logic.
    """

    @property
    def supports_vision(self) -> bool:
        """Mock provider does not support vision by default."""
        return False

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"),
            None,
        )
        if not last_user:
            return LLMResponse(
                message=Message(role="assistant", content="How can I help you?"),
                input_tokens=10,
                output_tokens=5,
            )

        text = last_user.content.lower()
        tool_map = {t.name: t for t in tools}

        # Estimate input tokens (rough: 4 chars per token)
        total_input_chars = sum(len(m.content) for m in messages) + len(system_prompt or "")
        est_input_tokens = max(total_input_chars // 4, 10)

        # Simulate tool calling based on keywords
        if "source leak" in text or "leak" in text:
            if "get_source_leaks" in tool_map:
                return LLMResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(
                            id=str(uuid.uuid4()),
                            name="get_source_leaks",
                            arguments={"structure_id": "4obe", "uncertainty_threshold": 0.3, "min_depth": 1.5},
                        )],
                    ),
                    input_tokens=est_input_tokens,
                    output_tokens=50,
                )

        if "uncertainty" in text or "uncertain" in text:
            if "get_high_uncertainty_residues" in tool_map:
                return LLMResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(
                            id=str(uuid.uuid4()),
                            name="get_high_uncertainty_residues",
                            arguments={"structure_id": "4obe", "top_n": 20, "uncertainty_type": "epistemic"},
                        )],
                    ),
                    input_tokens=est_input_tokens,
                    output_tokens=50,
                )

        if "run" in text and ("pipeline" in text or "gnn" in text or "inference" in text):
            return LLMResponse(
                message=Message(
                    role="assistant",
                    content=(
                        "Compute is only triggered by structure ingest (POST /api/ingest). "
                        "Ingest the PDB ID first; the discovery pathway runs automatically."
                    ),
                ),
                input_tokens=est_input_tokens,
                output_tokens=50,
            )

        if "compare" in text or "mutant" in text or "differential" in text:
            if "compare_wt_mutant" in tool_map:
                return LLMResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(
                            id=str(uuid.uuid4()),
                            name="compare_wt_mutant",
                            arguments={"wt_structure_id": "4obe", "mutant_structure_id": "4obe_g12d"},
                        )],
                    ),
                    input_tokens=est_input_tokens,
                    output_tokens=50,
                )

        # Default: just respond with text
        return LLMResponse(
            message=Message(
                role="assistant",
                content=(
                    "I can help you analyze protein structures using the DTIE v5 pipeline. "
                    "I can detect source leaks, show uncertainty patterns, run the full pipeline, "
                    "or compare wild-type vs mutant structures. What would you like to explore?"
                ),
            ),
            input_tokens=est_input_tokens,
            output_tokens=80,
        )


def get_provider() -> LLMProvider:
    """Get the configured LLM provider based on environment.

    Provider selection is lazy — no network calls are made here.
    Credential validation happens on first actual LLM call.

    Priority:
    1. ANTHROPIC_API_KEY set → AnthropicProvider
    2. AWS credentials appear available → BedrockProvider
    3. Fallback → MockProvider
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic provider")
        return AnthropicProvider()

    # Check for AWS credentials without making a network call.
    # Actual credential validation happens lazily on first Bedrock call.
    # boto3's adaptive retry handles transient STS/Bedrock failures.
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE") or os.getenv("AWS_ROLE_ARN"):
        logger.info("Using Bedrock provider (credentials will be validated on first call)")
        return BedrockProvider()

    # Also try the default credential chain (EC2 instance role, ECS task role, etc.)
    # by checking if boto3 can resolve credentials without a network call.
    try:
        import botocore.session
        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is not None:
            logger.info("Using Bedrock provider (default credential chain)")
            return BedrockProvider()
    except Exception:
        pass

    logger.info("Using Mock provider (no API keys configured)")
    return MockProvider()
