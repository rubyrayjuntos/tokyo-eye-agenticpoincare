"""Base LLM interface — framework-agnostic tool-calling agent.

This module defines the core abstractions for the agent system.
It's designed to work with any LLM backend that supports tool calling:
- AWS Bedrock (Claude, Titan)
- Anthropic API directly
- OpenAI / Azure OpenAI
- Google Gemini
- Local models via Ollama

The actual LLM provider is configured via environment variables.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool the agent can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    handler: Callable[..., Awaitable[Any]]  # Async function to execute


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system", "tool_result"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False
    # Structured data preserved through the pipeline (avoids re-parsing JSON)
    structured_data: dict[str, Any] | None = None


@dataclass
class UsageStats:
    """Token usage and cost tracking for an agent run."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1


@dataclass
class AgentResponse:
    """Complete response from the agent (may include multiple tool calls)."""

    text: str
    tool_calls_made: list[str] = field(default_factory=list)
    viewport_directives: list[dict] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Raw response from an LLM provider, including usage metadata."""

    message: Message
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response (possibly with tool calls).

        Returns LLMResponse which includes the message and token usage stats.
        """
        ...


class Agent:
    """A tool-calling agent that uses an LLM to reason and act.

    The agent loop:
    1. Send user message + conversation history to LLM
    2. If LLM requests tool calls, execute them
    3. Send tool results back to LLM
    4. Repeat until LLM produces a final text response

    Budget controls:
    - max_iterations: hard cap on LLM round-trips
    - max_tokens: soft cap on total token usage (checked after each call)
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        llm: LLMProvider,
        max_iterations: int = 10,
        max_tokens: int = 100_000,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self._tool_map = {t.name: t for t in tools}

    async def run(self, user_message: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Run the agent on a user message.

        Args:
            user_message: The user's input.
            context: Additional context (viewport state, session data, etc.)
                     Injected into the system prompt, not repeated per turn.

        Returns:
            AgentResponse with the final text, tool calls made, and usage stats.
        """
        messages: list[Message] = []
        messages.append(Message(role="user", content=user_message))

        # Build system prompt with context (injected once, not per-turn)
        system_prompt = self.system_prompt
        if context:
            context_str = json.dumps(context, indent=2, default=str)
            system_prompt = f"{self.system_prompt}\n\n<session_context>\n{context_str}\n</session_context>"

        tool_calls_made: list[str] = []
        viewport_directives: list[dict] = []
        usage = UsageStats()

        for iteration in range(self.max_iterations):
            # Check token budget before making another call
            if usage.total_tokens >= self.max_tokens:
                logger.warning(
                    "Agent %s hit token budget (%d/%d) after %d iterations",
                    self.name, usage.total_tokens, self.max_tokens, iteration,
                )
                break

            # Get LLM response
            llm_response = await self.llm.chat(
                messages=messages,
                tools=self.tools,
                system_prompt=system_prompt,
            )

            # Track usage
            usage.add(
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
            )

            response = llm_response.message

            # If no tool calls, we're done
            if not response.tool_calls:
                return AgentResponse(
                    text=response.content,
                    tool_calls_made=tool_calls_made,
                    viewport_directives=viewport_directives,
                    usage=usage,
                )

            # Execute tool calls
            messages.append(response)  # Add assistant message with tool calls

            tool_results: list[ToolResult] = []
            for call in response.tool_calls:
                tool_calls_made.append(call.name)
                result = await self._execute_tool(call)
                tool_results.append(result)

                # Extract viewport directives from structured data (no re-parsing)
                if result.structured_data and "viewport_directives" in result.structured_data:
                    viewport_directives.extend(result.structured_data["viewport_directives"])

            # Add tool results to conversation
            messages.append(Message(
                role="tool_result",
                content="",
                tool_results=tool_results,
            ))

        # Max iterations or token budget reached
        logger.warning(
            "Agent %s stopped: iterations=%d, tokens=%d/%d",
            self.name, usage.llm_calls, usage.total_tokens, self.max_tokens,
        )
        return AgentResponse(
            text="I've been working on this but need to stop here. Here's what I found so far.",
            tool_calls_made=tool_calls_made,
            viewport_directives=viewport_directives,
            usage=usage,
        )

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Keeps structured data alongside the serialized content string
        to avoid double-serialization when extracting viewport directives.
        """
        tool = self._tool_map.get(call.name)
        if not tool:
            return ToolResult(
                tool_call_id=call.id,
                content=json.dumps({"error": f"Unknown tool: {call.name}"}),
                is_error=True,
            )

        try:
            result = await tool.handler(**call.arguments)

            # Convert to dict (structured) and string (for LLM context)
            if hasattr(result, 'model_dump'):
                structured = result.model_dump(mode="python")
            elif hasattr(result, '__dict__'):
                structured = result.__dict__
            else:
                structured = result if isinstance(result, dict) else {"value": result}

            content = json.dumps(structured, default=str)

            return ToolResult(
                tool_call_id=call.id,
                content=content,
                structured_data=structured,
            )
        except Exception as e:
            logger.exception("Tool %s failed", call.name)
            return ToolResult(
                tool_call_id=call.id,
                content=json.dumps({"error": str(e)}),
                is_error=True,
            )
