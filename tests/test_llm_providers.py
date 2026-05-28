"""Tests for LLM providers — mock provider, agent loop, and budget tracking."""

from __future__ import annotations

import pytest

from agent.llm.base import Agent, LLMResponse, Message, ToolCall, ToolDefinition, UsageStats


class TestUsageStats:
    def test_initial_state(self):
        usage = UsageStats()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.llm_calls == 0
        assert usage.total_tokens == 0

    def test_add_accumulates(self):
        usage = UsageStats()
        usage.add(input_tokens=100, output_tokens=50)
        usage.add(input_tokens=200, output_tokens=80)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 130
        assert usage.total_tokens == 430
        assert usage.llm_calls == 2


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_mock_responds_to_source_leak_query(self):
        from agent.llm.providers import MockProvider

        provider = MockProvider()
        tools = [
            ToolDefinition(
                name="get_source_leaks",
                description="Find source leaks",
                parameters={"type": "object", "properties": {}},
                handler=None,
            ),
        ]

        response = await provider.chat(
            messages=[Message(role="user", content="Show me source leaks in 4obe")],
            tools=tools,
        )
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.message.tool_calls is not None
        assert response.message.tool_calls[0].name == "get_source_leaks"

    @pytest.mark.asyncio
    async def test_mock_returns_text_for_generic_query(self):
        from agent.llm.providers import MockProvider

        provider = MockProvider()
        response = await provider.chat(
            messages=[Message(role="user", content="Hello, what can you do?")],
            tools=[],
        )
        assert response.message.content != ""
        assert response.message.tool_calls is None

    @pytest.mark.asyncio
    async def test_mock_reports_token_usage(self):
        from agent.llm.providers import MockProvider

        provider = MockProvider()
        response = await provider.chat(
            messages=[Message(role="user", content="Tell me about uncertainty")],
            tools=[],
        )
        assert response.input_tokens > 0
        assert response.output_tokens > 0


class TestAgent:
    """Test the Agent loop with a controllable mock provider."""

    @pytest.fixture
    def echo_tool(self):
        """A simple tool that echoes its input."""
        async def _echo(message: str = "hello") -> dict:
            return {"echoed": message}

        return ToolDefinition(
            name="echo",
            description="Echo a message",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            handler=_echo,
        )

    @pytest.fixture
    def counting_provider(self):
        """Provider that calls a tool once, then responds with text."""

        class CountingProvider:
            def __init__(self):
                self.call_count = 0

            async def chat(self, messages, tools, system_prompt=None):
                self.call_count += 1
                if self.call_count == 1:
                    # First call: request a tool call
                    return LLMResponse(
                        message=Message(
                            role="assistant",
                            content="",
                            tool_calls=[ToolCall(id="tc1", name="echo", arguments={"message": "test"})],
                        ),
                        input_tokens=100,
                        output_tokens=30,
                    )
                else:
                    # Second call: final response
                    return LLMResponse(
                        message=Message(role="assistant", content="Done! I echoed: test"),
                        input_tokens=150,
                        output_tokens=20,
                    )

        return CountingProvider()

    @pytest.mark.asyncio
    async def test_agent_executes_tool_and_returns(self, echo_tool, counting_provider):
        agent = Agent(
            name="test",
            system_prompt="You are a test agent.",
            tools=[echo_tool],
            llm=counting_provider,
        )

        response = await agent.run("Echo something")
        assert response.text == "Done! I echoed: test"
        assert "echo" in response.tool_calls_made
        assert response.usage.llm_calls == 2
        assert response.usage.total_tokens == 300  # 100+150 input + 30+20 output

    @pytest.mark.asyncio
    async def test_agent_respects_token_budget(self, echo_tool):
        """Agent stops when token budget is exceeded."""

        class ExpensiveProvider:
            async def chat(self, messages, tools, system_prompt=None):
                # Always request another tool call (infinite loop without budget)
                return LLMResponse(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(id="tc1", name="echo", arguments={"message": "x"})],
                    ),
                    input_tokens=50000,
                    output_tokens=10000,
                )

        agent = Agent(
            name="test",
            system_prompt="You are a test agent.",
            tools=[echo_tool],
            llm=ExpensiveProvider(),
            max_tokens=100_000,
        )

        response = await agent.run("Keep going forever")
        # Should stop after 1-2 iterations due to token budget
        assert response.usage.llm_calls <= 2
        assert "stop here" in response.text.lower()

    @pytest.mark.asyncio
    async def test_agent_context_in_system_prompt_not_message(self, echo_tool):
        """Context should be in system prompt, not appended to user message."""

        class InspectingProvider:
            def __init__(self):
                self.last_system_prompt = None
                self.last_messages = None

            async def chat(self, messages, tools, system_prompt=None):
                self.last_system_prompt = system_prompt
                self.last_messages = messages
                return LLMResponse(
                    message=Message(role="assistant", content="OK"),
                    input_tokens=10,
                    output_tokens=5,
                )

        provider = InspectingProvider()
        agent = Agent(
            name="test",
            system_prompt="Base prompt.",
            tools=[echo_tool],
            llm=provider,
        )

        await agent.run("Hello", context={"session_id": "abc", "viewport": {"structure_id": "4obe"}})

        # Context should be in system prompt
        assert "<session_context>" in provider.last_system_prompt
        assert "4obe" in provider.last_system_prompt

        # User message should NOT contain context
        user_msg = provider.last_messages[0]
        assert "<session_context>" not in user_msg.content
        assert user_msg.content == "Hello"
