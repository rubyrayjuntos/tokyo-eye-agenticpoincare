"""LLM agent backbone — multi-agent system for Tokyo Eyes.

Architecture:
- Coordinator agent: routes user requests to specialized sub-agents
- Discovery sub-agent: interprets pre-computed pathway artifacts
- Visualization sub-agent: generates viewport directives
- Knowledge sub-agent: answers questions about findings and biology

The agents use tool-calling to interact with the governed data layer.
Framework-agnostic: works with any LLM that supports tool/function calling
(Claude, GPT-4, Gemini, Bedrock, local models).
"""
