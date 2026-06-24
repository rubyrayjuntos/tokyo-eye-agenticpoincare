"""Helpers for compacting tool schemas before sending them to an LLM."""

from __future__ import annotations

from typing import Any

from agent.llm.base import ToolDefinition

_TOOL_DESCRIPTION_MAX_CHARS = 120


def compact_text(text: str | None, max_chars: int = _TOOL_DESCRIPTION_MAX_CHARS) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    first_sentence = normalized.split(". ")[0].strip()
    candidate = first_sentence if first_sentence else normalized
    if len(candidate) <= max_chars:
        return candidate
    return candidate[: max(max_chars - 3, 0)] + "..."


def compact_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    compact: dict[str, Any] = {}
    for key in ("type", "enum", "required", "default", "additionalProperties"):
        if key in schema:
            compact[key] = schema[key]

    if "properties" in schema and isinstance(schema["properties"], dict):
        compact["properties"] = {
            prop_name: compact_schema(prop_schema)
            for prop_name, prop_schema in schema["properties"].items()
        }

    if "items" in schema:
        compact["items"] = compact_schema(schema["items"])

    if "$ref" in schema:
        compact["$ref"] = schema["$ref"]

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        compact["anyOf"] = [compact_schema(item) for item in schema["anyOf"]]

    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        compact["oneOf"] = [compact_schema(item) for item in schema["oneOf"]]

    return compact


def compact_tool_definitions(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": compact_text(tool.description),
            "input_schema": compact_schema(tool.parameters),
        }
        for tool in tools
    ]
