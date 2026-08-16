"""Stable schemas for describing and returning LLM tool calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """One callable tool and its JSON Schema input contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=1000)
    input_schema: dict[str, Any]

    def as_function_schema(self) -> dict[str, Any]:
        """Return the common LLM function-tool representation."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolCallResult(BaseModel):
    """Serializable success or failure returned by the tool dispatcher."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_name: str
    ok: bool
    output: Any | None = None
    error_type: str | None = None
    error_message: str | None = None
