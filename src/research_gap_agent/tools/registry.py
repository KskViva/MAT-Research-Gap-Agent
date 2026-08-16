"""Small validated registry that an LLM runtime can inspect and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolCallResult, ToolDefinition

InputModel = TypeVar("InputModel", bound=BaseModel)
ToolHandler = Callable[[BaseModel], Any]


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    definition: ToolDefinition
    input_model: type[BaseModel]
    handler: ToolHandler


class ToolRegistry:
    """Register typed handlers and expose JSON schemas to an LLM runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        input_model: type[InputModel],
        handler: Callable[[InputModel], Any],
    ) -> ToolDefinition:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")

        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_model.model_json_schema(),
        )
        self._tools[name] = _RegisteredTool(
            definition=definition,
            input_model=input_model,
            handler=handler,
        )
        return definition

    def definitions(self) -> list[ToolDefinition]:
        """Return definitions in deterministic registration order."""

        return [tool.definition for tool in self._tools.values()]

    def function_schemas(self) -> list[dict[str, Any]]:
        """Return schemas ready for an LLM function/tool parameter."""

        return [definition.as_function_schema() for definition in self.definitions()]

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Validate arguments, dispatch the handler, and serialize its result."""

        registered = self._tools.get(name)
        if registered is None:
            return ToolCallResult(
                tool_name=name,
                ok=False,
                error_type="tool_not_found",
                error_message=f"unknown tool: {name}",
            )

        try:
            validated = registered.input_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolCallResult(
                tool_name=name,
                ok=False,
                error_type="validation_error",
                error_message=str(exc)[:1000],
            )

        try:
            output = registered.handler(validated)
        except Exception as exc:  # Boundary: never crash the outer LLM loop.
            return ToolCallResult(
                tool_name=name,
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )

        return ToolCallResult(
            tool_name=name,
            ok=True,
            output=_json_ready(output),
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
