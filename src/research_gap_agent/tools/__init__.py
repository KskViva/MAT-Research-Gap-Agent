"""Minimal, framework-neutral tools exposed to future LLM runtimes."""

from .models import ToolCallResult, ToolDefinition
from .registry import ToolRegistry
from .sciverse import (
    CatalogArguments,
    GetResourceArguments,
    ReadContentArguments,
    SearchPapersArguments,
    SemanticSearchArguments,
    build_sciverse_tool_registry,
)

__all__ = [
    "CatalogArguments",
    "GetResourceArguments",
    "ReadContentArguments",
    "SearchPapersArguments",
    "SemanticSearchArguments",
    "ToolCallResult",
    "ToolDefinition",
    "ToolRegistry",
    "build_sciverse_tool_registry",
]
