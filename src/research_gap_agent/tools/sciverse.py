"""Validated LLM tool wrappers for the standard SciVerse text capabilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_gap_agent.sciverse import (
    McpToolCaller,
    StdioSciverseMcpClient,
)

from .registry import ToolRegistry


class _StrictArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class CatalogArguments(_StrictArguments):
    collection: Literal["papers", "authors", "sources"] = "papers"
    include_sample_values: bool = False
    include_field_stats: bool = False


class AdvancedFilter(_StrictArguments):
    field: str = Field(min_length=1, max_length=200)
    operator: Literal[
        "FILTER_OP_EQ",
        "FILTER_OP_NE",
        "FILTER_OP_GT",
        "FILTER_OP_GTE",
        "FILTER_OP_LT",
        "FILTER_OP_LTE",
        "FILTER_OP_IN",
        "FILTER_OP_NIN",
        "FILTER_OP_CONTAINS",
        "FILTER_OP_MATCH",
        "FILTER_OP_MATCH_PHRASE",
    ] = "FILTER_OP_EQ"
    value: Any


class SearchPapersArguments(_StrictArguments):
    collection: Literal["papers", "authors", "sources"] = "papers"
    query: str | None = Field(default=None, min_length=1, max_length=4096)
    title_contains: str | None = Field(default=None, min_length=1, max_length=500)
    abstract_contains: str | None = Field(default=None, min_length=1, max_length=1000)
    authors: list[str] = Field(default_factory=list, max_length=100)
    year_from: int | None = Field(default=None, ge=1400, le=2200)
    year_to: int | None = Field(default=None, ge=1400, le=2200)
    journals: list[str] = Field(default_factory=list, max_length=100)
    subjects: list[str] = Field(default_factory=list, max_length=100)
    filters_advanced: list[AdvancedFilter] = Field(default_factory=list, max_length=50)
    sort_by_year: Literal["desc", "asc", "none"] = "none"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def validate_search(self) -> SearchPapersArguments:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        if not any(
            (
                self.query,
                self.title_contains,
                self.abstract_contains,
                self.authors,
                self.journals,
                self.subjects,
                self.filters_advanced,
            )
        ):
            raise ValueError("at least one search term or filter is required")
        return self


class SemanticSearchArguments(_StrictArguments):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["fast", "balanced", "quality"] = "balanced"
    source_types: list[Literal["web", "pdf"]] = Field(
        default_factory=lambda: ["pdf"],
        max_length=2,
    )
    filters: dict[str, Any] = Field(default_factory=dict)


class ReadContentArguments(_StrictArguments):
    doc_id: str = Field(min_length=1, max_length=256)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4096, ge=1, le=16384)


class GetResourceArguments(_StrictArguments):
    file_name: str = Field(min_length=1, max_length=500)

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        if value.startswith("/") or "\\" in value or ".." in value:
            raise ValueError("file_name must be a safe SciVerse relative resource path")
        return value


_TOOL_DESCRIPTIONS = {
    "list_catalog": (
        "Inspect SciVerse searchable fields and filter operators. Use this before "
        "building an unfamiliar advanced search filter; it does not retrieve papers."
    ),
    "search_papers": (
        "Search SciVerse paper metadata with explicit terms and filters. Use it to "
        "build a paper or doc_id candidate set; use semantic_search for evidence."
    ),
    "semantic_search": (
        "Retrieve relevant full-text evidence chunks from SciVerse for a natural-"
        "language scientific question. Returned metadata filters except doc_id are soft."
    ),
    "read_content": (
        "Read a bounded original-text range from SciVerse using a doc_id and byte "
        "offset returned by search. This returns text context, not a PDF file."
    ),
    "get_resource": (
        "Retrieve one figure or table image referenced by read_content. The result "
        "contains base64 image bytes and MIME type; it is not a paper or PDF file."
    ),
}


def build_sciverse_tool_registry(
    *,
    client: McpToolCaller | None = None,
    env_file: str | Path = ".env",
    server_command: str | Path | None = None,
    include_resources: bool = False,
) -> ToolRegistry:
    """Build the default text-safe SciVerse registry for an LLM runtime."""

    caller = client or StdioSciverseMcpClient(
        env_file=env_file,
        server_command=server_command,
    )
    registry = ToolRegistry()
    _register(registry, caller, "list_catalog", CatalogArguments)
    _register(registry, caller, "search_papers", SearchPapersArguments)
    _register(registry, caller, "semantic_search", SemanticSearchArguments)
    _register(registry, caller, "read_content", ReadContentArguments)
    if include_resources:
        _register(registry, caller, "get_resource", GetResourceArguments)
    return registry


def _register(
    registry: ToolRegistry,
    client: McpToolCaller,
    name: str,
    input_model: type[_StrictArguments],
) -> None:
    handler: Callable[[_StrictArguments], dict[str, Any]] = lambda arguments: (
        client.call_tool(
            name,
            arguments.model_dump(mode="json", exclude_none=True),
        )
    )
    registry.register(
        name=name,
        description=_TOOL_DESCRIPTIONS[name],
        input_model=input_model,
        handler=handler,
    )
