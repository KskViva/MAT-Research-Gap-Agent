"""Stable contracts for LLM tool orchestration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LlmToolCall(_StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class LlmReply(_StrictModel):
    content: str | None = None
    tool_calls: list[LlmToolCall] = Field(default_factory=list)


class AgentStatus(StrEnum):
    COMPLETE = "complete"
    MAX_TOOL_ROUNDS = "max_tool_rounds"


class AgentToolTrace(_StrictModel):
    round_index: int = Field(ge=1)
    call_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    ok: bool
    error_type: str | None = None
    error_message: str | None = None
    result_characters: int = Field(default=0, ge=0)
    result_truncated: bool = False


class PaperRecord(_StrictModel):
    """One SciVerse metadata hit with normalized accessors and untouched source."""

    source_tool_call_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    unique_id: str | None = None
    doc_id: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    publication_year: int | None = Field(default=None, ge=1400, le=2200)
    venue: str | None = None
    raw_metadata: dict[str, Any]


class AgentResult(_StrictModel):
    status: AgentStatus
    model: str
    answer: str
    tool_traces: list[AgentToolTrace] = Field(default_factory=list)
    papers: list[PaperRecord] = Field(default_factory=list)
