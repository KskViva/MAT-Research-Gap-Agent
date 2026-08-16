"""Stable contracts for SciVerse evidence acquisition."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_gap_agent.agent.models import PaperRecord


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class PaperScope(_StrictModel):
    """Optional structured scope used to build the paper candidate set."""

    query: str | None = Field(default=None, min_length=1, max_length=4096)
    authors: list[str] = Field(default_factory=list, max_length=100)
    year_from: int | None = Field(default=None, ge=1400, le=2200)
    year_to: int | None = Field(default=None, ge=1400, le=2200)
    journals: list[str] = Field(default_factory=list, max_length=100)
    subjects: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_years(self) -> PaperScope:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class ResearchLimits(_StrictModel):
    """Bounded limits for remote retrieval and returned context."""

    candidate_papers: int = Field(default=10, ge=1, le=50)
    evidence_chunks: int = Field(default=10, ge=1, le=100)
    context_expansions: int = Field(default=3, ge=0, le=20)
    read_bytes: int = Field(default=4096, ge=1, le=16384)


class ResearchRequest(_StrictModel):
    """One evidence-retrieval request, without synthesis or gap claims."""

    question: str = Field(min_length=3, max_length=4096)
    scope: PaperScope | None = None
    limits: ResearchLimits = Field(default_factory=ResearchLimits)
    semantic_mode: Literal["fast", "balanced", "quality"] = "balanced"
    strict_paper_scope: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 3:
            raise ValueError("question must contain at least 3 characters")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self) -> ResearchRequest:
        if self.limits.context_expansions > self.limits.evidence_chunks:
            raise ValueError(
                "context_expansions must not exceed evidence_chunks"
            )
        return self


class EvidenceToolTrace(_StrictModel):
    """One locally validated SciVerse call made by evidence research."""

    sequence: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error_type: str | None = None
    error_message: str | None = None
    result_count: int = Field(default=0, ge=0)


class OriginalTextExcerpt(_StrictModel):
    """A bounded original-text range returned by ``read_content``."""

    source_tool_call_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    offset: int = Field(ge=0)
    text: str = Field(min_length=1)
    bytes_returned: int | None = Field(default=None, ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    more: bool | None = None


class EvidenceRef(_StrictModel):
    """One SciVerse semantic hit with unchanged source identifiers."""

    evidence_id: str = Field(pattern=r"^evidence-[1-9][0-9]*$")
    source_tool_call_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    chunk_id: str | None = None
    offset: int = Field(ge=0)
    title: str | None = None
    publication_year: int | None = Field(default=None, ge=1400, le=2200)
    score: float | None = None
    quoted_text: str = Field(min_length=1)
    context: OriginalTextExcerpt | None = None
    raw_chunk: dict[str, Any]


class EvidenceResearchStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class EvidenceResearchResult(_StrictModel):
    """Structured output of Module 2 evidence retrieval."""

    status: EvidenceResearchStatus
    request: ResearchRequest
    papers: list[PaperRecord] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    tool_traces: list[EvidenceToolTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
