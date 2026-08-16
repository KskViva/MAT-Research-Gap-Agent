"""Stable contracts for deterministic, traceable research reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from research_gap_agent.evidence import EvidenceResearchStatus, EvidenceToolTrace
from research_gap_agent.synthesis import (
    Claim,
    GapAnalysisResult,
    GapAnalysisStatus,
    GapCandidate,
    GapVerificationStatus,
    ModelReasoningTrace,
    VerifiedGap,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class ReportStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_CONCLUSIONS = "no_conclusions"
    INVALID_TRACEABILITY = "invalid_traceability"


class ReportConclusionKind(StrEnum):
    CLAIM = "claim"
    VERIFIED_GAP = "verified_gap"
    REFINED_GAP = "refined_gap"
    REJECTED_CANDIDATE = "rejected_candidate"


class EvidenceOrigin(StrEnum):
    SOURCE = "source"
    COUNTER = "counter"


class ReportRequest(_StrictModel):
    """Module 4 input and presentation-only options."""

    analysis_result: GapAnalysisResult
    title: str = Field(
        default="Traceable Research Gap Report",
        min_length=3,
        max_length=300,
    )
    include_rejected_candidates: bool = True
    include_evidence_excerpts: bool = True
    max_quote_characters: int = Field(default=500, ge=100, le=4000)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return " ".join(value.split())


class EvidenceCitation(_StrictModel):
    """One qualified report reference resolving to SciVerse provenance."""

    reference_id: str = Field(min_length=1, max_length=300)
    origin: EvidenceOrigin
    gap_id: str | None = None
    evidence_id: str = Field(min_length=1, max_length=200)
    source_tool_call_id: str = Field(min_length=1, max_length=300)
    doc_id: str = Field(min_length=1, max_length=300)
    chunk_id: str | None = None
    offset: int = Field(ge=0)
    title: str | None = None
    publication_year: int | None = Field(default=None, ge=1400, le=2200)
    score: float | None = None
    excerpt: str | None = None
    excerpt_truncated: bool = False


class ReportConclusion(_StrictModel):
    """A conclusion whose references have passed local integrity checks."""

    conclusion_id: str = Field(pattern=r"^conclusion-[1-9][0-9]*$")
    kind: ReportConclusionKind
    text: str = Field(min_length=3, max_length=6000)
    status: GapVerificationStatus | None = None
    rationale: str | None = Field(default=None, min_length=3, max_length=6000)
    uncertainty: str | None = Field(default=None, min_length=3, max_length=4000)
    claim_ids: list[str] = Field(default_factory=list, max_length=50)
    gap_id: str | None = None
    evidence_references: list[str] = Field(min_length=1, max_length=100)
    counter_evidence_references: list[str] = Field(
        default_factory=list,
        max_length=50,
    )


class CounterSearchAudit(_StrictModel):
    """Bounded counter-search provenance retained in the report."""

    gap_id: str = Field(pattern=r"^gap-[1-9][0-9]*$")
    query: str = Field(min_length=3, max_length=4096)
    status: EvidenceResearchStatus
    evidence_references: list[str] = Field(default_factory=list, max_length=100)
    tool_traces: list[EvidenceToolTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TraceabilityIssue(_StrictModel):
    """A broken or ambiguous link that prevented report inclusion."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    path: str = Field(min_length=1, max_length=500)
    message: str = Field(min_length=3, max_length=1000)


class TraceableReport(_StrictModel):
    """Structured report plus a deterministic Markdown rendering."""

    title: str
    question: str
    status: ReportStatus
    analysis_status: GapAnalysisStatus
    conclusions: list[ReportConclusion] = Field(default_factory=list)
    claim_register: list[Claim] = Field(default_factory=list)
    gap_candidate_register: list[GapCandidate] = Field(default_factory=list)
    gap_verification_register: list[VerifiedGap] = Field(default_factory=list)
    evidence_register: list[EvidenceCitation] = Field(default_factory=list)
    counter_searches: list[CounterSearchAudit] = Field(default_factory=list)
    model_traces: list[ModelReasoningTrace] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    integrity_issues: list[TraceabilityIssue] = Field(default_factory=list)
    markdown: str
