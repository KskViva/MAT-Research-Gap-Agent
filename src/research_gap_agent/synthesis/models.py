"""Stable contracts for evidence-grounded synthesis and gap verification."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    ResearchLimits,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class GapCategory(StrEnum):
    SPARSE_COVERAGE = "sparse_coverage"
    MISSING_CONDITION = "missing_condition"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    METHODOLOGICAL_GAP = "methodological_gap"


class GapVerificationStatus(StrEnum):
    REJECTED = "rejected"
    REFINED = "refined"
    VERIFIED = "verified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GapAnalysisStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class ClaimProposal(_StrictModel):
    """LLM proposal that may reference only supplied evidence IDs."""

    statement: str = Field(min_length=3, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


class GapCandidateProposal(_StrictModel):
    """LLM gap proposal referencing 1-based ClaimProposal positions."""

    statement: str = Field(min_length=3, max_length=4000)
    rationale: str = Field(min_length=3, max_length=6000)
    category: GapCategory
    supporting_claim_numbers: list[int] = Field(min_length=1, max_length=50)
    uncertainty: str = Field(min_length=3, max_length=4000)
    counter_query: str = Field(min_length=3, max_length=4096)

    @field_validator("supporting_claim_numbers")
    @classmethod
    def validate_claim_numbers(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("supporting_claim_numbers must be positive")
        return list(dict.fromkeys(values))


class SynthesisDraft(_StrictModel):
    """Strict JSON contract returned by the synthesis reasoning step."""

    claims: list[ClaimProposal] = Field(default_factory=list, max_length=50)
    gap_candidates: list[GapCandidateProposal] = Field(
        default_factory=list,
        max_length=10,
    )


class GapAssessmentDraft(_StrictModel):
    """Strict JSON contract returned after counter-evidence retrieval."""

    status: GapVerificationStatus
    rationale: str = Field(min_length=3, max_length=6000)
    counter_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    refined_scope: str | None = Field(default=None, min_length=3, max_length=4000)

    @field_validator("counter_evidence_ids")
    @classmethod
    def unique_counter_evidence_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def require_refined_scope(self) -> GapAssessmentDraft:
        if (
            self.status == GapVerificationStatus.REFINED
            and self.refined_scope is None
        ):
            raise ValueError("refined_scope is required when status is refined")
        return self


class Claim(_StrictModel):
    """Evidence-grounded scientific statement."""

    claim_id: str = Field(pattern=r"^claim-[1-9][0-9]*$")
    statement: str = Field(min_length=3, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)


class GapCandidate(_StrictModel):
    """A hypothesis to falsify, not yet a verified research gap."""

    gap_id: str = Field(pattern=r"^gap-[1-9][0-9]*$")
    statement: str = Field(min_length=3, max_length=4000)
    rationale: str = Field(min_length=3, max_length=6000)
    category: GapCategory
    supporting_claim_ids: list[str] = Field(min_length=1, max_length=50)
    uncertainty: str = Field(min_length=3, max_length=4000)
    counter_query: str = Field(min_length=3, max_length=4096)


class VerifiedGap(_StrictModel):
    """A candidate after counter-search and locally enforced status policy."""

    gap_id: str = Field(pattern=r"^gap-[1-9][0-9]*$")
    status: GapVerificationStatus
    rationale: str = Field(min_length=3, max_length=6000)
    supporting_claim_ids: list[str] = Field(min_length=1, max_length=50)
    supporting_evidence_ids: list[str] = Field(min_length=1, max_length=100)
    counter_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    refined_scope: str | None = Field(default=None, min_length=3, max_length=4000)


class GapAnalysisLimits(_StrictModel):
    """Bounds both LLM synthesis and one counter-search per candidate."""

    max_input_evidence: int = Field(default=20, ge=1, le=100)
    max_claims: int = Field(default=12, ge=1, le=50)
    max_gap_candidates: int = Field(default=3, ge=1, le=10)
    counter_research: ResearchLimits = Field(
        default_factory=lambda: ResearchLimits(
            candidate_papers=10,
            evidence_chunks=5,
            context_expansions=2,
            read_bytes=4096,
        )
    )
    counter_semantic_mode: Literal["fast", "balanced", "quality"] = "balanced"


class GapAnalysisRequest(_StrictModel):
    """Module 3 input: a Module 2 result plus bounded analysis limits."""

    evidence_result: EvidenceResearchResult
    limits: GapAnalysisLimits = Field(default_factory=GapAnalysisLimits)


class ModelReasoningTrace(_StrictModel):
    """Safe status record for one structured reasoning request."""

    sequence: int = Field(ge=1)
    stage: Literal["synthesis", "verification"]
    gap_id: str | None = None
    ok: bool
    error_type: str | None = None
    error_message: str | None = None


class CounterResearchRecord(_StrictModel):
    """Counter-query and complete Module 2 provenance for one candidate."""

    gap_id: str = Field(pattern=r"^gap-[1-9][0-9]*$")
    query: str = Field(min_length=3, max_length=4096)
    result: EvidenceResearchResult


class GapAnalysisResult(_StrictModel):
    """Structured Module 3 output; it is not a narrative report."""

    status: GapAnalysisStatus
    question: str = Field(min_length=3, max_length=4096)
    model: str
    source_evidence: list[EvidenceRef] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    gap_candidates: list[GapCandidate] = Field(default_factory=list)
    verified_gaps: list[VerifiedGap] = Field(default_factory=list)
    counter_research: list[CounterResearchRecord] = Field(default_factory=list)
    model_traces: list[ModelReasoningTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
