from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    ResearchLimits,
    ResearchRequest,
)
from research_gap_agent.synthesis import (
    Claim,
    ClaimProposal,
    EvidenceGroundedGapAnalyzer,
    GapAnalysisLimits,
    GapAnalysisRequest,
    GapAnalysisStatus,
    GapAssessmentDraft,
    GapCandidate,
    GapCandidateProposal,
    GapCategory,
    GapVerificationStatus,
    SynthesisDraft,
)


def _evidence(
    evidence_id: str,
    *,
    doc_id: str = "doc-source",
    text: str = "Observed source evidence.",
) -> EvidenceRef:
    rank = int(evidence_id.rsplit("-", 1)[-1])
    return EvidenceRef(
        evidence_id=evidence_id,
        source_tool_call_id="evidence-call-002",
        rank=rank,
        doc_id=doc_id,
        chunk_id=f"chunk-{rank}",
        offset=rank * 100,
        title="Source paper",
        publication_year=2025,
        score=0.9,
        quoted_text=text,
        raw_chunk={"unchanged": True, "rank": rank},
    )


def _research_result(
    *,
    status: EvidenceResearchStatus = EvidenceResearchStatus.COMPLETE,
    evidence: list[EvidenceRef] | None = None,
    question: str = "What remains uncertain about the LLZO interface?",
) -> EvidenceResearchResult:
    return EvidenceResearchResult(
        status=status,
        request=ResearchRequest(
            question=question,
            limits=ResearchLimits(context_expansions=0),
        ),
        evidence=(
            [_evidence("evidence-1"), _evidence("evidence-2")]
            if evidence is None
            else evidence
        ),
    )


def _draft() -> SynthesisDraft:
    return SynthesisDraft(
        claims=[
            ClaimProposal(
                statement="Interface degradation was observed.",
                evidence_ids=["evidence-1"],
            ),
            ClaimProposal(
                statement="The tested conditions were limited.",
                evidence_ids=["evidence-2"],
            ),
        ],
        gap_candidates=[
            GapCandidateProposal(
                statement="Degradation outside the tested conditions is unclear.",
                rationale="The supplied evidence covers a limited condition set.",
                category=GapCategory.MISSING_CONDITION,
                supporting_claim_numbers=[1, 2],
                uncertainty="The initial retrieval may not cover all conditions.",
                counter_query=(
                    "Studies testing LLZO interface degradation across broader "
                    "operating conditions"
                ),
            )
        ],
    )


@dataclass
class FakeReasoningModel:
    draft: SynthesisDraft
    assessments: list[GapAssessmentDraft] = field(default_factory=list)
    synthesize_error: Exception | None = None
    assessment_error: Exception | None = None
    model_name: str = "fake-deepseek"
    synthesis_calls: list[tuple[str, list[EvidenceRef], int, int]] = field(
        default_factory=list
    )
    assessment_calls: list[
        tuple[GapCandidate, list[Claim], EvidenceResearchResult]
    ] = field(default_factory=list)

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_claims: int,
        max_gap_candidates: int,
    ) -> SynthesisDraft:
        self.synthesis_calls.append(
            (question, evidence, max_claims, max_gap_candidates)
        )
        if self.synthesize_error:
            raise self.synthesize_error
        return self.draft

    def assess_gap(
        self,
        candidate: GapCandidate,
        claims: list[Claim],
        counter_result: EvidenceResearchResult,
    ) -> GapAssessmentDraft:
        self.assessment_calls.append((candidate, claims, counter_result))
        if self.assessment_error:
            raise self.assessment_error
        return self.assessments.pop(0)


@dataclass
class FakeEvidenceResearcher:
    results: list[EvidenceResearchResult]
    requests: list[ResearchRequest] = field(default_factory=list)

    def research(
        self,
        request: ResearchRequest | dict[str, Any],
    ) -> EvidenceResearchResult:
        validated = ResearchRequest.model_validate(request)
        self.requests.append(validated)
        return self.results.pop(0)


def test_analysis_builds_grounded_claims_and_rejects_gap_with_counter_evidence() -> None:
    counter = _research_result(
        evidence=[
            _evidence(
                "evidence-1",
                doc_id="doc-counter",
                text="Broader operating conditions were already tested.",
            )
        ],
        question="Counter query",
    )
    reasoning = FakeReasoningModel(
        draft=_draft(),
        assessments=[
            GapAssessmentDraft(
                status=GapVerificationStatus.REJECTED,
                rationale="Counter-evidence covers the proposed condition range.",
                counter_evidence_ids=["evidence-1"],
            )
        ],
    )
    researcher = FakeEvidenceResearcher([counter])

    result = EvidenceGroundedGapAnalyzer(reasoning, researcher).analyze(
        GapAnalysisRequest(evidence_result=_research_result())
    )

    assert result.status == GapAnalysisStatus.COMPLETE
    assert result.model == "fake-deepseek"
    assert result.claims[0].claim_id == "claim-1"
    assert result.claims[0].evidence_ids == ["evidence-1"]
    assert result.gap_candidates[0].supporting_claim_ids == [
        "claim-1",
        "claim-2",
    ]
    verified = result.verified_gaps[0]
    assert verified.status == GapVerificationStatus.REJECTED
    assert verified.supporting_evidence_ids == ["evidence-1", "evidence-2"]
    assert verified.counter_evidence_ids == ["evidence-1"]
    assert researcher.requests[0].strict_paper_scope is False
    assert researcher.requests[0].question == result.gap_candidates[0].counter_query
    assert [trace.ok for trace in result.model_traces] == [True, True]
    assert '"status":"rejected"' in result.model_dump_json()


def test_unknown_source_evidence_reference_rejects_claim_and_gap() -> None:
    draft = _draft()
    draft.claims[0].evidence_ids = ["evidence-999"]
    reasoning = FakeReasoningModel(draft=draft)
    researcher = FakeEvidenceResearcher([])

    result = EvidenceGroundedGapAnalyzer(reasoning, researcher).analyze(
        GapAnalysisRequest(evidence_result=_research_result())
    )

    assert result.status == GapAnalysisStatus.INSUFFICIENT_EVIDENCE
    assert [claim.claim_id for claim in result.claims] == ["claim-1"]
    assert result.gap_candidates == []
    assert researcher.requests == []
    assert any("unknown evidence" in warning for warning in result.warnings)


def test_empty_module_two_evidence_bypasses_llm_and_counter_search() -> None:
    reasoning = FakeReasoningModel(draft=_draft())
    researcher = FakeEvidenceResearcher([])

    result = EvidenceGroundedGapAnalyzer(reasoning, researcher).analyze(
        GapAnalysisRequest(evidence_result=_research_result(evidence=[]))
    )

    assert result.status == GapAnalysisStatus.INSUFFICIENT_EVIDENCE
    assert reasoning.synthesis_calls == []
    assert researcher.requests == []
    assert result.model_traces == []


def test_synthesis_failure_is_returned_as_structured_failed_result() -> None:
    reasoning = FakeReasoningModel(
        draft=_draft(),
        synthesize_error=RuntimeError("invalid structured reply"),
    )

    result = EvidenceGroundedGapAnalyzer(
        reasoning,
        FakeEvidenceResearcher([]),
    ).analyze(GapAnalysisRequest(evidence_result=_research_result()))

    assert result.status == GapAnalysisStatus.FAILED
    assert result.claims == []
    assert result.model_traces[0].ok is False
    assert result.model_traces[0].stage == "synthesis"


def test_failed_counter_search_cannot_verify_candidate() -> None:
    failed_counter = _research_result(
        status=EvidenceResearchStatus.FAILED,
        evidence=[],
        question="Counter query",
    )
    reasoning = FakeReasoningModel(draft=_draft())

    result = EvidenceGroundedGapAnalyzer(
        reasoning,
        FakeEvidenceResearcher([failed_counter]),
    ).analyze(GapAnalysisRequest(evidence_result=_research_result()))

    assert result.status == GapAnalysisStatus.PARTIAL
    assert result.verified_gaps[0].status == (
        GapVerificationStatus.INSUFFICIENT_EVIDENCE
    )
    assert reasoning.assessment_calls == []
    assert result.model_traces[-1].error_type == "counter_research_failed"


def test_unknown_counter_evidence_reference_is_downgraded() -> None:
    counter = _research_result(
        evidence=[_evidence("evidence-1", doc_id="doc-counter")],
        question="Counter query",
    )
    reasoning = FakeReasoningModel(
        draft=_draft(),
        assessments=[
            GapAssessmentDraft(
                status=GapVerificationStatus.REJECTED,
                rationale="A supposed counterexample exists.",
                counter_evidence_ids=["evidence-404"],
            )
        ],
    )

    result = EvidenceGroundedGapAnalyzer(
        reasoning,
        FakeEvidenceResearcher([counter]),
    ).analyze(GapAnalysisRequest(evidence_result=_research_result()))

    assert result.status == GapAnalysisStatus.PARTIAL
    assert result.verified_gaps[0].status == (
        GapVerificationStatus.INSUFFICIENT_EVIDENCE
    )
    assert result.verified_gaps[0].counter_evidence_ids == []
    assert result.model_traces[-1].error_type == "grounding_policy_error"


def test_partial_counter_search_cannot_establish_verified_status() -> None:
    counter = _research_result(
        status=EvidenceResearchStatus.PARTIAL,
        evidence=[_evidence("evidence-1", doc_id="doc-counter")],
        question="Counter query",
    )
    reasoning = FakeReasoningModel(
        draft=_draft(),
        assessments=[
            GapAssessmentDraft(
                status=GapVerificationStatus.VERIFIED,
                rationale="No disproof was identified.",
            )
        ],
    )

    result = EvidenceGroundedGapAnalyzer(
        reasoning,
        FakeEvidenceResearcher([counter]),
    ).analyze(GapAnalysisRequest(evidence_result=_research_result()))

    assert result.status == GapAnalysisStatus.PARTIAL
    assert result.verified_gaps[0].status == (
        GapVerificationStatus.INSUFFICIENT_EVIDENCE
    )
    assert "partial counter-search" in result.warnings[-1]


def test_completed_empty_counter_search_can_return_verified() -> None:
    counter = _research_result(
        status=EvidenceResearchStatus.NO_EVIDENCE,
        evidence=[],
        question="Counter query",
    )
    reasoning = FakeReasoningModel(
        draft=_draft(),
        assessments=[
            GapAssessmentDraft(
                status=GapVerificationStatus.VERIFIED,
                rationale=(
                    "The bounded counter-search found no evidence closing the gap."
                ),
            )
        ],
    )

    result = EvidenceGroundedGapAnalyzer(
        reasoning,
        FakeEvidenceResearcher([counter]),
    ).analyze(GapAnalysisRequest(evidence_result=_research_result()))

    assert result.status == GapAnalysisStatus.COMPLETE
    assert result.verified_gaps[0].status == GapVerificationStatus.VERIFIED


def test_refined_assessment_requires_refined_scope() -> None:
    with pytest.raises(ValidationError):
        GapAssessmentDraft(
            status=GapVerificationStatus.REFINED,
            rationale="The candidate needs a narrower scope.",
            counter_evidence_ids=["evidence-1"],
        )


def test_analysis_limits_reject_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        GapAnalysisLimits(max_gap_candidates=11)
