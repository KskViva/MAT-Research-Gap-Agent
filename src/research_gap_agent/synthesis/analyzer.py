"""Evidence-grounded claim synthesis and counter-search verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from research_gap_agent.agent import DeepSeekChatModel
from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    ResearchRequest,
    SciVerseEvidenceResearcher,
    build_sciverse_evidence_researcher,
)

from .deepseek import DeepSeekGapReasoningModel, GapReasoningModel
from .models import (
    Claim,
    ClaimProposal,
    CounterResearchRecord,
    GapAnalysisRequest,
    GapAnalysisResult,
    GapAnalysisStatus,
    GapAssessmentDraft,
    GapCandidate,
    GapCandidateProposal,
    GapVerificationStatus,
    ModelReasoningTrace,
    SynthesisDraft,
    VerifiedGap,
)


class EvidenceResearcher(Protocol):
    def research(
        self,
        request: ResearchRequest | dict[str, Any],
    ) -> EvidenceResearchResult: ...


class EvidenceGroundedGapAnalyzer:
    """Create grounded claims, then try to falsify every gap candidate."""

    def __init__(
        self,
        reasoning_model: GapReasoningModel,
        evidence_researcher: EvidenceResearcher,
    ) -> None:
        self._reasoning_model = reasoning_model
        self._evidence_researcher = evidence_researcher

    def analyze(
        self,
        request: GapAnalysisRequest | dict[str, Any],
    ) -> GapAnalysisResult:
        validated = GapAnalysisRequest.model_validate(request)
        source_result = validated.evidence_result
        source_evidence = source_result.evidence[
            : validated.limits.max_input_evidence
        ]
        question = source_result.request.question
        traces: list[ModelReasoningTrace] = []
        warnings: list[str] = []

        if not source_evidence:
            warnings.append(
                "Module 2 supplied no evidence; synthesis and gap verification "
                "were not attempted"
            )
            return self._result(
                GapAnalysisStatus.INSUFFICIENT_EVIDENCE,
                question,
                source_evidence,
                warnings=warnings,
            )

        try:
            draft = self._reasoning_model.synthesize(
                question,
                source_evidence,
                max_claims=validated.limits.max_claims,
                max_gap_candidates=validated.limits.max_gap_candidates,
            )
        except Exception as exc:
            traces.append(_failed_trace(1, "synthesis", None, exc))
            warnings.append("Structured evidence synthesis failed")
            return self._result(
                GapAnalysisStatus.FAILED,
                question,
                source_evidence,
                traces=traces,
                warnings=warnings,
            )

        traces.append(
            ModelReasoningTrace(sequence=1, stage="synthesis", ok=True)
        )
        claims, claims_by_proposal, claim_warnings = _materialize_claims(
            draft,
            source_evidence,
            validated.limits.max_claims,
        )
        warnings.extend(claim_warnings)
        candidates, candidate_warnings = _materialize_candidates(
            draft,
            claims_by_proposal,
            validated.limits.max_gap_candidates,
        )
        warnings.extend(candidate_warnings)

        if not claims or not candidates:
            warnings.append(
                "No fully grounded claim-and-gap candidate set was produced"
            )
            return self._result(
                GapAnalysisStatus.INSUFFICIENT_EVIDENCE,
                question,
                source_evidence,
                claims=claims,
                candidates=candidates,
                traces=traces,
                warnings=warnings,
            )

        counter_records: list[CounterResearchRecord] = []
        verified_gaps: list[VerifiedGap] = []
        partial = source_result.status == EvidenceResearchStatus.PARTIAL

        for candidate in candidates:
            counter_request = ResearchRequest(
                question=candidate.counter_query,
                limits=validated.limits.counter_research,
                semantic_mode=validated.limits.counter_semantic_mode,
                strict_paper_scope=False,
            )
            try:
                counter_result = self._evidence_researcher.research(
                    counter_request
                )
            except Exception:
                counter_result = EvidenceResearchResult(
                    status=EvidenceResearchStatus.FAILED,
                    request=counter_request,
                    warnings=["Counter-evidence research could not be completed"],
                )
            counter_records.append(
                CounterResearchRecord(
                    gap_id=candidate.gap_id,
                    query=candidate.counter_query,
                    result=counter_result,
                )
            )

            supporting_claims = _candidate_claims(candidate, claims)
            supporting_evidence_ids = _supporting_evidence_ids(
                supporting_claims
            )
            trace_sequence = len(traces) + 1
            if counter_result.status == EvidenceResearchStatus.FAILED:
                partial = True
                traces.append(
                    ModelReasoningTrace(
                        sequence=trace_sequence,
                        stage="verification",
                        gap_id=candidate.gap_id,
                        ok=False,
                        error_type="counter_research_failed",
                        error_message=(
                            "Counter-evidence research failed before assessment"
                        ),
                    )
                )
                verified_gaps.append(
                    _insufficient_gap(
                        candidate,
                        supporting_evidence_ids,
                        "Counter-evidence retrieval failed, so the candidate "
                        "cannot be verified.",
                    )
                )
                continue

            try:
                assessment = self._reasoning_model.assess_gap(
                    candidate,
                    supporting_claims,
                    counter_result,
                )
            except Exception as exc:
                partial = True
                traces.append(
                    _failed_trace(
                        trace_sequence,
                        "verification",
                        candidate.gap_id,
                        exc,
                    )
                )
                verified_gaps.append(
                    _insufficient_gap(
                        candidate,
                        supporting_evidence_ids,
                        "Structured counter-evidence assessment failed.",
                    )
                )
                continue

            verified, policy_error = _enforce_assessment(
                candidate,
                supporting_evidence_ids,
                counter_result,
                assessment,
            )
            if policy_error is None:
                traces.append(
                    ModelReasoningTrace(
                        sequence=trace_sequence,
                        stage="verification",
                        gap_id=candidate.gap_id,
                        ok=True,
                    )
                )
            else:
                partial = True
                warnings.append(f"{candidate.gap_id}: {policy_error}")
                traces.append(
                    ModelReasoningTrace(
                        sequence=trace_sequence,
                        stage="verification",
                        gap_id=candidate.gap_id,
                        ok=False,
                        error_type="grounding_policy_error",
                        error_message=policy_error,
                    )
                )
            if counter_result.status == EvidenceResearchStatus.PARTIAL:
                partial = True
            verified_gaps.append(verified)

        return self._result(
            GapAnalysisStatus.PARTIAL if partial else GapAnalysisStatus.COMPLETE,
            question,
            source_evidence,
            claims=claims,
            candidates=candidates,
            verified_gaps=verified_gaps,
            counter_records=counter_records,
            traces=traces,
            warnings=warnings,
        )

    def close(self) -> None:
        close = getattr(self._reasoning_model, "close", None)
        if callable(close):
            close()

    def _result(
        self,
        status: GapAnalysisStatus,
        question: str,
        source_evidence: list[EvidenceRef],
        *,
        claims: list[Claim] | None = None,
        candidates: list[GapCandidate] | None = None,
        verified_gaps: list[VerifiedGap] | None = None,
        counter_records: list[CounterResearchRecord] | None = None,
        traces: list[ModelReasoningTrace] | None = None,
        warnings: list[str] | None = None,
    ) -> GapAnalysisResult:
        return GapAnalysisResult(
            status=status,
            question=question,
            model=self._reasoning_model.model_name,
            source_evidence=source_evidence,
            claims=claims or [],
            gap_candidates=candidates or [],
            verified_gaps=verified_gaps or [],
            counter_research=counter_records or [],
            model_traces=traces or [],
            warnings=warnings or [],
        )


def build_gap_analyzer(
    *,
    env_file: str | Path = ".env",
    server_command: str | Path | None = None,
) -> EvidenceGroundedGapAnalyzer:
    """Build Module 3 from the existing DeepSeek and SciVerse components."""

    chat_model = DeepSeekChatModel.from_env(env_file)
    reasoning_model = DeepSeekGapReasoningModel(chat_model)
    evidence_researcher: SciVerseEvidenceResearcher = (
        build_sciverse_evidence_researcher(
            env_file=env_file,
            server_command=server_command,
        )
    )
    return EvidenceGroundedGapAnalyzer(reasoning_model, evidence_researcher)


def _materialize_claims(
    draft: SynthesisDraft,
    evidence: list[EvidenceRef],
    limit: int,
) -> tuple[list[Claim], dict[int, Claim], list[str]]:
    known_evidence = {item.evidence_id for item in evidence}
    claims: list[Claim] = []
    by_proposal: dict[int, Claim] = {}
    warnings: list[str] = []
    for proposal_number, proposal in enumerate(draft.claims[:limit], start=1):
        unknown = [
            item for item in proposal.evidence_ids if item not in known_evidence
        ]
        if unknown:
            warnings.append(
                f"Claim proposal {proposal_number} referenced unknown evidence "
                "and was rejected"
            )
            continue
        claim = Claim(
            claim_id=f"claim-{len(claims) + 1}",
            statement=proposal.statement,
            evidence_ids=proposal.evidence_ids,
        )
        claims.append(claim)
        by_proposal[proposal_number] = claim
    return claims, by_proposal, warnings


def _materialize_candidates(
    draft: SynthesisDraft,
    claims_by_proposal: dict[int, Claim],
    limit: int,
) -> tuple[list[GapCandidate], list[str]]:
    candidates: list[GapCandidate] = []
    warnings: list[str] = []
    for proposal_number, proposal in enumerate(
        draft.gap_candidates[:limit],
        start=1,
    ):
        supporting = [
            claims_by_proposal.get(number)
            for number in proposal.supporting_claim_numbers
        ]
        if any(item is None for item in supporting):
            warnings.append(
                f"Gap proposal {proposal_number} referenced an unavailable claim "
                "and was rejected"
            )
            continue
        candidates.append(
            GapCandidate(
                gap_id=f"gap-{len(candidates) + 1}",
                statement=proposal.statement,
                rationale=proposal.rationale,
                category=proposal.category,
                supporting_claim_ids=[
                    item.claim_id for item in supporting if item is not None
                ],
                uncertainty=proposal.uncertainty,
                counter_query=proposal.counter_query,
            )
        )
    return candidates, warnings


def _candidate_claims(
    candidate: GapCandidate,
    claims: list[Claim],
) -> list[Claim]:
    by_id = {item.claim_id: item for item in claims}
    return [by_id[item] for item in candidate.supporting_claim_ids]


def _supporting_evidence_ids(claims: list[Claim]) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_id
            for claim in claims
            for evidence_id in claim.evidence_ids
        )
    )


def _enforce_assessment(
    candidate: GapCandidate,
    supporting_evidence_ids: list[str],
    counter_result: EvidenceResearchResult,
    assessment: GapAssessmentDraft,
) -> tuple[VerifiedGap, str | None]:
    available_counter_ids = {
        item.evidence_id for item in counter_result.evidence
    }
    unknown = [
        item
        for item in assessment.counter_evidence_ids
        if item not in available_counter_ids
    ]
    if unknown:
        message = "Assessment referenced counter-evidence not returned by SciVerse"
        return (
            _insufficient_gap(candidate, supporting_evidence_ids, message),
            message,
        )
    if (
        assessment.status
        in {GapVerificationStatus.REJECTED, GapVerificationStatus.REFINED}
        and not assessment.counter_evidence_ids
    ):
        message = (
            "Rejected/refined status requires at least one grounded "
            "counter-evidence reference"
        )
        return (
            _insufficient_gap(candidate, supporting_evidence_ids, message),
            message,
        )
    if (
        assessment.status == GapVerificationStatus.VERIFIED
        and counter_result.status == EvidenceResearchStatus.PARTIAL
    ):
        message = "A partial counter-search cannot establish verified status"
        return (
            _insufficient_gap(candidate, supporting_evidence_ids, message),
            message,
        )
    return (
        VerifiedGap(
            gap_id=candidate.gap_id,
            status=assessment.status,
            rationale=assessment.rationale,
            supporting_claim_ids=candidate.supporting_claim_ids,
            supporting_evidence_ids=supporting_evidence_ids,
            counter_evidence_ids=assessment.counter_evidence_ids,
            refined_scope=assessment.refined_scope,
        ),
        None,
    )


def _insufficient_gap(
    candidate: GapCandidate,
    supporting_evidence_ids: list[str],
    rationale: str,
) -> VerifiedGap:
    return VerifiedGap(
        gap_id=candidate.gap_id,
        status=GapVerificationStatus.INSUFFICIENT_EVIDENCE,
        rationale=rationale,
        supporting_claim_ids=candidate.supporting_claim_ids,
        supporting_evidence_ids=supporting_evidence_ids,
    )


def _failed_trace(
    sequence: int,
    stage: str,
    gap_id: str | None,
    exc: Exception,
) -> ModelReasoningTrace:
    return ModelReasoningTrace(
        sequence=sequence,
        stage=stage,
        gap_id=gap_id,
        ok=False,
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
    )
