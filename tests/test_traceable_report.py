from typing import Any

import pytest
from pydantic import ValidationError

from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    EvidenceToolTrace,
    ResearchLimits,
    ResearchRequest,
)
from research_gap_agent.reporting import (
    ReportConclusionKind,
    ReportRequest,
    ReportStatus,
    build_traceable_report,
)
from research_gap_agent.synthesis import (
    Claim,
    CounterResearchRecord,
    GapAnalysisResult,
    GapAnalysisStatus,
    GapCandidate,
    GapCategory,
    GapVerificationStatus,
    VerifiedGap,
)


def _evidence(
    evidence_id: str,
    *,
    doc_id: str = "doc-source",
    text: str = "Observed source evidence.",
    call_id: str = "source-call",
) -> EvidenceRef:
    rank = int(evidence_id.rsplit("-", 1)[-1])
    return EvidenceRef(
        evidence_id=evidence_id,
        source_tool_call_id=call_id,
        rank=rank,
        doc_id=doc_id,
        chunk_id=f"chunk-{rank}",
        offset=rank * 100,
        title="Evidence paper",
        publication_year=2025,
        score=0.91,
        quoted_text=text,
        raw_chunk={"not_for_report": True},
    )


def _counter_result(
    query: str,
    *,
    evidence: list[EvidenceRef] | None = None,
    status: EvidenceResearchStatus = EvidenceResearchStatus.COMPLETE,
) -> EvidenceResearchResult:
    return EvidenceResearchResult(
        status=status,
        request=ResearchRequest(
            question=query,
            limits=ResearchLimits(context_expansions=0),
        ),
        evidence=evidence or [],
        tool_traces=[
            EvidenceToolTrace(
                sequence=1,
                call_id="counter-call-001",
                tool_name="semantic_search",
                arguments={"query": query},
                ok=True,
                result_count=len(evidence or []),
            )
        ],
    )


def _candidate(
    gap_id: str,
    claim_id: str = "claim-1",
) -> GapCandidate:
    return GapCandidate(
        gap_id=gap_id,
        statement=f"Candidate statement for {gap_id}.",
        rationale=f"Candidate rationale for {gap_id}.",
        category=GapCategory.MISSING_CONDITION,
        supporting_claim_ids=[claim_id],
        uncertainty=f"Retrieval uncertainty for {gap_id}.",
        counter_query=f"Counter query for {gap_id}",
    )


def _complete_analysis() -> GapAnalysisResult:
    candidate = _candidate("gap-1")
    return GapAnalysisResult(
        status=GapAnalysisStatus.COMPLETE,
        question="What remains uncertain?",
        model="fake-deepseek",
        source_evidence=[_evidence("evidence-1")],
        claims=[
            Claim(
                claim_id="claim-1",
                statement="A grounded observation was reported.",
                evidence_ids=["evidence-1"],
            )
        ],
        gap_candidates=[candidate],
        verified_gaps=[
            VerifiedGap(
                gap_id="gap-1",
                status=GapVerificationStatus.VERIFIED,
                rationale="The bounded counter-search found no disproof.",
                supporting_claim_ids=["claim-1"],
                supporting_evidence_ids=["evidence-1"],
            )
        ],
        counter_research=[
            CounterResearchRecord(
                gap_id="gap-1",
                query=candidate.counter_query,
                result=_counter_result(
                    candidate.counter_query,
                    status=EvidenceResearchStatus.NO_EVIDENCE,
                ),
            )
        ],
    )


def test_report_resolves_claim_and_verified_gap_to_sciverse_evidence() -> None:
    report = build_traceable_report(
        ReportRequest(
            analysis_result=_complete_analysis(),
            title="LLZO Traceability Report",
        )
    )

    assert report.status == ReportStatus.COMPLETE
    assert [item.kind for item in report.conclusions] == [
        ReportConclusionKind.CLAIM,
        ReportConclusionKind.VERIFIED_GAP,
    ]
    claim = report.conclusions[0]
    gap = report.conclusions[1]
    assert claim.evidence_references == ["source:evidence-1"]
    assert gap.claim_ids == ["claim-1"]
    assert gap.evidence_references == ["source:evidence-1"]
    assert report.claim_register[0].claim_id == "claim-1"
    assert report.gap_candidate_register[0].gap_id == "gap-1"
    assert report.gap_verification_register[0].gap_id == "gap-1"
    assert report.evidence_register[0].doc_id == "doc-source"
    assert report.evidence_register[0].source_tool_call_id == "source-call"
    assert report.counter_searches[0].tool_traces[0].call_id == (
        "counter-call-001"
    )
    assert "Retrieval uncertainty for gap-1." in report.markdown
    assert "doc-source" in report.markdown
    assert '"reference_id":"source:evidence-1"' in report.model_dump_json()


def test_counter_evidence_ids_are_qualified_by_gap_and_do_not_collide() -> None:
    source = _evidence("evidence-1", doc_id="doc-source")
    counter_one = _evidence(
        "evidence-1",
        doc_id="doc-counter-1",
        call_id="counter-source-1",
    )
    counter_two = _evidence(
        "evidence-1",
        doc_id="doc-counter-2",
        call_id="counter-source-2",
    )
    gap_one = _candidate("gap-1")
    gap_two = _candidate("gap-2")
    analysis = GapAnalysisResult(
        status=GapAnalysisStatus.COMPLETE,
        question="Which candidates survive?",
        model="fake-deepseek",
        source_evidence=[source],
        claims=[
            Claim(
                claim_id="claim-1",
                statement="One grounded claim.",
                evidence_ids=["evidence-1"],
            )
        ],
        gap_candidates=[gap_one, gap_two],
        verified_gaps=[
            VerifiedGap(
                gap_id="gap-1",
                status=GapVerificationStatus.REFINED,
                rationale="Counter-evidence narrows the scope.",
                supporting_claim_ids=["claim-1"],
                supporting_evidence_ids=["evidence-1"],
                counter_evidence_ids=["evidence-1"],
                refined_scope="Refined condition-specific gap.",
            ),
            VerifiedGap(
                gap_id="gap-2",
                status=GapVerificationStatus.REJECTED,
                rationale="Counter-evidence closes the candidate.",
                supporting_claim_ids=["claim-1"],
                supporting_evidence_ids=["evidence-1"],
                counter_evidence_ids=["evidence-1"],
            ),
        ],
        counter_research=[
            CounterResearchRecord(
                gap_id="gap-1",
                query=gap_one.counter_query,
                result=_counter_result(
                    gap_one.counter_query,
                    evidence=[counter_one],
                ),
            ),
            CounterResearchRecord(
                gap_id="gap-2",
                query=gap_two.counter_query,
                result=_counter_result(
                    gap_two.counter_query,
                    evidence=[counter_two],
                ),
            ),
        ],
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.COMPLETE
    assert report.conclusions[1].counter_evidence_references == [
        "counter:gap-1:evidence-1"
    ]
    assert report.conclusions[2].counter_evidence_references == [
        "counter:gap-2:evidence-1"
    ]
    assert {item.reference_id for item in report.evidence_register} == {
        "source:evidence-1",
        "counter:gap-1:evidence-1",
        "counter:gap-2:evidence-1",
    }


def test_insufficient_gap_is_a_limitation_not_a_report_conclusion() -> None:
    analysis = _complete_analysis()
    analysis.verified_gaps[0] = VerifiedGap(
        gap_id="gap-1",
        status=GapVerificationStatus.INSUFFICIENT_EVIDENCE,
        rationale="Counter-search was incomplete.",
        supporting_claim_ids=["claim-1"],
        supporting_evidence_ids=["evidence-1"],
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.PARTIAL
    assert [item.kind for item in report.conclusions] == [
        ReportConclusionKind.CLAIM
    ]
    assert any("gap-1" in item for item in report.limitations)
    assert "Retrieval uncertainty for gap-1." in report.limitations[0]


def test_broken_claim_evidence_is_excluded_and_reported_as_integrity_issue() -> None:
    analysis = _complete_analysis()
    analysis.claims[0] = Claim(
        claim_id="claim-1",
        statement="This statement must not enter the report.",
        evidence_ids=["evidence-999"],
    )
    analysis.verified_gaps[0] = analysis.verified_gaps[0].model_copy(
        update={"supporting_evidence_ids": ["evidence-999"]}
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.INVALID_TRACEABILITY
    assert report.conclusions == []
    assert report.integrity_issues[0].code == "unresolved_claim_evidence"
    assert "This statement must not enter the report." not in report.markdown


def test_unknown_counter_evidence_excludes_gap_but_keeps_grounded_claim() -> None:
    analysis = _complete_analysis()
    analysis.verified_gaps[0] = VerifiedGap(
        gap_id="gap-1",
        status=GapVerificationStatus.REJECTED,
        rationale="Unresolvable counter reference.",
        supporting_claim_ids=["claim-1"],
        supporting_evidence_ids=["evidence-1"],
        counter_evidence_ids=["evidence-404"],
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.INVALID_TRACEABILITY
    assert [item.kind for item in report.conclusions] == [
        ReportConclusionKind.CLAIM
    ]
    assert report.integrity_issues[-1].code == "unresolved_counter_evidence"


def test_duplicate_source_evidence_id_is_ambiguous_and_not_reported() -> None:
    analysis = _complete_analysis()
    analysis.source_evidence.append(
        _evidence("evidence-1", doc_id="different-doc")
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.INVALID_TRACEABILITY
    assert report.conclusions == []
    assert report.evidence_register == []
    assert any(
        issue.code == "duplicate_identifier"
        for issue in report.integrity_issues
    )


def test_refined_gap_without_scope_is_excluded_by_report_integrity_check() -> None:
    analysis = _complete_analysis()
    counter = _evidence(
        "evidence-1",
        doc_id="counter-doc",
        call_id="counter-source",
    )
    analysis.counter_research[0] = CounterResearchRecord(
        gap_id="gap-1",
        query=analysis.gap_candidates[0].counter_query,
        result=_counter_result(
            analysis.gap_candidates[0].counter_query,
            evidence=[counter],
        ),
    )
    analysis.verified_gaps[0] = VerifiedGap(
        gap_id="gap-1",
        status=GapVerificationStatus.REFINED,
        rationale="The candidate should be narrower.",
        supporting_claim_ids=["claim-1"],
        supporting_evidence_ids=["evidence-1"],
        counter_evidence_ids=["evidence-1"],
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.INVALID_TRACEABILITY
    assert [item.kind for item in report.conclusions] == [
        ReportConclusionKind.CLAIM
    ]
    assert report.integrity_issues[-1].code == "missing_refined_scope"


def test_evidence_excerpt_is_bounded_or_can_be_disabled() -> None:
    analysis = _complete_analysis()
    analysis.source_evidence[0] = _evidence(
        "evidence-1",
        text="x" * 300,
    )

    bounded = build_traceable_report(
        ReportRequest(
            analysis_result=analysis,
            max_quote_characters=100,
        )
    )
    disabled = build_traceable_report(
        ReportRequest(
            analysis_result=analysis,
            include_evidence_excerpts=False,
        )
    )

    assert len(bounded.evidence_register[0].excerpt or "") == 100
    assert bounded.evidence_register[0].excerpt_truncated is True
    assert "[truncated]" in bounded.markdown
    assert disabled.evidence_register[0].excerpt is None
    assert "> " not in disabled.markdown


def test_failed_empty_analysis_produces_report_without_conclusions() -> None:
    analysis = GapAnalysisResult(
        status=GapAnalysisStatus.FAILED,
        question="Why did analysis fail?",
        model="fake-deepseek",
        warnings=["Structured synthesis failed."],
    )

    report = build_traceable_report(ReportRequest(analysis_result=analysis))

    assert report.status == ReportStatus.NO_CONCLUSIONS
    assert report.conclusions == []
    assert "No conclusion passed" in report.markdown
    assert "Structured synthesis failed." in report.limitations


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "  "},
        {"max_quote_characters": 99},
        {"max_quote_characters": 4001},
    ],
)
def test_report_request_rejects_invalid_presentation_bounds(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ReportRequest(analysis_result=_complete_analysis(), **payload)
