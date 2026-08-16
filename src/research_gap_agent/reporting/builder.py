"""Deterministic traceability validation and Markdown report rendering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from research_gap_agent.evidence import EvidenceRef
from research_gap_agent.synthesis import (
    Claim,
    CounterResearchRecord,
    GapAnalysisStatus,
    GapCandidate,
    GapVerificationStatus,
    VerifiedGap,
)

from .models import (
    CounterSearchAudit,
    EvidenceCitation,
    EvidenceOrigin,
    ReportConclusion,
    ReportConclusionKind,
    ReportRequest,
    ReportStatus,
    TraceabilityIssue,
    TraceableReport,
)

IndexedModel = TypeVar("IndexedModel", bound=BaseModel)


class TraceableReportBuilder:
    """Build reports without adding LLM-generated narrative or source facts."""

    def build(
        self,
        request: ReportRequest | dict[str, Any],
    ) -> TraceableReport:
        validated = ReportRequest.model_validate(request)
        analysis = validated.analysis_result
        issues: list[TraceabilityIssue] = []
        limitations = list(analysis.warnings)

        source_evidence, ambiguous_source_ids = _unique_index(
            analysis.source_evidence,
            "evidence_id",
            "source_evidence",
            issues,
        )
        claims, ambiguous_claim_ids = _unique_index(
            analysis.claims,
            "claim_id",
            "claims",
            issues,
        )
        candidates, ambiguous_gap_ids = _unique_index(
            analysis.gap_candidates,
            "gap_id",
            "gap_candidates",
            issues,
        )
        verified_gaps, ambiguous_verified_gap_ids = _unique_index(
            analysis.verified_gaps,
            "gap_id",
            "verified_gaps",
            issues,
        )
        counter_records, ambiguous_counter_gap_ids = _unique_index(
            analysis.counter_research,
            "gap_id",
            "counter_research",
            issues,
        )
        ambiguous_gaps = (
            ambiguous_gap_ids
            | ambiguous_verified_gap_ids
            | ambiguous_counter_gap_ids
        )

        citations: dict[str, EvidenceCitation] = {}
        for evidence_id, evidence in source_evidence.items():
            if evidence_id in ambiguous_source_ids:
                continue
            citation = _citation(
                evidence,
                EvidenceOrigin.SOURCE,
                validated,
            )
            citations[citation.reference_id] = citation

        counter_evidence: dict[str, dict[str, EvidenceRef]] = {}
        counter_audits: list[CounterSearchAudit] = []
        for gap_id, record in counter_records.items():
            if gap_id in ambiguous_gaps:
                continue
            indexed, ambiguous_ids = _unique_index(
                record.result.evidence,
                "evidence_id",
                f"counter_research[{gap_id}].evidence",
                issues,
            )
            usable = {
                evidence_id: evidence
                for evidence_id, evidence in indexed.items()
                if evidence_id not in ambiguous_ids
            }
            counter_evidence[gap_id] = usable
            references: list[str] = []
            for evidence in usable.values():
                citation = _citation(
                    evidence,
                    EvidenceOrigin.COUNTER,
                    validated,
                    gap_id=gap_id,
                )
                citations[citation.reference_id] = citation
                references.append(citation.reference_id)
            counter_audits.append(
                CounterSearchAudit(
                    gap_id=gap_id,
                    query=record.query,
                    status=record.result.status,
                    evidence_references=references,
                    tool_traces=record.result.tool_traces,
                    warnings=record.result.warnings,
                )
            )

        conclusions: list[ReportConclusion] = []
        valid_claim_ids: set[str] = set()
        for claim_id, claim in claims.items():
            if claim_id in ambiguous_claim_ids:
                continue
            references = _source_references(
                claim.evidence_ids,
                source_evidence,
                ambiguous_source_ids,
            )
            if references is None:
                issues.append(
                    _issue(
                        "unresolved_claim_evidence",
                        f"claims[{claim_id}].evidence_ids",
                        "Claim references missing or ambiguous source evidence",
                    )
                )
                continue
            conclusions.append(
                ReportConclusion(
                    conclusion_id=f"conclusion-{len(conclusions) + 1}",
                    kind=ReportConclusionKind.CLAIM,
                    text=claim.statement,
                    claim_ids=[claim_id],
                    evidence_references=references,
                )
            )
            valid_claim_ids.add(claim_id)

        for gap_id, verified in verified_gaps.items():
            if gap_id in ambiguous_gaps:
                continue
            candidate = candidates.get(gap_id)
            counter_record = counter_records.get(gap_id)
            if candidate is None or counter_record is None:
                issues.append(
                    _issue(
                        "unresolved_gap_chain",
                        f"verified_gaps[{gap_id}]",
                        "Verified gap requires both its candidate and counter-search",
                    )
                )
                continue
            if verified.status == GapVerificationStatus.INSUFFICIENT_EVIDENCE:
                limitations.append(
                    _gap_limitation(candidate, verified.rationale)
                )
                continue
            if any(
                claim_id not in valid_claim_ids
                for claim_id in verified.supporting_claim_ids
            ):
                issues.append(
                    _issue(
                        "unresolved_gap_claim",
                        f"verified_gaps[{gap_id}].supporting_claim_ids",
                        "Gap references a Claim that is absent or not reportable",
                    )
                )
                continue
            if set(candidate.supporting_claim_ids) != set(
                verified.supporting_claim_ids
            ):
                issues.append(
                    _issue(
                        "gap_claim_mismatch",
                        f"verified_gaps[{gap_id}].supporting_claim_ids",
                        "Verified gap and candidate have different supporting Claims",
                    )
                )
                continue
            expected_evidence = _claim_evidence_ids(
                verified.supporting_claim_ids,
                claims,
            )
            if set(expected_evidence) != set(verified.supporting_evidence_ids):
                issues.append(
                    _issue(
                        "gap_evidence_mismatch",
                        f"verified_gaps[{gap_id}].supporting_evidence_ids",
                        "Verified gap evidence does not match its supporting Claims",
                    )
                )
                continue
            source_references = _source_references(
                verified.supporting_evidence_ids,
                source_evidence,
                ambiguous_source_ids,
            )
            if source_references is None:
                issues.append(
                    _issue(
                        "unresolved_gap_evidence",
                        f"verified_gaps[{gap_id}].supporting_evidence_ids",
                        "Verified gap references missing source evidence",
                    )
                )
                continue
            counter_references = _counter_references(
                gap_id,
                verified.counter_evidence_ids,
                counter_evidence,
            )
            if counter_references is None:
                issues.append(
                    _issue(
                        "unresolved_counter_evidence",
                        f"verified_gaps[{gap_id}].counter_evidence_ids",
                        "Gap references counter-evidence absent from its counter-search",
                    )
                )
                continue
            if (
                verified.status
                in {
                    GapVerificationStatus.REJECTED,
                    GapVerificationStatus.REFINED,
                }
                and not counter_references
            ):
                issues.append(
                    _issue(
                        "missing_counter_evidence",
                        f"verified_gaps[{gap_id}].counter_evidence_ids",
                        "Rejected/refined conclusions require counter-evidence",
                    )
                )
                continue
            kind, text = _gap_conclusion_kind_and_text(candidate, verified)
            if kind is None or text is None:
                issues.append(
                    _issue(
                        "missing_refined_scope",
                        f"verified_gaps[{gap_id}].refined_scope",
                        "Refined gap conclusion requires a refined scope",
                    )
                )
                continue
            if (
                kind == ReportConclusionKind.REJECTED_CANDIDATE
                and not validated.include_rejected_candidates
            ):
                continue
            conclusions.append(
                ReportConclusion(
                    conclusion_id=f"conclusion-{len(conclusions) + 1}",
                    kind=kind,
                    text=text,
                    status=verified.status,
                    rationale=verified.rationale,
                    uncertainty=candidate.uncertainty,
                    claim_ids=verified.supporting_claim_ids,
                    gap_id=gap_id,
                    evidence_references=source_references,
                    counter_evidence_references=counter_references,
                )
            )

        for gap_id, candidate in candidates.items():
            if gap_id not in verified_gaps and gap_id not in ambiguous_gaps:
                limitations.append(
                    _gap_limitation(
                        candidate,
                        "No verification result was produced for this candidate.",
                    )
                )

        if analysis.status != GapAnalysisStatus.COMPLETE:
            limitations.append(
                f"Module 3 analysis status was {analysis.status.value}."
            )
        limitations.append(
            "Verified means the candidate survived the recorded bounded "
            "counter-search; it is not a global absence-of-literature guarantee."
        )
        limitations = list(dict.fromkeys(limitations))

        status = _report_status(analysis.status, conclusions, issues, limitations)
        report = TraceableReport(
            title=validated.title,
            question=analysis.question,
            status=status,
            analysis_status=analysis.status,
            conclusions=conclusions,
            claim_register=[
                claim
                for claim_id, claim in claims.items()
                if claim_id in valid_claim_ids
            ],
            gap_candidate_register=[
                candidate
                for gap_id, candidate in candidates.items()
                if gap_id not in ambiguous_gaps
            ],
            gap_verification_register=[
                verified
                for gap_id, verified in verified_gaps.items()
                if gap_id not in ambiguous_gaps
            ],
            evidence_register=list(citations.values()),
            counter_searches=counter_audits,
            model_traces=analysis.model_traces,
            limitations=limitations,
            integrity_issues=issues,
            markdown="pending",
        )
        return report.model_copy(update={"markdown": _render_markdown(report)})


def build_traceable_report(
    request: ReportRequest | dict[str, Any],
) -> TraceableReport:
    """Convenience entry point for deterministic Module 4 generation."""

    return TraceableReportBuilder().build(request)


def _unique_index(
    values: Iterable[IndexedModel],
    field_name: str,
    path: str,
    issues: list[TraceabilityIssue],
) -> tuple[dict[str, IndexedModel], set[str]]:
    indexed: dict[str, IndexedModel] = {}
    ambiguous: set[str] = set()
    for item in values:
        identifier = str(getattr(item, field_name))
        if identifier in indexed:
            ambiguous.add(identifier)
            issues.append(
                _issue(
                    "duplicate_identifier",
                    f"{path}[{identifier}]",
                    "Identifier occurs more than once and is ambiguous",
                )
            )
        else:
            indexed[identifier] = item
    return indexed, ambiguous


def _citation(
    evidence: EvidenceRef,
    origin: EvidenceOrigin,
    request: ReportRequest,
    *,
    gap_id: str | None = None,
) -> EvidenceCitation:
    reference_id = (
        f"source:{evidence.evidence_id}"
        if origin == EvidenceOrigin.SOURCE
        else f"counter:{gap_id}:{evidence.evidence_id}"
    )
    excerpt: str | None = None
    truncated = False
    if request.include_evidence_excerpts:
        excerpt = evidence.quoted_text[: request.max_quote_characters]
        truncated = len(evidence.quoted_text) > request.max_quote_characters
    return EvidenceCitation(
        reference_id=reference_id,
        origin=origin,
        gap_id=gap_id,
        evidence_id=evidence.evidence_id,
        source_tool_call_id=evidence.source_tool_call_id,
        doc_id=evidence.doc_id,
        chunk_id=evidence.chunk_id,
        offset=evidence.offset,
        title=evidence.title,
        publication_year=evidence.publication_year,
        score=evidence.score,
        excerpt=excerpt,
        excerpt_truncated=truncated,
    )


def _source_references(
    evidence_ids: list[str],
    source_evidence: dict[str, EvidenceRef],
    ambiguous_ids: set[str],
) -> list[str] | None:
    if any(
        evidence_id not in source_evidence or evidence_id in ambiguous_ids
        for evidence_id in evidence_ids
    ):
        return None
    return [f"source:{evidence_id}" for evidence_id in evidence_ids]


def _counter_references(
    gap_id: str,
    evidence_ids: list[str],
    counter_evidence: dict[str, dict[str, EvidenceRef]],
) -> list[str] | None:
    available = counter_evidence.get(gap_id, {})
    if any(evidence_id not in available for evidence_id in evidence_ids):
        return None
    return [f"counter:{gap_id}:{evidence_id}" for evidence_id in evidence_ids]


def _claim_evidence_ids(
    claim_ids: list[str],
    claims: dict[str, Claim],
) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_id
            for claim_id in claim_ids
            for evidence_id in claims[claim_id].evidence_ids
        )
    )


def _gap_conclusion_kind_and_text(
    candidate: GapCandidate,
    verified: VerifiedGap,
) -> tuple[ReportConclusionKind | None, str | None]:
    if verified.status == GapVerificationStatus.VERIFIED:
        return ReportConclusionKind.VERIFIED_GAP, candidate.statement
    if verified.status == GapVerificationStatus.REFINED:
        return ReportConclusionKind.REFINED_GAP, verified.refined_scope
    if verified.status == GapVerificationStatus.REJECTED:
        return ReportConclusionKind.REJECTED_CANDIDATE, candidate.statement
    return None, None


def _gap_limitation(candidate: GapCandidate, rationale: str) -> str:
    return (
        f"{candidate.gap_id}: {rationale} Uncertainty retained: "
        f"{candidate.uncertainty}"
    )


def _report_status(
    analysis_status: GapAnalysisStatus,
    conclusions: list[ReportConclusion],
    issues: list[TraceabilityIssue],
    limitations: list[str],
) -> ReportStatus:
    if issues:
        return ReportStatus.INVALID_TRACEABILITY
    if not conclusions:
        return ReportStatus.NO_CONCLUSIONS
    if analysis_status != GapAnalysisStatus.COMPLETE or any(
        item.startswith("gap-") for item in limitations
    ):
        return ReportStatus.PARTIAL
    return ReportStatus.COMPLETE


def _issue(code: str, path: str, message: str) -> TraceabilityIssue:
    return TraceabilityIssue(code=code, path=path, message=message)


def _render_markdown(report: TraceableReport) -> str:
    lines = [
        f"# {_inline(report.title)}",
        "",
        "## Research Question",
        "",
        _inline(report.question),
        "",
        "## Status",
        "",
        f"Report: `{report.status.value}`; analysis: `{report.analysis_status.value}`.",
        "",
        "## Traceable Conclusions",
        "",
    ]
    if not report.conclusions:
        lines.append("No conclusion passed the report traceability checks.")
    for conclusion in report.conclusions:
        lines.extend(_render_conclusion(conclusion))

    lines.extend(["", "## Gap Verification Register", ""])
    if not report.gap_verification_register:
        lines.append("No gap verification was recorded.")
    for verification in report.gap_verification_register:
        candidate = next(
            (
                item
                for item in report.gap_candidate_register
                if item.gap_id == verification.gap_id
            ),
            None,
        )
        lines.extend(
            [
                f"### {verification.gap_id} — {verification.status.value}",
                "",
                (
                    _inline(candidate.statement)
                    if candidate is not None
                    else "Candidate unavailable due to traceability failure."
                ),
                "",
                f"- Rationale: {_inline(verification.rationale)}",
                (
                    f"- Uncertainty: {_inline(candidate.uncertainty)}"
                    if candidate is not None
                    else "- Uncertainty: candidate unavailable"
                ),
                "",
            ]
        )

    lines.extend(["", "## Evidence Register", ""])
    if not report.evidence_register:
        lines.append("No evidence reference is available.")
    for citation in report.evidence_register:
        lines.extend(_render_citation(citation))

    lines.extend(["", "## Counter-search Audit", ""])
    if not report.counter_searches:
        lines.append("No counter-search was recorded.")
    for audit in report.counter_searches:
        lines.extend(
            [
                f"### {_inline(audit.gap_id)}",
                "",
                f"- Query: {_inline(audit.query)}",
                f"- Status: `{audit.status.value}`",
                "- Evidence: " + _references(audit.evidence_references),
                "- Tool calls: "
                + (
                    ", ".join(
                        f"`{trace.call_id}` ({trace.tool_name}, ok={trace.ok})"
                        for trace in audit.tool_traces
                    )
                    or "none"
                ),
            ]
        )
        for warning in audit.warnings:
            lines.append(f"- Warning: {_inline(warning)}")
        lines.append("")

    lines.extend(["## Model Reasoning Audit", ""])
    if not report.model_traces:
        lines.append("No model reasoning trace was recorded.")
    for trace in report.model_traces:
        gap_suffix = f", gap={trace.gap_id}" if trace.gap_id else ""
        lines.append(
            f"- `{trace.stage}`: ok={trace.ok}{gap_suffix}"
            + (f", error={_inline(trace.error_type)}" if trace.error_type else "")
        )
    lines.append("")

    lines.extend(["## Limitations", ""])
    for limitation in report.limitations:
        lines.append(f"- {_inline(limitation)}")

    if report.integrity_issues:
        lines.extend(["", "## Integrity Issues", ""])
        for issue in report.integrity_issues:
            lines.append(
                f"- `{issue.code}` at `{_inline(issue.path)}`: "
                f"{_inline(issue.message)}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_conclusion(conclusion: ReportConclusion) -> list[str]:
    lines = [
        f"### {conclusion.conclusion_id} — {conclusion.kind.value}",
        "",
        _inline(conclusion.text),
        "",
    ]
    if conclusion.status is not None:
        lines.append(f"- Verification status: `{conclusion.status.value}`")
    if conclusion.gap_id is not None:
        lines.append(f"- Gap: `{conclusion.gap_id}`")
    lines.append("- Claims: " + _references(conclusion.claim_ids))
    lines.append("- Evidence: " + _references(conclusion.evidence_references))
    if conclusion.counter_evidence_references:
        lines.append(
            "- Counter-evidence: "
            + _references(conclusion.counter_evidence_references)
        )
    if conclusion.rationale is not None:
        lines.append(f"- Rationale: {_inline(conclusion.rationale)}")
    if conclusion.uncertainty is not None:
        lines.append(f"- Uncertainty: {_inline(conclusion.uncertainty)}")
    lines.append("")
    return lines


def _render_citation(citation: EvidenceCitation) -> list[str]:
    lines = [
        f"### {citation.reference_id}",
        "",
        f"- Origin: `{citation.origin.value}`",
        f"- doc_id: `{citation.doc_id}`",
        f"- chunk_id: `{citation.chunk_id or 'not returned'}`",
        f"- offset: `{citation.offset}`",
        f"- tool call: `{citation.source_tool_call_id}`",
        f"- title: {_inline(citation.title or 'not returned')}",
        f"- year: `{citation.publication_year or 'not returned'}`",
        f"- score: `{citation.score if citation.score is not None else 'not returned'}`",
    ]
    if citation.excerpt is not None:
        suffix = " … [truncated]" if citation.excerpt_truncated else ""
        lines.extend(["", _blockquote(citation.excerpt + suffix)])
    lines.append("")
    return lines


def _references(values: list[str]) -> str:
    return ", ".join(f"`{_inline(value)}`" for value in values) or "none"


def _inline(value: str) -> str:
    return " ".join(value.split()).replace("`", "'")


def _blockquote(value: str) -> str:
    return "\n".join(f"> {line}" for line in value.splitlines())
