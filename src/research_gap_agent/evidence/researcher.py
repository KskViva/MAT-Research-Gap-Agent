"""Deterministic SciVerse evidence-research orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from research_gap_agent.agent.models import PaperRecord
from research_gap_agent.tools import ToolCallResult, ToolRegistry
from research_gap_agent.tools.sciverse import build_sciverse_tool_registry

from .models import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    EvidenceToolTrace,
    OriginalTextExcerpt,
    PaperScope,
    ResearchRequest,
)


class SciVerseEvidenceResearcher:
    """Acquire paper candidates, semantic chunks, and bounded source context."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def research(
        self,
        request: ResearchRequest | dict[str, Any],
    ) -> EvidenceResearchResult:
        validated = ResearchRequest.model_validate(request)
        traces: list[EvidenceToolTrace] = []
        warnings: list[str] = []

        search_arguments = _search_arguments(validated)
        search_call_id, search_result = self._invoke(
            traces,
            "search_papers",
            search_arguments,
        )
        papers = _paper_records(search_call_id, search_result)
        if not search_result.ok:
            warnings.append(
                f"search_papers failed; see tool trace {search_call_id}"
            )

        doc_ids = _unique_strings(
            paper.doc_id for paper in papers if paper.doc_id is not None
        )
        missing_doc_ids = sum(paper.doc_id is None for paper in papers)
        if missing_doc_ids:
            warnings.append(
                f"{missing_doc_ids} paper candidate(s) have no doc_id and cannot "
                "be used for full-text retrieval"
            )

        semantic_arguments = _semantic_arguments(validated, doc_ids)
        semantic_call_id, semantic_result = self._invoke(
            traces,
            "semantic_search",
            semantic_arguments,
        )
        evidence, skipped = _evidence_refs(semantic_call_id, semantic_result)
        if skipped:
            warnings.append(
                f"{skipped} semantic hit(s) lacked a usable doc_id, offset, or text"
            )
        if not semantic_result.ok:
            warnings.append(
                f"semantic_search failed; see tool trace {semantic_call_id}"
            )

        expanded: list[EvidenceRef] = []
        for index, item in enumerate(evidence):
            if index >= validated.limits.context_expansions:
                expanded.append(item)
                continue
            read_arguments = {
                "doc_id": item.doc_id,
                "offset": item.offset,
                "limit": validated.limits.read_bytes,
            }
            read_call_id, read_result = self._invoke(
                traces,
                "read_content",
                read_arguments,
            )
            context = _original_text_excerpt(
                read_call_id,
                item.doc_id,
                item.offset,
                read_result,
            )
            if context is None:
                warnings.append(
                    f"read_content produced no usable context for {item.evidence_id}; "
                    f"see tool trace {read_call_id}"
                )
                expanded.append(item)
            else:
                expanded.append(item.model_copy(update={"context": context}))

        status = _status(search_result, semantic_result, traces, expanded)
        if status == EvidenceResearchStatus.NO_EVIDENCE:
            warnings.append("SciVerse returned no usable semantic evidence")

        return EvidenceResearchResult(
            status=status,
            request=validated,
            papers=papers,
            evidence=expanded,
            tool_traces=traces,
            warnings=warnings,
        )

    def _invoke(
        self,
        traces: list[EvidenceToolTrace],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, ToolCallResult]:
        sequence = len(traces) + 1
        call_id = f"evidence-call-{sequence:03d}"
        result = self._registry.invoke(tool_name, arguments)
        traces.append(
            EvidenceToolTrace(
                sequence=sequence,
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                ok=result.ok,
                error_type=result.error_type,
                error_message=result.error_message,
                result_count=_result_count(tool_name, result),
            )
        )
        return call_id, result


def build_sciverse_evidence_researcher(
    *,
    env_file: str | Path = ".env",
    server_command: str | Path | None = None,
) -> SciVerseEvidenceResearcher:
    """Build Module 2 on the validated Module 1 SciVerse registry."""

    return SciVerseEvidenceResearcher(
        build_sciverse_tool_registry(
            env_file=env_file,
            server_command=server_command,
        )
    )


def _search_arguments(request: ResearchRequest) -> dict[str, Any]:
    scope = request.scope or PaperScope()
    arguments: dict[str, Any] = {
        "query": scope.query or request.question,
        "page": 1,
        "page_size": request.limits.candidate_papers,
    }
    for name in ("authors", "journals", "subjects"):
        value = getattr(scope, name)
        if value:
            arguments[name] = value
    if scope.year_from is not None:
        arguments["year_from"] = scope.year_from
    if scope.year_to is not None:
        arguments["year_to"] = scope.year_to
    return arguments


def _semantic_arguments(
    request: ResearchRequest,
    doc_ids: list[str],
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "query": request.question,
        "top_k": request.limits.evidence_chunks,
        "mode": request.semantic_mode,
        "source_types": ["pdf"],
    }
    filters = _soft_semantic_filters(request.scope)
    if request.strict_paper_scope:
        filters["doc_id"] = doc_ids
    if filters:
        arguments["filters"] = filters
    return arguments


def _soft_semantic_filters(scope: PaperScope | None) -> dict[str, Any]:
    if scope is None:
        return {}
    filters: dict[str, Any] = {}
    if scope.authors:
        filters["author"] = scope.authors
    if scope.journals:
        filters["publication_venue_name_unified"] = scope.journals
    if scope.year_from is not None or scope.year_to is not None:
        year_range: dict[str, int] = {}
        if scope.year_from is not None:
            year_range["gte"] = scope.year_from
        if scope.year_to is not None:
            year_range["lte"] = scope.year_to
        filters["publication_published_year"] = year_range
    return filters


def _paper_records(call_id: str, result: ToolCallResult) -> list[PaperRecord]:
    if not result.ok:
        return []
    records: list[PaperRecord] = []
    for rank, hit in enumerate(_result_items(result.output), start=1):
        source = _source(hit)
        records.append(
            PaperRecord(
                source_tool_call_id=call_id,
                rank=rank,
                unique_id=_optional_string(source.get("unique_id")),
                doc_id=_optional_string(source.get("doc_id")),
                title=_optional_string(source.get("title")),
                authors=_authors(source.get("author") or source.get("authors")),
                abstract=_optional_string(source.get("abstract")),
                doi=_optional_string(source.get("doi")),
                publication_year=_year(
                    source.get("publication_published_year")
                    or source.get("publication_year")
                    or source.get("year")
                ),
                venue=_optional_string(
                    source.get("publication_venue_name_unified")
                    or source.get("venue")
                    or source.get("source_title")
                ),
                raw_metadata=hit,
            )
        )
    return records


def _evidence_refs(
    call_id: str,
    result: ToolCallResult,
) -> tuple[list[EvidenceRef], int]:
    if not result.ok:
        return [], 0
    evidence: list[EvidenceRef] = []
    skipped = 0
    for hit in _result_items(result.output):
        source = _source(hit)
        doc_id = _optional_string(source.get("doc_id"))
        offset = _nonnegative_int(source.get("offset"))
        text = _optional_string(
            source.get("chunk") or source.get("text") or source.get("content")
        )
        if doc_id is None or offset is None or text is None:
            skipped += 1
            continue
        score = _number(source.get("score", hit.get("_score")))
        evidence.append(
            EvidenceRef(
                evidence_id=f"evidence-{len(evidence) + 1}",
                source_tool_call_id=call_id,
                rank=len(evidence) + 1,
                doc_id=doc_id,
                chunk_id=_optional_string(source.get("chunk_id")),
                offset=offset,
                title=_optional_string(source.get("title")),
                publication_year=_year(
                    source.get("publication_published_year")
                    or source.get("publication_year")
                    or source.get("year")
                ),
                score=score,
                quoted_text=text,
                raw_chunk=hit,
            )
        )
    return evidence, skipped


def _original_text_excerpt(
    call_id: str,
    doc_id: str,
    offset: int,
    result: ToolCallResult,
) -> OriginalTextExcerpt | None:
    if not result.ok or not isinstance(result.output, dict):
        return None
    root = _result_root(result.output)
    text = _optional_string(
        root.get("text")
        or root.get("content")
        or root.get("fragment")
        or root.get("chunk")
    )
    if text is None:
        return None
    return OriginalTextExcerpt(
        source_tool_call_id=call_id,
        doc_id=doc_id,
        offset=offset,
        text=text,
        bytes_returned=_nonnegative_int(root.get("bytes_returned")),
        next_offset=_nonnegative_int(root.get("next_offset")),
        more=root.get("more") if isinstance(root.get("more"), bool) else None,
    )


def _result_root(output: dict[str, Any]) -> dict[str, Any]:
    data = output.get("data")
    return data if isinstance(data, dict) else output


def _result_items(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    root: Any = output.get("data", output)
    if isinstance(root, list):
        items = root
    elif isinstance(root, dict):
        items = next(
            (
                root[name]
                for name in ("hits", "papers", "chunks", "results", "items")
                if isinstance(root.get(name), list)
            ),
            [],
        )
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _source(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source")
    return source if isinstance(source, dict) else hit


def _result_count(tool_name: str, result: ToolCallResult) -> int:
    if not result.ok:
        return 0
    if tool_name in {"search_papers", "semantic_search"}:
        return len(_result_items(result.output))
    if tool_name == "read_content" and isinstance(result.output, dict):
        return 1
    return 0


def _status(
    search_result: ToolCallResult,
    semantic_result: ToolCallResult,
    traces: list[EvidenceToolTrace],
    evidence: list[EvidenceRef],
) -> EvidenceResearchStatus:
    if not semantic_result.ok:
        return EvidenceResearchStatus.FAILED
    any_failure = not search_result.ok or any(not trace.ok for trace in traces)
    if evidence:
        return (
            EvidenceResearchStatus.PARTIAL
            if any_failure
            else EvidenceResearchStatus.COMPLETE
        )
    return (
        EvidenceResearchStatus.FAILED
        if any_failure
        else EvidenceResearchStatus.NO_EVIDENCE
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _year(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and 1400 <= parsed <= 2200 else None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _authors(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            authors.append(item.strip())
        elif isinstance(item, dict):
            name = _optional_string(
                item.get("display_name")
                or item.get("name")
                or item.get("author_name")
            )
            if name:
                authors.append(name)
    return authors


def _unique_strings(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
