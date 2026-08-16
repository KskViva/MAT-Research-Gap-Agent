from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from research_gap_agent.evidence import (
    EvidenceResearchStatus,
    PaperScope,
    ResearchLimits,
    ResearchRequest,
    SciVerseEvidenceResearcher,
)
from research_gap_agent.tools import build_sciverse_tool_registry


@dataclass
class FakeMcpClient:
    responses: dict[str, Any]
    failures: set[str] = field(default_factory=set)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name in self.failures:
            raise RuntimeError(f"simulated {name} failure")
        response = self.responses.get(name, {})
        if callable(response):
            return response(arguments)
        return response


def _researcher(client: FakeMcpClient) -> SciVerseEvidenceResearcher:
    return SciVerseEvidenceResearcher(
        build_sciverse_tool_registry(client=client)
    )


def _successful_client() -> FakeMcpClient:
    return FakeMcpClient(
        responses={
            "search_papers": {
                "data": {
                    "hits": [
                        {
                            "unique_id": "paper:10.1000/llzo",
                            "doc_id": "doc-llzo",
                            "title": "LLZO interface evidence",
                            "author": ["Ada Researcher"],
                            "abstract": "Original abstract.",
                            "doi": "10.1000/llzo",
                            "publication_published_year": 2025,
                            "publication_venue_name_unified": "Evidence Journal",
                            "provider_extra": {"unchanged": True},
                        }
                    ]
                }
            },
            "semantic_search": {
                "data": {
                    "hits": [
                        {
                            "chunk_id": "chunk-7",
                            "doc_id": "doc-llzo",
                            "chunk": "Observed interfacial degradation evidence.",
                            "offset": 2048,
                            "score": 0.91,
                            "title": "LLZO interface evidence",
                            "publication_published_year": 2025,
                            "provider_extra": {"unchanged": True},
                        }
                    ]
                }
            },
            "read_content": {
                "data": {
                    "text": "Expanded original text around the evidence.",
                    "bytes_returned": 512,
                    "next_offset": 2560,
                    "more": True,
                }
            },
        }
    )


def test_research_returns_traceable_papers_evidence_and_original_context() -> None:
    client = _successful_client()
    request = ResearchRequest(
        question="How does the LLZO interface degrade?",
        scope=PaperScope(
            query="LLZO interface degradation",
            year_from=2020,
            year_to=2025,
        ),
        limits=ResearchLimits(
            candidate_papers=5,
            evidence_chunks=4,
            context_expansions=1,
            read_bytes=512,
        ),
        semantic_mode="quality",
        strict_paper_scope=True,
    )

    result = _researcher(client).research(request)

    assert result.status == EvidenceResearchStatus.COMPLETE
    assert [name for name, _ in client.calls] == [
        "search_papers",
        "semantic_search",
        "read_content",
    ]
    assert client.calls[1][1]["filters"]["doc_id"] == ["doc-llzo"]
    assert client.calls[1][1]["filters"]["publication_published_year"] == {
        "gte": 2020,
        "lte": 2025,
    }
    assert result.papers[0].raw_metadata["provider_extra"] == {
        "unchanged": True
    }
    evidence = result.evidence[0]
    assert evidence.chunk_id == "chunk-7"
    assert evidence.doc_id == "doc-llzo"
    assert evidence.offset == 2048
    assert evidence.score == 0.91
    assert evidence.raw_chunk["provider_extra"] == {"unchanged": True}
    assert evidence.context is not None
    assert evidence.context.text == "Expanded original text around the evidence."
    assert evidence.context.source_tool_call_id == "evidence-call-003"
    assert [trace.result_count for trace in result.tool_traces] == [1, 1, 1]
    assert '"chunk_id":"chunk-7"' in result.model_dump_json()


def test_broad_research_does_not_turn_soft_filters_into_doc_id_scope() -> None:
    client = _successful_client()

    result = _researcher(client).research(
        ResearchRequest(
            question="Broad LLZO degradation evidence",
            limits=ResearchLimits(context_expansions=0),
        )
    )

    assert result.status == EvidenceResearchStatus.COMPLETE
    assert "doc_id" not in client.calls[1][1]["filters"]
    assert [name for name, _ in client.calls] == [
        "search_papers",
        "semantic_search",
    ]
    assert result.evidence[0].context is None


def test_empty_strict_candidate_scope_never_falls_back_to_global_search() -> None:
    client = FakeMcpClient(
        responses={
            "search_papers": {"data": {"hits": []}},
            "semantic_search": lambda arguments: (
                {"data": {"hits": []}}
                if arguments["filters"]["doc_id"] == []
                else pytest.fail("strict empty scope must remain empty")
            ),
        }
    )

    result = _researcher(client).research(
        ResearchRequest(
            question="Evidence within an empty candidate set",
            strict_paper_scope=True,
        )
    )

    assert result.status == EvidenceResearchStatus.NO_EVIDENCE
    assert client.calls[1][1]["filters"]["doc_id"] == []
    assert result.evidence == []
    assert len(result.tool_traces) == 2


def test_read_failure_is_preserved_as_partial_result() -> None:
    client = _successful_client()
    client.failures.add("read_content")

    result = _researcher(client).research(
        ResearchRequest(question="LLZO context failure evidence")
    )

    assert result.status == EvidenceResearchStatus.PARTIAL
    assert len(result.evidence) == 1
    assert result.evidence[0].context is None
    assert result.tool_traces[-1].ok is False
    assert result.tool_traces[-1].error_type == "RuntimeError"
    assert "read_content produced no usable context" in result.warnings[-1]


def test_semantic_search_failure_returns_failed_structured_result() -> None:
    client = _successful_client()
    client.failures.add("semantic_search")

    result = _researcher(client).research(
        ResearchRequest(question="LLZO semantic failure evidence")
    )

    assert result.status == EvidenceResearchStatus.FAILED
    assert result.evidence == []
    assert result.papers
    assert result.tool_traces[-1].tool_name == "semantic_search"
    assert result.tool_traces[-1].ok is False


def test_malformed_semantic_hits_are_skipped_without_fabricating_ids() -> None:
    client = _successful_client()
    client.responses["semantic_search"] = {
        "hits": [
            {"chunk_id": "missing-doc", "chunk": "Text", "offset": 0},
            {"doc_id": "doc-1", "chunk": "Text", "offset": -1},
            {"doc_id": "doc-2", "offset": 0},
        ]
    }

    result = _researcher(client).research(
        ResearchRequest(question="Malformed evidence records")
    )

    assert result.status == EvidenceResearchStatus.NO_EVIDENCE
    assert result.evidence == []
    assert any("3 semantic hit(s)" in warning for warning in result.warnings)


def test_non_finite_score_is_not_copied_into_structured_evidence() -> None:
    client = _successful_client()
    client.responses["semantic_search"]["data"]["hits"][0]["score"] = "NaN"

    result = _researcher(client).research(
        ResearchRequest(
            question="Evidence with an invalid provider score",
            limits=ResearchLimits(context_expansions=0),
        )
    )

    assert result.status == EvidenceResearchStatus.COMPLETE
    assert result.evidence[0].score is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "  "},
        {
            "question": "valid question",
            "scope": {"year_from": 2025, "year_to": 2020},
        },
        {
            "question": "valid question",
            "limits": {"evidence_chunks": 2, "context_expansions": 3},
        },
        {
            "question": "valid question",
            "limits": {"read_bytes": 16385},
        },
    ],
)
def test_request_rejects_empty_and_out_of_range_inputs(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest.model_validate(payload)
