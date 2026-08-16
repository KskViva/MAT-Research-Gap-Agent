import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from research_gap_agent.agent import LlmReply
from research_gap_agent.evidence import (
    EvidenceRef,
    EvidenceResearchResult,
    EvidenceResearchStatus,
    ResearchLimits,
    ResearchRequest,
)
from research_gap_agent.synthesis import (
    Claim,
    DeepSeekGapReasoningModel,
    GapCandidate,
    GapCategory,
    GapReasoningError,
)


@dataclass
class FakeChatModel:
    replies: list[LlmReply]
    model_name: str = "fake-deepseek"
    calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = field(
        default_factory=list
    )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LlmReply:
        self.calls.append((messages, tools))
        return self.replies.pop(0)


def _evidence(text: str = "Observed evidence.") -> EvidenceRef:
    return EvidenceRef(
        evidence_id="evidence-1",
        source_tool_call_id="evidence-call-002",
        rank=1,
        doc_id="doc-1",
        chunk_id="chunk-1",
        offset=100,
        title="Evidence paper",
        publication_year=2025,
        score=0.9,
        quoted_text=text,
        raw_chunk={"provider_secret_field": "must-not-enter-prompt"},
    )


def test_deepseek_reasoning_parses_fenced_synthesis_json_and_bounds_text() -> None:
    content = """```json
{"claims":[{"statement":"Observed effect.","evidence_ids":["evidence-1"]}],"gap_candidates":[]}
```"""
    chat = FakeChatModel([LlmReply(content=content)])
    model = DeepSeekGapReasoningModel(
        chat,
        max_evidence_text_characters=500,
    )

    draft = model.synthesize(
        "What evidence is missing?",
        [_evidence("x" * 1000)],
        max_claims=5,
        max_gap_candidates=2,
    )

    assert draft.claims[0].evidence_ids == ["evidence-1"]
    assert chat.calls[0][1] == []
    payload = json.loads(chat.calls[0][0][1]["content"])
    assert len(payload["evidence"][0]["quoted_text"]) == 500
    assert "raw_chunk" not in payload["evidence"][0]
    assert "provider_secret_field" not in chat.calls[0][0][1]["content"]


def test_deepseek_reasoning_rejects_invalid_structured_json() -> None:
    model = DeepSeekGapReasoningModel(
        FakeChatModel([LlmReply(content="not-json")])
    )

    with pytest.raises(GapReasoningError, match="invalid structured JSON"):
        model.synthesize(
            "What evidence is missing?",
            [_evidence()],
            max_claims=5,
            max_gap_candidates=2,
        )


def test_deepseek_reasoning_assessment_contains_only_bounded_evidence_fields() -> None:
    response = json.dumps(
        {
            "status": "rejected",
            "rationale": "Counter-evidence closes the candidate.",
            "counter_evidence_ids": ["evidence-1"],
            "refined_scope": None,
        }
    )
    chat = FakeChatModel([LlmReply(content=response)])
    model = DeepSeekGapReasoningModel(chat)
    candidate = GapCandidate(
        gap_id="gap-1",
        statement="A condition has not been studied.",
        rationale="Initial evidence omitted the condition.",
        category=GapCategory.MISSING_CONDITION,
        supporting_claim_ids=["claim-1"],
        uncertainty="The retrieval may be incomplete.",
        counter_query="Studies covering the omitted condition",
    )
    claims = [
        Claim(
            claim_id="claim-1",
            statement="Initial evidence omitted the condition.",
            evidence_ids=["evidence-1"],
        )
    ]
    counter = EvidenceResearchResult(
        status=EvidenceResearchStatus.COMPLETE,
        request=ResearchRequest(
            question="Studies covering the omitted condition",
            limits=ResearchLimits(context_expansions=0),
        ),
        evidence=[_evidence()],
    )

    assessment = model.assess_gap(candidate, claims, counter)

    assert assessment.counter_evidence_ids == ["evidence-1"]
    payload = json.loads(chat.calls[0][0][1]["content"])
    assert payload["counter_research_status"] == "complete"
    assert payload["counter_evidence"][0]["doc_id"] == "doc-1"
    assert "raw_chunk" not in payload["counter_evidence"][0]
