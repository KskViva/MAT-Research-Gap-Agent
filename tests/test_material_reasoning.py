import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from research_gap_agent.agent import LlmReply, LlmToolCall
from research_gap_agent.evidence import EvidenceRef, OriginalTextExcerpt
from research_gap_agent.materials import (
    DeepSeekMaterialKnowledgeModel,
    MaterialEntityProposal,
    MaterialKnowledgeDraft,
    MaterialKnowledgeReasoningError,
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


def _evidence(text: str = "Explicit material evidence.") -> EvidenceRef:
    return EvidenceRef(
        evidence_id="evidence-1",
        source_tool_call_id="evidence-call-002",
        rank=1,
        doc_id="doc-1",
        chunk_id="chunk-1",
        offset=100,
        title="Materials paper",
        publication_year=2025,
        score=0.9,
        quoted_text=text,
        context=OriginalTextExcerpt(
            source_tool_call_id="evidence-call-003",
            doc_id="doc-1",
            offset=100,
            text="Expanded original context.",
        ),
        raw_chunk={"provider_secret_field": "must-not-enter-prompt"},
    )


def test_deepseek_material_model_parses_fenced_json_and_bounds_source_text() -> None:
    draft = MaterialKnowledgeDraft(
        materials=[
            MaterialEntityProposal(
                name="LLZO",
                formula="Li7La3Zr2O12",
                evidence_ids=["evidence-1"],
            )
        ]
    )
    content = f"```json\n{draft.model_dump_json()}\n```"
    chat = FakeChatModel([LlmReply(content=content)])
    model = DeepSeekMaterialKnowledgeModel(
        chat,
        max_evidence_text_characters=500,
    )

    result = model.extract(
        "What materials information is reported?",
        [_evidence("x" * 1000)],
        max_records_per_type=10,
    )

    assert result.materials[0].formula == "Li7La3Zr2O12"
    assert chat.calls[0][1] == []
    payload = json.loads(chat.calls[0][0][1]["content"])
    assert len(payload["evidence"][0]["quoted_text"]) == 500
    assert payload["evidence"][0]["original_context"] == (
        "Expanded original context."
    )
    assert "raw_chunk" not in payload["evidence"][0]
    assert "provider_secret_field" not in chat.calls[0][0][1]["content"]
    assert payload["limits"]["max_records_per_type"] == 10


def test_deepseek_material_model_rejects_invalid_structured_json() -> None:
    model = DeepSeekMaterialKnowledgeModel(
        FakeChatModel([LlmReply(content="not-json")])
    )

    with pytest.raises(
        MaterialKnowledgeReasoningError,
        match="invalid structured JSON",
    ):
        model.extract(
            "What materials information is reported?",
            [_evidence()],
            max_records_per_type=10,
        )

def test_deepseek_material_model_rejects_unexpected_tool_calls() -> None:
    reply = LlmReply(
        content=None,
        tool_calls=[
            LlmToolCall(
                call_id="call-1",
                name="semantic_search",
                arguments_json='{"query":"not allowed here"}',
            )
        ],
    )
    model = DeepSeekMaterialKnowledgeModel(FakeChatModel([reply]))

    with pytest.raises(
        MaterialKnowledgeReasoningError,
        match="unexpected tool calls",
    ):
        model.extract(
            "What materials information is reported?",
            [_evidence()],
            max_records_per_type=10,
        )
