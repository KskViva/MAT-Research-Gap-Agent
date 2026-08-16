"""Structured DeepSeek reasoning for evidence synthesis and falsification."""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from research_gap_agent.agent.runtime import ChatModel
from research_gap_agent.evidence import EvidenceRef, EvidenceResearchResult

from .models import (
    Claim,
    GapAssessmentDraft,
    GapCandidate,
    SynthesisDraft,
)

StructuredReply = TypeVar("StructuredReply", bound=BaseModel)


class GapReasoningError(RuntimeError):
    """A structured synthesis or verification reply could not be trusted."""


class GapReasoningModel(Protocol):
    model_name: str

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_claims: int,
        max_gap_candidates: int,
    ) -> SynthesisDraft: ...

    def assess_gap(
        self,
        candidate: GapCandidate,
        claims: list[Claim],
        counter_result: EvidenceResearchResult,
    ) -> GapAssessmentDraft: ...


_SYNTHESIS_SYSTEM_PROMPT = """You synthesize only the supplied SciVerse evidence.
Return exactly one JSON object matching the supplied schema and no prose.
Every claim must cite one or more supplied evidence_id values. Do not invent or
alter evidence IDs, doc_id, chunk_id, offsets, DOI, or quoted text. A gap is only
a candidate: derive it from sparse coverage, a missing condition, conflicting
evidence, or a methodological gap, and provide a counter-query designed to find
evidence that would disprove or narrow it. Never claim that nobody has studied a
topic and never use model memory as evidence."""


_ASSESSMENT_SYSTEM_PROMPT = """Assess one gap candidate using only the supplied
claims and newly retrieved SciVerse counter-evidence. Return exactly one JSON
object matching the supplied schema and no prose. Cite counter evidence only by
the supplied evidence_id. Use rejected when counter-evidence directly closes the
candidate, refined when it narrows the candidate, verified only when the bounded
counter-search completed without disproof, and insufficient_evidence when the
available material cannot support a decision. Never invent source identifiers or
scientific facts."""


class DeepSeekGapReasoningModel:
    """Use an existing ChatModel for strict, bounded JSON reasoning."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_evidence_text_characters: int = 8000,
        max_reply_characters: int = 40_000,
    ) -> None:
        if max_evidence_text_characters < 500:
            raise ValueError("max_evidence_text_characters must be at least 500")
        if max_reply_characters < 1000:
            raise ValueError("max_reply_characters must be at least 1000")
        self._chat_model = chat_model
        self.model_name = chat_model.model_name
        self._max_evidence_text_characters = max_evidence_text_characters
        self._max_reply_characters = max_reply_characters

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_claims: int,
        max_gap_candidates: int,
    ) -> SynthesisDraft:
        payload = {
            "question": question,
            "limits": {
                "max_claims": max_claims,
                "max_gap_candidates": max_gap_candidates,
            },
            "evidence": [self._evidence_payload(item) for item in evidence],
            "output_schema": SynthesisDraft.model_json_schema(),
        }
        return self._complete_json(
            _SYNTHESIS_SYSTEM_PROMPT,
            payload,
            SynthesisDraft,
            "synthesis",
        )

    def assess_gap(
        self,
        candidate: GapCandidate,
        claims: list[Claim],
        counter_result: EvidenceResearchResult,
    ) -> GapAssessmentDraft:
        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "supporting_claims": [item.model_dump(mode="json") for item in claims],
            "counter_research_status": counter_result.status.value,
            "counter_evidence": [
                self._evidence_payload(item) for item in counter_result.evidence
            ],
            "output_schema": GapAssessmentDraft.model_json_schema(),
        }
        return self._complete_json(
            _ASSESSMENT_SYSTEM_PROMPT,
            payload,
            GapAssessmentDraft,
            "gap assessment",
        )

    def close(self) -> None:
        close = getattr(self._chat_model, "close", None)
        if callable(close):
            close()

    def _evidence_payload(self, item: EvidenceRef) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence_id": item.evidence_id,
            "doc_id": item.doc_id,
            "chunk_id": item.chunk_id,
            "offset": item.offset,
            "title": item.title,
            "publication_year": item.publication_year,
            "score": item.score,
            "quoted_text": item.quoted_text[
                : self._max_evidence_text_characters
            ],
        }
        if item.context is not None:
            payload["original_context"] = item.context.text[
                : self._max_evidence_text_characters
            ]
        return payload

    def _complete_json(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        output_model: type[StructuredReply],
        stage: str,
    ) -> StructuredReply:
        try:
            reply = self._chat_model.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                [],
            )
        except Exception as exc:
            raise GapReasoningError(
                f"DeepSeek {stage} request could not be completed"
            ) from exc
        if reply.tool_calls:
            raise GapReasoningError(
                f"DeepSeek {stage} returned unexpected tool calls"
            )
        content = (reply.content or "").strip()
        if not content:
            raise GapReasoningError(f"DeepSeek {stage} returned no JSON content")
        if len(content) > self._max_reply_characters:
            raise GapReasoningError(
                f"DeepSeek {stage} JSON exceeded the configured size limit"
            )
        try:
            decoded = json.loads(_strip_json_fence(content))
            return output_model.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise GapReasoningError(
                f"DeepSeek {stage} returned invalid structured JSON"
            ) from exc


def _strip_json_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content
