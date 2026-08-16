"""Structured DeepSeek reasoning for materials knowledge extraction."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from research_gap_agent.agent.runtime import ChatModel
from research_gap_agent.evidence import EvidenceRef

from .models import MaterialKnowledgeDraft


class MaterialKnowledgeReasoningError(RuntimeError):
    """A structured extraction reply could not be trusted."""


class MaterialKnowledgeReasoningModel(Protocol):
    model_name: str

    def extract(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_records_per_type: int,
    ) -> MaterialKnowledgeDraft: ...


_EXTRACTION_SYSTEM_PROMPT = """You extract materials-science knowledge only
from the supplied SciVerse evidence. Return exactly one JSON object matching the
supplied schema and no prose. Extract only facts explicitly stated in the text:
materials and compositions, structures or phases, property observations with
reported values and conditions, synthesis procedures, simulation methods, and
explicit relationships. Preserve source wording for formulas, values, units,
conditions, method names, and software. Every record must cite one or more
supplied evidence_id values. Never invent or alter evidence IDs, doc_id,
chunk_id, offsets, source text, or scientific facts. Do not infer missing values
and do not merge materials merely because their names appear similar."""


class DeepSeekMaterialKnowledgeModel:
    """Use an existing ChatModel for strict, bounded extraction JSON."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_evidence_text_characters: int = 8000,
        max_reply_characters: int = 80_000,
    ) -> None:
        if max_evidence_text_characters < 500:
            raise ValueError("max_evidence_text_characters must be at least 500")
        if max_reply_characters < 1000:
            raise ValueError("max_reply_characters must be at least 1000")
        self._chat_model = chat_model
        self.model_name = chat_model.model_name
        self._max_evidence_text_characters = max_evidence_text_characters
        self._max_reply_characters = max_reply_characters

    def extract(
        self,
        question: str,
        evidence: list[EvidenceRef],
        *,
        max_records_per_type: int,
    ) -> MaterialKnowledgeDraft:
        payload = {
            "question": question,
            "limits": {"max_records_per_type": max_records_per_type},
            "evidence": [self._evidence_payload(item) for item in evidence],
            "output_schema": MaterialKnowledgeDraft.model_json_schema(),
        }
        try:
            reply = self._chat_model.complete(
                [
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                [],
            )
        except Exception as exc:
            raise MaterialKnowledgeReasoningError(
                "DeepSeek material extraction request could not be completed"
            ) from exc
        if reply.tool_calls:
            raise MaterialKnowledgeReasoningError(
                "DeepSeek material extraction returned unexpected tool calls"
            )
        content = (reply.content or "").strip()
        if not content:
            raise MaterialKnowledgeReasoningError(
                "DeepSeek material extraction returned no JSON content"
            )
        if len(content) > self._max_reply_characters:
            raise MaterialKnowledgeReasoningError(
                "DeepSeek material extraction JSON exceeded the configured size limit"
            )
        try:
            return MaterialKnowledgeDraft.model_validate_json(
                _strip_json_fence(content)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise MaterialKnowledgeReasoningError(
                "DeepSeek material extraction returned invalid structured JSON"
            ) from exc

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
            "quoted_text": item.quoted_text[
                : self._max_evidence_text_characters
            ],
        }
        if item.context is not None:
            payload["original_context"] = item.context.text[
                : self._max_evidence_text_characters
            ]
        return payload


def _strip_json_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    lines = content.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content
