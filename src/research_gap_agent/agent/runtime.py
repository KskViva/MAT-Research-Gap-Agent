"""Bounded LLM tool-calling loop for SciVerse-grounded research."""

from __future__ import annotations

import json
from typing import Any, Protocol

from research_gap_agent.tools import ToolCallResult, ToolRegistry

from .models import (
    AgentResult,
    AgentStatus,
    AgentToolTrace,
    LlmReply,
    LlmToolCall,
    PaperRecord,
)


class ChatModel(Protocol):
    model_name: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LlmReply: ...


DEFAULT_SYSTEM_PROMPT = """You are an evidence-first scientific literature research agent.
Use SciVerse tools instead of model memory for literature facts.
Always call search_papers at least once before the final answer so the caller can
receive original paper metadata. Use semantic_search for evidence chunks and
read_content when the surrounding original text is needed. Preserve doc_id,
chunk_id, offsets, titles, and years in the answer. Never claim that a PDF was
downloaded: the available tools return metadata, text evidence, and provenance.
State uncertainty when evidence is incomplete and do not declare a research gap
without first searching for counter-evidence."""


class DeepResearchAgent:
    """Execute validated SciVerse tool calls until the model returns an answer."""

    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_rounds: int = 6,
        max_tool_result_characters: int = 60_000,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if max_tool_result_characters < 1000:
            raise ValueError("max_tool_result_characters must be at least 1000")
        self._model = model
        self._registry = registry
        self._system_prompt = system_prompt.strip()
        self._max_tool_rounds = max_tool_rounds
        self._max_tool_result_characters = max_tool_result_characters

    def run(self, research_direction: str) -> AgentResult:
        question = " ".join(research_direction.split())
        if len(question) < 3:
            raise ValueError("research_direction must contain at least 3 characters")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        traces: list[AgentToolTrace] = []
        papers: list[PaperRecord] = []
        schemas = self._registry.function_schemas()

        for round_index in range(1, self._max_tool_rounds + 2):
            reply = self._model.complete(messages, schemas)
            if not reply.tool_calls:
                return AgentResult(
                    status=AgentStatus.COMPLETE,
                    model=self._model.model_name,
                    answer=reply.content or "",
                    tool_traces=traces,
                    papers=papers,
                )

            if round_index > self._max_tool_rounds:
                break

            messages.append(_assistant_tool_message(reply))
            for call in reply.tool_calls:
                arguments, result = self._invoke(call)
                papers.extend(_paper_records(call, result))
                content, truncated, original_length = self._bounded_result(result)
                traces.append(
                    AgentToolTrace(
                        round_index=round_index,
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=arguments,
                        ok=result.ok,
                        error_type=result.error_type,
                        error_message=result.error_message,
                        result_characters=original_length,
                        result_truncated=truncated,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": content,
                    }
                )

        return AgentResult(
            status=AgentStatus.MAX_TOOL_ROUNDS,
            model=self._model.model_name,
            answer="The research stopped after reaching the configured tool-round limit.",
            tool_traces=traces,
            papers=papers,
        )

    def _invoke(
        self,
        call: LlmToolCall,
    ) -> tuple[dict[str, object], ToolCallResult]:
        try:
            decoded = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            return {}, ToolCallResult(
                tool_name=call.name,
                ok=False,
                error_type="invalid_json_arguments",
                error_message="tool arguments are not valid JSON",
            )
        if not isinstance(decoded, dict):
            return {}, ToolCallResult(
                tool_name=call.name,
                ok=False,
                error_type="invalid_json_arguments",
                error_message="tool arguments must decode to a JSON object",
            )
        arguments = {str(key): value for key, value in decoded.items()}
        return arguments, self._registry.invoke(call.name, arguments)

    def _bounded_result(
        self,
        result: ToolCallResult,
    ) -> tuple[str, bool, int]:
        serialized = result.model_dump_json()
        original_length = len(serialized)
        if original_length <= self._max_tool_result_characters:
            return serialized, False, original_length
        bounded = json.dumps(
            {
                "tool_name": result.tool_name,
                "ok": result.ok,
                "truncated": True,
                "original_characters": original_length,
                "prefix": serialized[: self._max_tool_result_characters],
            },
            ensure_ascii=False,
        )
        return bounded, True, original_length


def _assistant_tool_message(reply: LlmReply) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments_json,
                },
            }
            for call in reply.tool_calls
        ],
    }


def _paper_records(
    call: LlmToolCall,
    result: ToolCallResult,
) -> list[PaperRecord]:
    if call.name != "search_papers" or not result.ok:
        return []
    if not isinstance(result.output, dict):
        return []
    root = result.output.get("data", result.output)
    if not isinstance(root, dict):
        return []
    hits = next(
        (
            root[key]
            for key in ("hits", "papers", "results", "items")
            if isinstance(root.get(key), list)
        ),
        [],
    )
    records: list[PaperRecord] = []
    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source", hit)
        if not isinstance(source, dict):
            source = hit
        records.append(
            PaperRecord(
                source_tool_call_id=call.call_id,
                rank=rank,
                unique_id=_optional_string(source.get("unique_id")),
                doc_id=_optional_string(source.get("doc_id")),
                title=_optional_string(source.get("title")),
                authors=_authors(source.get("author") or source.get("authors")),
                abstract=_optional_string(source.get("abstract")),
                doi=_optional_string(source.get("doi")),
                publication_year=_publication_year(
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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _publication_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1400 <= year <= 2200 else None


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
