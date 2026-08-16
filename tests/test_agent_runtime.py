from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from research_gap_agent.agent import (
    AgentStatus,
    DeepResearchAgent,
    LlmReply,
    LlmToolCall,
)
from research_gap_agent.tools import ToolRegistry


class EchoArguments(BaseModel):
    value: str = Field(min_length=1)


@dataclass
class FakeChatModel:
    replies: list[LlmReply]
    model_name: str = "fake-deepseek"
    messages_seen: list[list[dict[str, object]]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmReply:
        assert tools
        self.messages_seen.append([dict(message) for message in messages])
        return self.replies.pop(0)


def _registry(*, long_output: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo a test value.",
        input_model=EchoArguments,
        handler=(
            (lambda arguments: {"text": arguments.value * 2000})
            if long_output
            else (lambda arguments: {"echo": arguments.value})
        ),
    )
    return registry


def _call(arguments: str = '{"value":"evidence"}') -> LlmReply:
    return LlmReply(
        tool_calls=[
            LlmToolCall(
                call_id="call-1",
                name="echo",
                arguments_json=arguments,
            )
        ]
    )


def _paper_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="search_papers",
        description="Search paper metadata.",
        input_model=EchoArguments,
        handler=lambda arguments: {
            "hits": [
                {
                    "unique_id": "paper:10.1000/original",
                    "doc_id": "doc-original",
                    "title": "Original paper title",
                    "author": [
                        "Ada Lovelace",
                        {"display_name": "Grace Hopper"},
                    ],
                    "abstract": "Original abstract text.",
                    "doi": "10.1000/original",
                    "publication_published_year": 2025,
                    "publication_venue_name_unified": "Journal of Evidence",
                    "provider_extra": {"unchanged": True},
                }
            ]
        },
    )
    return registry


def test_agent_executes_tool_and_returns_final_answer() -> None:
    model = FakeChatModel(replies=[_call(), LlmReply(content="grounded answer")])
    agent = DeepResearchAgent(model, _registry())

    result = agent.run("solid electrolyte evidence")

    assert result.status == AgentStatus.COMPLETE
    assert result.answer == "grounded answer"
    assert result.tool_traces[0].ok is True
    second_turn = model.messages_seen[1]
    assert second_turn[-1]["role"] == "tool"
    assert '"echo":"evidence"' in str(second_turn[-1]["content"])


def test_agent_returns_invalid_json_to_model_as_tool_error() -> None:
    model = FakeChatModel(replies=[_call("not-json"), LlmReply(content="recovered")])
    agent = DeepResearchAgent(model, _registry())

    result = agent.run("valid research question")

    assert result.status == AgentStatus.COMPLETE
    assert result.tool_traces[0].ok is False
    assert result.tool_traces[0].error_type == "invalid_json_arguments"


def test_agent_stops_at_configured_tool_round_limit() -> None:
    model = FakeChatModel(replies=[_call(), _call(), _call()])
    agent = DeepResearchAgent(model, _registry(), max_tool_rounds=2)

    result = agent.run("valid research question")

    assert result.status == AgentStatus.MAX_TOOL_ROUNDS
    assert len(result.tool_traces) == 2


def test_agent_bounds_large_tool_results() -> None:
    model = FakeChatModel(replies=[_call(), LlmReply(content="bounded")])
    agent = DeepResearchAgent(
        model,
        _registry(long_output=True),
        max_tool_result_characters=1000,
    )

    result = agent.run("valid research question")

    assert result.tool_traces[0].result_truncated is True
    assert result.tool_traces[0].result_characters > 1000


def test_agent_returns_normalized_and_original_paper_metadata() -> None:
    model = FakeChatModel(
        replies=[
            LlmReply(
                tool_calls=[
                    LlmToolCall(
                        call_id="paper-call",
                        name="search_papers",
                        arguments_json='{"value":"solid electrolyte"}',
                    )
                ]
            ),
            LlmReply(content="paper list ready"),
        ]
    )

    result = DeepResearchAgent(model, _paper_registry()).run(
        "solid electrolyte papers"
    )

    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.source_tool_call_id == "paper-call"
    assert paper.rank == 1
    assert paper.unique_id == "paper:10.1000/original"
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.publication_year == 2025
    assert paper.raw_metadata["provider_extra"] == {"unchanged": True}
