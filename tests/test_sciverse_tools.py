from dataclasses import dataclass, field
from typing import Any

from research_gap_agent.tools import build_sciverse_tool_registry


@dataclass
class FakeMcpClient:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"tool": name, "arguments": arguments}


def test_sciverse_registry_exposes_only_default_text_tools() -> None:
    registry = build_sciverse_tool_registry(client=FakeMcpClient())

    assert [item["function"]["name"] for item in registry.function_schemas()] == [
        "list_catalog",
        "search_papers",
        "semantic_search",
        "read_content",
    ]


def test_sciverse_registry_can_explicitly_enable_resource_tool() -> None:
    registry = build_sciverse_tool_registry(
        client=FakeMcpClient(),
        include_resources=True,
    )

    assert registry.function_schemas()[-1]["function"]["name"] == "get_resource"
    unsafe = registry.invoke("get_resource", {"file_name": "../paper.pdf"})
    assert unsafe.ok is False
    assert unsafe.error_type == "validation_error"


def test_sciverse_registry_validates_and_dispatches_semantic_search() -> None:
    client = FakeMcpClient()
    registry = build_sciverse_tool_registry(client=client)

    result = registry.invoke(
        "semantic_search",
        {"query": "LLZO interfacial stability", "top_k": 5, "mode": "quality"},
    )

    assert result.ok is True
    assert client.calls[0][0] == "semantic_search"
    assert client.calls[0][1]["query"] == "LLZO interfacial stability"
    assert client.calls[0][1]["source_types"] == ["pdf"]


def test_sciverse_registry_rejects_empty_search_and_invalid_limits() -> None:
    client = FakeMcpClient()
    registry = build_sciverse_tool_registry(client=client)

    empty = registry.invoke("search_papers", {})
    too_large = registry.invoke(
        "read_content",
        {"doc_id": "doc-1", "limit": 20_000},
    )

    assert empty.ok is False
    assert empty.error_type == "validation_error"
    assert too_large.ok is False
    assert client.calls == []


def test_search_papers_omits_absent_optional_values() -> None:
    client = FakeMcpClient()
    registry = build_sciverse_tool_registry(client=client)

    result = registry.invoke("search_papers", {"query": "solid electrolyte"})

    assert result.ok is True
    assert "year_from" not in client.calls[0][1]
    assert "year_to" not in client.calls[0][1]
