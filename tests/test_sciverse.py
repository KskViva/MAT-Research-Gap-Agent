from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, ImageContent, TextContent

from research_gap_agent.sciverse import (
    SciverseRequestError,
    StdioSciverseMcpClient,
)
from research_gap_agent.sciverse.client import _decode_tool_result


def test_stdio_client_loads_token_from_env_file_without_global_mutation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("SCIVERSE_API_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SCIVERSE_API_TOKEN=test-token\n", encoding="utf-8")
    client = StdioSciverseMcpClient(
        env_file=env_file,
        server_command=tmp_path / "server.cmd",
    )

    assert client._child_environment() == {"SCIVERSE_API_TOKEN": "test-token"}


def test_mcp_v2_result_fields_are_decoded_from_python_names() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text='{"hits": []}')],
        isError=False,
    )

    assert _decode_tool_result(result, "search_papers") == {"hits": []}


def test_mcp_image_result_is_normalized_without_claiming_pdf() -> None:
    result = CallToolResult(
        content=[ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png")],
        isError=False,
    )

    assert _decode_tool_result(result, "get_resource") == {
        "resources": [
            {"data_base64": "aW1hZ2U=", "mime_type": "image/png"}
        ]
    }


def test_mcp_error_is_mapped_without_raw_structural_failure() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="invalid token")],
        isError=True,
    )

    try:
        _decode_tool_result(result, "search_papers")
    except SciverseRequestError as exc:
        assert "search_papers" in str(exc)
    else:
        raise AssertionError("expected MCP error result to fail")
