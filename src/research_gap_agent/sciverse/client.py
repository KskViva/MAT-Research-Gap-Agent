"""SciVerse MCP stdio client shared by the LLM tool layer."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class SciverseError(RuntimeError):
    """Base error for the SciVerse MCP boundary."""


class SciverseConnectionError(SciverseError):
    """The SciVerse MCP process or upstream service could not be reached."""


class SciverseRequestError(SciverseError):
    """SciVerse rejected a request or returned an invalid result."""


class McpToolCaller(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool and return a JSON-serializable object."""


class StdioSciverseMcpClient:
    """Start the installed SciVerse stdio server for one bounded tool call."""

    def __init__(
        self,
        *,
        env_file: str | Path = ".env",
        server_command: str | Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._env_file = Path(env_file)
        (
            self._server_command,
            self._server_args,
            self._server_entrypoint,
        ) = _resolve_server_launch(server_command)
        self._timeout_seconds = timeout_seconds

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("MCP tool name must not be blank")
        if not self._server_command.is_file() or not self._server_entrypoint.is_file():
            raise SciverseConnectionError(
                "SciVerse MCP server executable was not found at the configured path"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._call_tool_async(name, arguments))
        raise SciverseConnectionError(
            "the synchronous SciVerse client cannot run inside an active event loop"
        )

    async def _call_tool_async(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        parameters = StdioServerParameters(
            command=str(self._server_command),
            args=self._server_args,
            env=self._child_environment(),
            cwd=str(self._env_file.parent.resolve()),
        )
        try:
            with open(os.devnull, "w", encoding="utf-8") as errlog:
                async with asyncio.timeout(self._timeout_seconds):
                    async with stdio_client(parameters, errlog=errlog) as streams:
                        read_stream, write_stream = streams
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(name, arguments)
        except TimeoutError as exc:
            raise SciverseConnectionError("SciVerse MCP call timed out") from exc
        except OSError as exc:
            raise SciverseConnectionError(
                "SciVerse MCP server process could not be started"
            ) from exc
        except SciverseError:
            raise
        except Exception as exc:
            raise SciverseConnectionError(
                "SciVerse MCP session could not be completed"
            ) from exc
        return _decode_tool_result(result, name)

    def _child_environment(self) -> dict[str, str]:
        values = dotenv_values(self._env_file) if self._env_file.is_file() else {}
        token = os.environ.get("SCIVERSE_API_TOKEN") or values.get(
            "SCIVERSE_API_TOKEN"
        )
        if not token or not token.strip():
            raise SciverseRequestError(
                "SCIVERSE_API_TOKEN is missing from the environment and .env"
            )
        child_env = {"SCIVERSE_API_TOKEN": token.strip()}
        base_url = os.environ.get("SCIVERSE_BASE_URL") or values.get(
            "SCIVERSE_BASE_URL"
        )
        if base_url and base_url.strip():
            child_env["SCIVERSE_BASE_URL"] = base_url.strip()
        return child_env


def _decode_tool_result(result: Any, name: str) -> dict[str, Any]:
    is_error = getattr(result, "is_error", getattr(result, "isError", False))
    if is_error:
        detail = _text_content(result.content)
        suffix = f": {detail[:300]}" if detail else ""
        raise SciverseRequestError(f"SciVerse MCP tool {name} failed{suffix}")

    structured = getattr(
        result,
        "structured_content",
        getattr(result, "structuredContent", None),
    )
    if isinstance(structured, dict):
        return structured

    images = _image_content(result.content)
    if images:
        return {"resources": images}

    text = _text_content(result.content)
    if not text:
        raise SciverseRequestError("SciVerse MCP returned no JSON text content")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SciverseRequestError("SciVerse MCP returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SciverseRequestError("SciVerse MCP JSON payload must be an object")
    return payload


def _resolve_server_launch(
    server_command: str | Path | None,
) -> tuple[Path, list[str], Path]:
    if server_command is not None:
        entrypoint = Path(server_command)
        if os.name == "nt" and entrypoint.suffix.casefold() in {".cmd", ".bat"}:
            command = Path(os.environ.get("COMSPEC") or shutil.which("cmd") or "cmd.exe")
            return command, ["/d", "/s", "/c", str(entrypoint)], entrypoint
        return entrypoint, [], entrypoint

    environment_directory = Path(sys.executable).parent
    script = (
        environment_directory
        / "node_modules"
        / "sciverse-mcp-server"
        / "dist"
        / "cli.js"
    )
    node = shutil.which("node")
    command = Path(node) if node else environment_directory / "node.exe"
    return command, [str(script)], script


def _text_content(content: list[Any]) -> str:
    texts = [
        item.text
        for item in content
        if getattr(item, "type", None) == "text" and isinstance(item.text, str)
    ]
    return "\n".join(texts).strip()


def _image_content(content: list[Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for item in content:
        if getattr(item, "type", None) != "image":
            continue
        data = getattr(item, "data", None)
        mime_type = getattr(
            item,
            "mime_type",
            getattr(item, "mimeType", None),
        )
        if isinstance(data, str) and isinstance(mime_type, str):
            images.append({"data_base64": data, "mime_type": mime_type})
    return images
