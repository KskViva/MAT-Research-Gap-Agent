"""DeepSeek Chat Completions adapter using the OpenAI Python SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import APIError, OpenAI

from .models import LlmReply, LlmToolCall


class DeepSeekError(RuntimeError):
    """A DeepSeek request could not be completed safely."""


class DeepSeekChatModel:
    """Small injectable adapter around DeepSeek's OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-pro"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_name = model.strip()
        self._owns_client = client is None
        self._client = client or OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    @classmethod
    def from_env(
        cls,
        env_file: str | Path = ".env",
        *,
        client: Any | None = None,
    ) -> DeepSeekChatModel:
        values = dotenv_values(env_file) if Path(env_file).is_file() else {}
        api_key = os.environ.get("DEEPSEEK_API_KEY") or values.get(
            "DEEPSEEK_API_KEY"
        )
        if not api_key or not api_key.strip():
            raise DeepSeekError(
                "DEEPSEEK_API_KEY is missing from the environment and .env"
            )
        model = os.environ.get("DEEPSEEK_MODEL") or values.get("DEEPSEEK_MODEL")
        base_url = os.environ.get("DEEPSEEK_BASE_URL") or values.get(
            "DEEPSEEK_BASE_URL"
        )
        return cls(
            api_key,
            model=(model or cls.DEFAULT_MODEL),
            base_url=(base_url or cls.DEFAULT_BASE_URL),
            client=client,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LlmReply:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        try:
            response = self._client.chat.completions.create(**request)
        except APIError as exc:
            request_id = getattr(exc, "request_id", None)
            suffix = f" (request_id={request_id})" if request_id else ""
            raise DeepSeekError(f"DeepSeek API request failed{suffix}") from exc
        except Exception as exc:
            raise DeepSeekError("DeepSeek client request could not be completed") from exc

        if not response.choices:
            raise DeepSeekError("DeepSeek returned no completion choices")
        message = response.choices[0].message
        calls = [
            LlmToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments_json=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ]
        return LlmReply(content=message.content, tool_calls=calls)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
