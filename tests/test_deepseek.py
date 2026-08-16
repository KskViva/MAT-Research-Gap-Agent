from pathlib import Path
from types import SimpleNamespace
from typing import Any

from research_gap_agent.agent import DeepSeekChatModel


class FakeCompletions:
    def __init__(self, message: Any) -> None:
        self.message = message
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class FakeOpenAIClient:
    def __init__(self, message: Any) -> None:
        self.completions = FakeCompletions(message)
        self.chat = SimpleNamespace(completions=self.completions)


def test_deepseek_adapter_uses_chat_completions_and_parses_tool_calls() -> None:
    tool_call = SimpleNamespace(
        id="call-7",
        function=SimpleNamespace(name="semantic_search", arguments='{"query":"x"}'),
    )
    client = FakeOpenAIClient(
        SimpleNamespace(content=None, tool_calls=[tool_call])
    )
    model = DeepSeekChatModel("test-key", client=client)

    reply = model.complete(
        [{"role": "user", "content": "research"}],
        [{"type": "function", "function": {"name": "semantic_search"}}],
    )

    assert reply.tool_calls[0].name == "semantic_search"
    assert client.completions.kwargs["model"] == "deepseek-v4-pro"
    assert client.completions.kwargs["tool_choice"] == "auto"
    assert client.completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_deepseek_adapter_omits_tool_parameters_for_text_only_reasoning() -> None:
    client = FakeOpenAIClient(SimpleNamespace(content="structured", tool_calls=[]))
    model = DeepSeekChatModel("test-key", client=client)

    reply = model.complete([{"role": "user", "content": "synthesize"}], [])

    assert reply.content == "structured"
    assert "tools" not in client.completions.kwargs
    assert "tool_choice" not in client.completions.kwargs


def test_deepseek_from_env_uses_safe_defaults(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    client = FakeOpenAIClient(SimpleNamespace(content="done", tool_calls=[]))

    model = DeepSeekChatModel.from_env(env_file, client=client)

    assert model.model_name == "deepseek-v4-pro"
