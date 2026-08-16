"""Composition root for the default DeepSeek + SciVerse research agent."""

from __future__ import annotations

from pathlib import Path

from research_gap_agent.tools import build_sciverse_tool_registry

from .deepseek import DeepSeekChatModel
from .runtime import DeepResearchAgent


def build_deep_research_agent(
    *,
    env_file: str | Path = ".env",
    server_command: str | Path | None = None,
    max_tool_rounds: int = 6,
) -> DeepResearchAgent:
    model = DeepSeekChatModel.from_env(env_file)
    registry = build_sciverse_tool_registry(
        env_file=env_file,
        server_command=server_command,
    )
    return DeepResearchAgent(
        model,
        registry,
        max_tool_rounds=max_tool_rounds,
    )
