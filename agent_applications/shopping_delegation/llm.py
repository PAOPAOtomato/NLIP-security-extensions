"""Build the LLM behind each agent; tests substitute a scripted stand-in."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.tools import BaseTool

from nlip_examples.nlip_agnostic_agents.langchain_agent import LangChainTestAgent


class ChatAgent(Protocol):
    async def chat(self, prompt: str) -> str: ...


# (system prompt, tools) -> a stateful agent for one conversation.
AgentFactory = Callable[[str, Sequence[BaseTool]], ChatAgent]


def langchain_agent_factory(entity_settings: dict[str, Any]) -> AgentFactory:
    """Create LangChain agents with the model configured for one entity."""

    def build(system_prompt: str, tools: Sequence[BaseTool]) -> ChatAgent:
        return LangChainTestAgent(
            model=entity_settings["model"],
            use_responses_api=entity_settings.get("use_responses_api"),
            system_prompt=system_prompt,
            tools=tools,
        )

    return build
