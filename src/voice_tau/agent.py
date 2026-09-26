"""The system under test: a LiveKit cascaded voice Agent for tau2 retail.

The same `build_agent()` is used in two places:
  * eval  (bridge.py)  — tool calls are handed to tau2's orchestrator, which executes
                          them against its env so tau2 can grade DB state + actions.
  * live  (worker.py)  — tool calls execute directly on a tau2 Environment inside a
                          real LiveKit room with STT/VAD/TTS (for the Loom demo).

Tools are generated from tau2's own OpenAI schemas, so the agent sees exactly the
tool surface tau2 grades against.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents import llm as lk_llm
from livekit.agents.llm import ToolError

from .config import RunCfg, load_prompt


class ToolBackend(Protocol):
    async def call(self, name: str, arguments: dict[str, Any], call_id: str) -> str:
        """Return tool output text; raise ToolError on tool failure."""


def _make_tool(schema: dict[str, Any], backend: ToolBackend):
    fn = schema["function"] if "function" in schema else schema
    raw = {"name": fn["name"], "description": fn.get("description", ""), "parameters": fn["parameters"]}

    async def _tool(raw_arguments: dict[str, Any], context: RunContext) -> str:
        return await backend.call(fn["name"], dict(raw_arguments), context.function_call.call_id)

    return function_tool(_tool, raw_schema=raw)


class RetailVoiceAgent(Agent):
    """Thin subclass so live-mode hooks (greeting, llm_node tweaks) have a home."""

    def __init__(self, *, greeting: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._greeting = greeting

    async def on_enter(self) -> None:
        # live mode only: speak the greeting. In eval mode the greeting is already in
        # chat_ctx (tau2 sends it as turn 0) and we pass greeting=None.
        if self._greeting:
            await self.session.say(self._greeting, add_to_chat_ctx=True)


def build_agent(
    cfg: RunCfg,
    *,
    tau_tools: list,  # list[tau2.environment.tool.Tool]
    domain_policy: str,
    backend: ToolBackend,
    history: list[tuple[str, str]] | None = None,  # (role, text) to pre-seed
    greeting: str | None = None,
    **agent_kw: Any,
) -> RetailVoiceAgent:
    instructions = load_prompt(cfg.prompt).replace("{domain_policy}", domain_policy)
    tools = [_make_tool(t.openai_schema, backend) for t in tau_tools]
    ctx = lk_llm.ChatContext.empty()
    for role, text in history or []:
        ctx.add_message(role=role, content=text)
    return RetailVoiceAgent(instructions=instructions, tools=tools, chat_ctx=ctx, greeting=greeting, **agent_kw)


class DirectEnvBackend:
    """Executes tools straight on a tau2 Environment (live/demo mode)."""

    def __init__(self, env) -> None:
        self.env = env

    async def call(self, name: str, arguments: dict[str, Any], call_id: str) -> str:
        from tau2.data_model.message import ToolCall

        msg = self.env.get_response(ToolCall(id=call_id, name=name, arguments=arguments))
        if msg.error:
            raise ToolError(msg.content or "error")
        return msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
