"""A scripted LiveKit LLM so the bridge can be tested with zero API keys."""

from __future__ import annotations

import json
import uuid

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS


class ScriptedLLM(llm.LLM):
    """Each chat() call pops the next scripted step: {"text": ...} or {"tool": name, "args": {...}}."""

    def __init__(self, script: list[dict]) -> None:
        super().__init__()
        self.script = list(script)
        self.seen_ctx: list[llm.ChatContext] = []

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_):
        self.seen_ctx.append(chat_ctx.copy())
        step = self.script.pop(0) if self.script else {"text": "Is there anything else?"}
        return _Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options, step=step)


class _Stream(llm.LLMStream):
    def __init__(self, llm_, *, step, **kw) -> None:
        super().__init__(llm_, **kw)
        self._step = step

    async def _run(self) -> None:
        rid = uuid.uuid4().hex
        if "tool" in self._step:
            call = llm.FunctionToolCall(
                name=self._step["tool"],
                arguments=json.dumps(self._step.get("args", {})),
                call_id=f"call_{uuid.uuid4().hex[:8]}",
            )
            delta = llm.ChoiceDelta(role="assistant", tool_calls=[call])
        else:
            delta = llm.ChoiceDelta(role="assistant", content=self._step["text"])
        self._event_ch.send_nowait(llm.ChatChunk(id=rid, delta=delta))
