"""Bridge: run a real LiveKit AgentSession as a tau2 HalfDuplexAgent.

Why a bridge instead of calling the LLM directly: the thing under test is the
LiveKit Agent (its instructions, tool loop, chat-context handling, plugin LLM),
not a bare LLM call. Why not let LiveKit execute tools itself: tau2 grades the
*trajectory* (tool calls it saw) and the *final DB state* of its own env, so tool
calls must round-trip through tau2's orchestrator.

Flow per tau2 step (sync, called from tau2's worker thread):

    UserMessage ─► session.run(user_input=…) on a private asyncio loop
                    │
                    ├─ LLM calls tool X ─► proxy tool parks on a Future,
                    │                     bridge returns AssistantMessage(tool_calls=[X])
    ToolMessage ─►  resolve Future ─► LiveKit feeds result back to LLM ─► …
                    │
                    └─ run finishes with text ─► AssistantMessage(content=text)
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from livekit.agents import AgentSession
from livekit.agents.llm import ToolError
from livekit.agents.voice.run_result import ChatMessageEvent, RunResult
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from . import audio_loop
from .agent import build_agent
from .config import RunCfg
from .pipeline import build_llm, build_stt, build_tts

TOOL_BATCH_GRACE_S = 0.05  # collect parallel tool calls emitted in the same LLM step


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self.loop.run_forever, daemon=True, name="lk-bridge")
        self._t.start()

    def run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._t.join(timeout=5)


class _OrchestratorBackend:
    """Tool backend that hands every call to tau2 and waits for its ToolMessage."""

    def __init__(self) -> None:
        self.events: asyncio.Queue = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}

    async def call(self, name: str, arguments: dict[str, Any], call_id: str) -> str:
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[call_id] = fut
        await self.events.put(("tool", ToolCall(id=call_id, name=name, arguments=arguments)))
        content, is_error = await fut
        if is_error:
            raise ToolError(content)
        return content

    def resolve(self, msg: ToolMessage) -> None:
        fut = self.pending.pop(msg.id, None)
        if fut is None:
            raise RuntimeError(f"tau2 returned result for unknown tool call {msg.id}")
        fut.set_result((msg.content or "", msg.error))


class BridgeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    turns: int = 0
    transcripts: list[dict] = Field(default_factory=list)  # audio-loop ASR log


class LiveKitTauAgent(HalfDuplexAgent[BridgeState]):
    def __init__(self, tools: list, domain_policy: str, cfg: RunCfg, task_id: str | None = None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.cfg = cfg
        self.task_id = task_id or "task"
        self._lt: _LoopThread | None = None
        self._backend: _OrchestratorBackend | None = None
        self._session: AgentSession | None = None
        self._run: RunResult | None = None
        self._cursor = 0  # index into self._run.events already consumed
        self._stt = self._tts = self._user_tts = None
        self._llm_override = None  # tests inject a fake LLM here

    # ------------------------------------------------------------------ tau2 API
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> BridgeState:
        history = [
            (m.role, m.content)
            for m in message_history or []
            if isinstance(m, (UserMessage, AssistantMessage)) and m.content
        ]
        self._lt = _LoopThread()
        self._lt.run(self._astart(history), timeout=60)
        return BridgeState()

    def generate_next_message(
        self, message: UserMessage | ToolMessage | MultiToolMessage, state: BridgeState
    ) -> tuple[AssistantMessage, BridgeState]:
        t0 = time.perf_counter()
        raw: dict[str, Any] = {}
        if isinstance(message, UserMessage):
            text = message.content or ""
            if self.cfg.audio_loop.enabled and text:
                text, asr = self._lt.run(self._hear(text, state.turns), timeout=self.cfg.turn_timeout_s)
                raw["asr"] = asr
                state.transcripts.append(asr)
            step = self._lt.run(self._user_turn(text), timeout=self.cfg.turn_timeout_s)
        else:
            msgs = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
            step = self._lt.run(self._tool_results(msgs), timeout=self.cfg.turn_timeout_s)

        kind, payload, extra = step
        raw.update(extra)
        if kind == "tools":
            out = AssistantMessage.text(content=None, tool_calls=payload, raw_data=raw or None)
        else:
            text = payload
            if not text.strip():
                raw["empty_reply"] = True
                text = "..."
            if self.cfg.audio_loop.enabled and self.cfg.audio_loop.synthesize_agent:
                raw["tts"] = self._lt.run(self._speak(text, state.turns), timeout=self.cfg.turn_timeout_s)
            out = AssistantMessage.text(content=text, raw_data=raw or None)
        out.generation_time_seconds = time.perf_counter() - t0
        state.turns += 1
        return out, state

    def stop(self, message=None, state=None) -> None:
        if self._lt is None:
            return
        try:
            self._lt.run(self._aclose(), timeout=15)
        except Exception as e:  # never fail the sim on teardown
            logger.warning(f"bridge teardown: {e}")
        self._lt.close()
        self._lt = None

    # ------------------------------------------------------------- async side
    async def _astart(self, history: list[tuple[str, str]]) -> None:
        self._backend = _OrchestratorBackend()
        llm = self._llm_override or build_llm(self.cfg.llm)
        agent = build_agent(
            self.cfg,
            tau_tools=self.tools,
            domain_policy=self.domain_policy,
            backend=self._backend,
            history=history,
        )
        # text-mode session: no room, no VAD. STT/TTS are exercised by audio_loop instead.
        self._session = AgentSession(llm=llm, max_tool_steps=self.cfg.max_tool_steps, user_away_timeout=None)
        await self._session.start(agent, record=False)
        if self.cfg.audio_loop.enabled:
            self._stt = build_stt(self.cfg.stt)
            self._tts = build_tts(self.cfg.tts)
            self._user_tts = build_tts(self.cfg.audio_loop.user_tts)

    async def _aclose(self) -> None:
        for fut in self._backend.pending.values():
            if not fut.done():
                fut.set_result(("conversation ended", True))
        if self._session is not None:
            await self._session.aclose()

    async def _user_turn(self, text: str):
        self._run = self._session.run(user_input=text)
        self._cursor = 0
        run = self._run

        async def _watch():
            try:
                await run
                await self._backend.events.put(("done", None))
            except Exception as e:
                await self._backend.events.put(("error", e))

        asyncio.create_task(_watch())
        return await self._next_step()

    async def _tool_results(self, msgs: list[ToolMessage]):
        for m in msgs:
            self._backend.resolve(m)
        return await self._next_step()

    async def _next_step(self):
        q = self._backend.events
        kind, val = await q.get()
        if kind == "tool":
            calls = [val]
            await asyncio.sleep(TOOL_BATCH_GRACE_S)
            while not q.empty():
                k2, v2 = q.get_nowait()
                if k2 == "tool":
                    calls.append(v2)
                else:
                    q.put_nowait((k2, v2))
                    break
            spoken = self._drain_text()
            # tau2 forbids text+tool in one message; keep what the agent "said" for analysis
            return "tools", calls, ({"spoken_with_tool_call": spoken} if spoken else {})
        if kind == "error":
            raise val
        return "text", self._drain_text(), {}

    def _drain_text(self) -> str:
        evs = self._run.events if self._run else []
        parts = [
            e.item.text_content
            for e in evs[self._cursor :]
            if isinstance(e, ChatMessageEvent) and e.item.role == "assistant" and e.item.text_content
        ]
        self._cursor = len(evs)
        return " ".join(p.strip() for p in parts).strip()

    async def _hear(self, text: str, turn: int):
        u = await audio_loop.synthesize(self._user_tts, text)
        heard = await audio_loop.transcribe(self._stt, u.frame, self.cfg.stt.language)
        self._maybe_save(u.frame, f"t{turn:02d}_user.wav")
        return heard.text, {
            "sent": text,
            "heard": heard.text,
            "user_tts_s": round(u.latency_s, 3),
            "stt_s": round(heard.latency_s, 3),
            "audio_s": round(u.audio_s, 2),
        }

    async def _speak(self, text: str, turn: int):
        r = await audio_loop.synthesize(self._tts, text)
        self._maybe_save(r.frame, f"t{turn:02d}_agent.wav")
        return {"tts_s": round(r.latency_s, 3), "audio_s": round(r.audio_s, 2)}

    def _maybe_save(self, frame, name: str) -> None:
        d = self.cfg.audio_loop.save_audio_dir
        if d:
            audio_loop.save_wav(frame, Path(d) / self.cfg.name / self.task_id / name)
