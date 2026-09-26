"""--audio: STT/TTS plugins must get a live http session on every turn.

Plugins like deepgram.STT fetch their aiohttp session from LiveKit's http_context on
first use and cache it. The bridge runs outside a job worker, so it must bind one session
for its whole lifetime: unbound -> "outside of a job context" on turn 1; bound per call
-> "Session is closed" on turn 2. The fakes below handle the session the same way, with
no network.
"""

import pytest
from livekit.agents import stt, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import http_context
from tau2.data_model.message import UserMessage
from tau2.runner import build_environment

from voice_tau import bridge
from voice_tau.config import RunCfg

from .fake_llm import ScriptedLLM

LAST_SPOKEN = {"text": ""}


def _use_session(plugin) -> None:
    """Same pattern as deepgram.STT._ensure_session(), plus the failure a closed one gives."""
    if plugin._session is None:
        plugin._session = http_context.http_session()  # raises outside a bound context
    if plugin._session.closed:
        raise RuntimeError("Session is closed")


class FakeTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=16000, num_channels=1)
        self._session = None

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return _FakeChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _FakeChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter) -> None:
        _use_session(self._tts)
        LAST_SPOKEN["text"] = self._input_text
        output_emitter.initialize(request_id="fake", sample_rate=16000, num_channels=1, mime_type="audio/pcm")
        output_emitter.push(b"\0\0" * 1600)  # 100 ms of silence
        output_emitter.flush()


class FakeSTT(stt.STT):
    """'Transcribes' by returning whatever the last TTS call spoke."""

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._session = None

    async def _recognize_impl(self, buffer, *, language, conn_options):
        _use_session(self)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text=LAST_SPOKEN["text"])],
        )


@pytest.fixture
def audio_agent(monkeypatch):
    plugins: list = []

    def make(cls):
        def build(_cfg):
            plugins.append(cls())
            return plugins[-1]

        return build

    monkeypatch.setattr(bridge, "build_stt", make(FakeSTT))
    monkeypatch.setattr(bridge, "build_tts", make(FakeTTS))
    env = build_environment("retail")
    cfg = RunCfg()
    cfg.audio_loop.enabled = True
    cfg.audio_loop.synthesize_agent = True
    agent = bridge.LiveKitTauAgent(env.get_tools(), env.get_policy(), cfg)
    agent._llm_override = ScriptedLLM([{"text": f"reply {i}"} for i in range(3)])
    return agent, plugins


def test_one_session_serves_every_turn_and_is_closed_at_teardown(audio_agent):
    agent, plugins = audio_agent
    state = agent.get_init_state()
    try:
        for i in range(3):  # the per-call version failed from turn 2 on
            said = f"turn {i}: my order is W2378156"
            msg, state = agent.generate_next_message(UserMessage.text(content=said), state)
            assert msg.content == f"reply {i}"
            assert msg.raw_data["asr"]["heard"] == said  # user TTS -> agent STT ran
            assert "tts" in msg.raw_data  # agent TTS ran
        assert len(plugins) == 3  # agent STT, agent TTS, user TTS
        assert len({id(p._session) for p in plugins}) == 1  # all share one session
        assert not plugins[0]._session.closed
    finally:
        agent.stop()
    assert plugins[0]._session.closed  # _aclose closed it
