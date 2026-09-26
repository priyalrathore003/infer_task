"""Half-duplex audio loopback using the pipeline's real STT/TTS plugins.

No realtime / speech-to-speech API is involved: each turn is one batch TTS call and
one batch STT call. This puts the agent's STT (the component most likely to break
tool-calling — misheard order IDs, emails, zips) in the loop while keeping tau2's
turn-based grading intact.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

from livekit import rtc
from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts


@dataclass
class TTSResult:
    frame: rtc.AudioFrame
    latency_s: float
    audio_s: float


@dataclass
class STTResult:
    text: str
    latency_s: float


async def synthesize(tts: lk_tts.TTS, text: str) -> TTSResult:
    t0 = time.perf_counter()
    frame = await tts.synthesize(text).collect()
    return TTSResult(frame=frame, latency_s=time.perf_counter() - t0, audio_s=frame.duration)


async def transcribe(stt: lk_stt.STT, frame: rtc.AudioFrame, language: str | None = None) -> STTResult:
    t0 = time.perf_counter()
    kw = {"language": language} if language else {}
    ev = await stt.recognize(frame, **kw)
    text = ev.alternatives[0].text if ev.alternatives else ""
    return STTResult(text=text.strip(), latency_s=time.perf_counter() - t0)


def save_wav(frame: rtc.AudioFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(frame.num_channels)
        w.setsampwidth(2)
        w.setframerate(frame.sample_rate)
        w.writeframes(bytes(frame.data))
