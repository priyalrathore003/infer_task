"""Plugin factory: turn a ComponentCfg into a LiveKit STT / LLM / TTS instance.

Add a provider = add one branch. Plugins are imported lazily so you only need
the packages for the providers you actually use.
"""

from __future__ import annotations

from typing import Any

from livekit.agents import llm as lk_llm
from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts

from .config import ComponentCfg


def _kw(cfg: ComponentCfg, **extra: Any) -> dict[str, Any]:
    out = {k: v for k, v in extra.items() if v is not None}
    out.update(cfg.kwargs)
    return out


def build_llm(cfg: ComponentCfg) -> lk_llm.LLM:
    p = cfg.provider
    if p == "openai":
        from livekit.plugins import openai

        # tau2 retail policy: at most one tool call at a time
        return openai.LLM(**_kw(cfg, model=cfg.model, parallel_tool_calls=False))
    if p in ("groq", "cerebras", "together", "fireworks", "deepseek", "openrouter"):
        from livekit.plugins import openai

        factory = getattr(openai.LLM, f"with_{p}")
        return factory(**_kw(cfg, model=cfg.model))
    if p == "anthropic":
        from livekit.plugins import anthropic

        return anthropic.LLM(**_kw(cfg, model=cfg.model))
    if p == "google":
        from livekit.plugins import google

        return google.LLM(**_kw(cfg, model=cfg.model))
    raise ValueError(f"unknown llm provider {p!r}")


def build_stt(cfg: ComponentCfg) -> lk_stt.STT:
    p = cfg.provider
    if p == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.STT(**_kw(cfg, model=cfg.model, language=cfg.language))
    if p == "openai":
        from livekit.plugins import openai

        return openai.STT(**_kw(cfg, model=cfg.model or "gpt-4o-mini-transcribe", language=cfg.language))
    if p == "groq":
        from livekit.plugins import openai

        return openai.STT.with_groq(**_kw(cfg, model=cfg.model or "whisper-large-v3-turbo"))
    raise ValueError(f"unknown stt provider {p!r}")


def build_tts(cfg: ComponentCfg) -> lk_tts.TTS:
    p = cfg.provider
    if p == "openai":
        from livekit.plugins import openai

        return openai.TTS(**_kw(cfg, model=cfg.model, voice=cfg.voice))
    if p == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.TTS(**_kw(cfg, model=cfg.model or "aura-2-thalia-en"))
    if p == "cartesia":
        from livekit.plugins import cartesia

        return cartesia.TTS(**_kw(cfg, model=cfg.model, voice=cfg.voice))
    raise ValueError(f"unknown tts provider {p!r}")
