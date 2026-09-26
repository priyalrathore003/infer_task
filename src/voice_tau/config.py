"""Run configuration: which STT / LLM / TTS the agent uses, which prompt, how to evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ComponentCfg(BaseModel):
    provider: str
    model: str | None = None
    voice: str | None = None
    language: str | None = "en"
    kwargs: dict[str, Any] = Field(default_factory=dict)  # passed straight to the plugin ctor


class AudioLoopCfg(BaseModel):
    """Half-duplex audio loopback (no realtime API needed).

    user text --user_tts--> audio --agent STT--> transcript --> agent LLM
    agent text --agent TTS--> audio (synthesised + timed; optionally saved)

    Off = pure text ground truth. On = same tau2 grading, but the agent hears the
    user through a real ASR pass, so ASR errors on IDs/emails/zips show up in the score.
    """

    enabled: bool = False
    user_tts: ComponentCfg = ComponentCfg(provider="openai", model="gpt-4o-mini-tts", voice="coral")
    synthesize_agent: bool = True  # run agent replies through its TTS (latency/cost realism)
    save_audio_dir: str | None = None


class RunCfg(BaseModel):
    name: str = "baseline"
    prompt: str = "baseline"  # src/voice_tau/prompts/<prompt>.md
    llm: ComponentCfg = ComponentCfg(provider="openai", model="gpt-4.1-mini", kwargs={"temperature": 0.0})
    stt: ComponentCfg = ComponentCfg(provider="deepgram", model="nova-3")
    tts: ComponentCfg = ComponentCfg(provider="openai", model="gpt-4o-mini-tts", voice="ash")
    audio_loop: AudioLoopCfg = AudioLoopCfg()

    # tau2 side
    domain: Literal["retail", "airline", "mock"] = "retail"
    task_split: str | None = "test"  # retail: train(74) / test(40) / base(114)
    task_ids: list[str] | None = None
    num_tasks: int | None = None
    num_trials: int = 1  # >1 gives pass^k
    user_llm: str = "openai/gpt-4.1-mini"
    user_llm_args: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    max_steps: int = 100
    max_concurrency: int = 4
    seed: int = 300

    # bridge
    max_tool_steps: int = 25  # LiveKit caps consecutive tool calls per turn (default 3 is too low)
    turn_timeout_s: float = 120.0

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> "RunCfg":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)


PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text()
