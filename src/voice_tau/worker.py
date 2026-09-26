"""Live voice mode: the SAME agent, in a real cascaded LiveKit session (for the demo).

    python -m voice_tau.worker console      # talk to it from your terminal mic/speaker
    python -m voice_tau.worker dev          # join a LiveKit room (needs LIVEKIT_URL/KEY/SECRET)

Env var VOICE_TAU_CONFIG picks the config (default configs/baseline.yaml).
Tools run directly on a fresh tau2 retail DB, so you can read a tau2 task's user
instructions aloud and afterwards diff the DB against the task's gold actions.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli
from livekit.plugins import silero

from .agent import DirectEnvBackend, build_agent
from .config import RunCfg
from .pipeline import build_llm, build_stt, build_tts

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    from tau2.runner import build_environment

    cfg = RunCfg.load(os.environ.get("VOICE_TAU_CONFIG", "configs/baseline.yaml"))
    env = build_environment(cfg.domain)
    agent = build_agent(
        cfg,
        tau_tools=env.get_tools(),
        domain_policy=env.get_policy(),
        backend=DirectEnvBackend(env),
        greeting="Hi! How can I help you today?",  # same opener tau2 uses
    )
    session = AgentSession(
        stt=build_stt(cfg.stt),
        llm=build_llm(cfg.llm),
        tts=build_tts(cfg.tts),
        vad=silero.VAD.load(),
        max_tool_steps=cfg.max_tool_steps,
    )
    await session.start(agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
