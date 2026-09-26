"""Register a RunCfg with tau2's registry as agent 'livekit:<cfg.name>'."""

from __future__ import annotations

from tau2.registry import registry

from .bridge import LiveKitTauAgent
from .config import RunCfg


def register(cfg: RunCfg) -> str:
    name = f"livekit_{cfg.name}"

    def factory(tools, domain_policy, task=None, **_):
        return LiveKitTauAgent(tools, domain_policy, cfg, task_id=getattr(task, "id", None))

    if name not in registry.get_agents():
        registry.register_agent_factory(factory, name)
    return name
