"""Full tau2 pipeline (registry -> orchestrator -> evaluator) with the LiveKit bridge.

Scripted agent LLM replays the gold actions of retail task 33 (DB-only reward);
scripted user LLM answers and then stops. Proves a correct LiveKit agent scores 1.0
and a wrong one scores 0.0 — i.e. the grading path is wired correctly. No API keys.
"""

import json

import pytest
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.runner import get_tasks, run_single_task

from voice_tau import bridge
from voice_tau.config import RunCfg
from voice_tau.register import register

from .fake_llm import ScriptedLLM

TASK = "33"


def _gold_script(task, *, corrupt=False):
    steps = []
    for a in task.evaluation_criteria.actions:
        args = dict(a.arguments)
        if corrupt and a.name == "modify_user_address":
            args["zip"] = "98196"  # the classic voice failure: one misheard digit
        steps.append({"tool": a.name, "args": args})
    steps.append({"text": "Your default address is updated to Seattle. Anything else?"})
    return steps


@pytest.fixture
def scripted_user(monkeypatch):
    lines = iter(
        [
            "Hi, I'm Noah Patel, zip 10108. Please change my default address to 517 Lakeview Drive Suite 183, Seattle WA 98195, USA. Yes, I confirm.",
            "Thanks! ###STOP###",
        ]
    )

    def fake_generate(*_, **__):
        return AssistantMessage.text(content=next(lines))

    monkeypatch.setattr("tau2.user.user_simulator.generate", fake_generate)


@pytest.mark.parametrize("corrupt,expected", [(False, 1.0), (True, 0.0)])
def test_task33_graded_by_tau2(scripted_user, monkeypatch, corrupt, expected):
    task = get_tasks("retail", task_ids=[TASK])[0]
    cfg = RunCfg(name=f"e2e_{int(corrupt)}")
    name = register(cfg)

    orig = bridge.LiveKitTauAgent._astart

    async def _astart(self, history):
        self._llm_override = ScriptedLLM(_gold_script(task, corrupt=corrupt))
        await orig(self, history)

    monkeypatch.setattr(bridge.LiveKitTauAgent, "_astart", _astart)

    run = run_single_task(
        TextRunConfig(domain="retail", agent=name, llm_agent="scripted", llm_user="scripted", max_steps=40),
        task,
        seed=1,
    )
    tool_calls = [tc.name for m in run.messages if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    print(json.dumps({"reward": run.reward_info.reward, "tools": tool_calls, "term": run.termination_reason}))
    assert tool_calls == [a.name for a in task.evaluation_criteria.actions]
    assert run.reward_info.reward == expected
