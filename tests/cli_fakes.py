"""Stand-ins for tau2's run_domain() so `voice-tau run` can be tested without API calls."""

from __future__ import annotations

import json
from pathlib import Path


def write_fake_results(data_dir: Path, run_name: str, terminations: list[str]) -> Path:
    """Write a minimal tau2-shaped results.json where tau2 would have saved it."""
    sims = [
        {
            "task_id": str(i),
            "trial": 0,
            "termination_reason": term,
            "reward_info": None if term == "infrastructure_error" else {"reward": 1.0},
            "messages": [],
            "info": {"error": "fake"} if term == "infrastructure_error" else {},
        }
        for i, term in enumerate(terminations)
    ]
    out = data_dir / "simulations" / run_name / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tasks": [{"id": s["task_id"]} for s in sims], "simulations": sims}))
    return out
