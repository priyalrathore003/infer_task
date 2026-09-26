"""Summarise a tau2 results.json into metrics + a failure taxonomy.

tau2 gives you the reward. This adds the *why*, in categories that map to prompt
changes you can actually make (the assignment's "improve via prompt changes" loop).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any

WRITE_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
YES = re.compile(r"\b(yes|yeah|yep|confirm|go ahead|proceed|sure|correct|please do)\b", re.I)


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def classify(sim: dict[str, Any], task: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    msgs = sim.get("messages") or []
    ri = sim.get("reward_info") or {}
    authed = False
    last_user = ""
    gold = {a["name"] for a in (task.get("evaluation_criteria") or {}).get("actions") or []}
    for m in msgs:
        role = m.get("role")
        if role == "user" and m.get("content"):
            last_user = m["content"]
        calls = (m.get("tool_calls") or []) if role == "assistant" else []
        if len(calls) > 1:
            tags.append("multi_tool_call")
        for c in calls:
            n = c["name"]
            if n in AUTH_TOOLS:
                authed = True
            if n in WRITE_TOOLS:
                if not authed:
                    tags.append("write_before_auth")
                if not YES.search(last_user):
                    tags.append("write_without_confirmation")
                if n not in gold:
                    tags.append("unexpected_write_tool")
            if n == "transfer_to_human_agents" and "transfer_to_human_agents" not in gold:
                tags.append("unneeded_transfer")
        raw = m.get("raw_data") or {}
        if raw.get("empty_reply"):
            tags.append("empty_reply")
        if raw.get("spoken_with_tool_call"):
            tags.append("text_with_tool_call")
    for ac in ri.get("action_checks") or []:
        if not ac.get("action_match"):
            tags.append(f"missed_action:{ac['action']['name']}")
    db = ri.get("db_check") or {}
    if db and not db.get("db_match"):
        tags.append("db_mismatch")
    for nl in ri.get("nl_assertions") or []:
        if not nl.get("met"):
            tags.append("nl_assertion_failed")
    term = sim.get("termination_reason")
    if term not in (None, "user_stop", "agent_stop"):
        tags.append(f"terminated:{term}")
    return sorted(set(tags))


def pass_hat_k(results: dict[str, list[float]], k: int) -> float | None:
    vals = []
    for rs in results.values():
        n, c = len(rs), sum(r >= 1.0 for r in rs)
        if n < k:
            return None
        vals.append(comb(c, k) / comb(n, k))
    return mean(vals) if vals else None


def summarize(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    tasks = {t["id"]: t for t in data["tasks"]}
    by_task: dict[str, list[float]] = defaultdict(list)
    tags = Counter()
    rows = []
    gen_times, wers = [], []
    for s in data["simulations"]:
        r = (s.get("reward_info") or {}).get("reward", 0.0) or 0.0
        by_task[s["task_id"]].append(r)
        t = classify(s, tasks.get(s["task_id"], {}))
        if r < 1.0:
            tags.update(t)
        for m in s.get("messages") or []:
            if m.get("role") == "assistant" and m.get("generation_time_seconds"):
                gen_times.append(m["generation_time_seconds"])
            asr = (m.get("raw_data") or {}).get("asr")
            if asr:
                wers.append(_wer(asr["sent"], asr["heard"]))
        rows.append({"task_id": s["task_id"], "trial": s.get("trial"), "reward": r, "tags": t})
    max_k = min(len(v) for v in by_task.values()) if by_task else 0
    return {
        "n_sims": len(rows),
        "n_tasks": len(by_task),
        "avg_reward": mean(r["reward"] for r in rows) if rows else 0.0,
        **{f"pass^{k}": pass_hat_k(by_task, k) for k in range(1, max_k + 1)},
        "agent_step_latency_s": {"p50": median(gen_times), "max": max(gen_times)} if gen_times else None,
        "asr_wer_mean": mean(wers) if wers else None,
        "failure_tags": dict(tags.most_common()),
        "per_sim": rows,
    }
