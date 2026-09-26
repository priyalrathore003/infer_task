"""voice-tau CLI.

  voice-tau run configs/baseline.yaml --task-ids 0,1,2      # smoke
  voice-tau run configs/baseline.yaml                        # full split
  voice-tau report results/baseline/results.json
  voice-tau compare results/baseline/results.json results/v2_confirm/results.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()
app = typer.Typer(add_completion=False, no_args_is_help=True)
con = Console()
RESULTS = Path("results")


_OPENAI_COMPAT = {"groq", "cerebras", "together", "fireworks", "deepseek", "openrouter"}


def _preimport_configured_plugins(cfg) -> None:
    import importlib

    comps = [cfg.llm, cfg.stt, cfg.tts, cfg.audio_loop.user_tts]
    for p in {"openai" if c.provider in _OPENAI_COMPAT else c.provider for c in comps}:
        importlib.import_module(f"livekit.plugins.{p}")  # ImportError here = missing extra


def _preregister_plugins(cfg) -> None:
    """LiveKit plugins must register on the main thread; the bridge builds STT/LLM/TTS on
    worker threads. tau2 only does this itself for VoiceRunConfig + provider=="livekit"."""
    from tau2.voice.audio_native.livekit import preregister_livekit_plugins

    preregister_livekit_plugins()  # openai + deepgram (warns if anthropic/elevenlabs absent)
    _preimport_configured_plugins(cfg)  # anything else this config uses (anthropic, google, cartesia)


def _infra_error_counts(results: Path) -> tuple[int, int]:
    """(infrastructure_error sims, total sims) in a tau2 results.json."""
    sims = json.loads(results.read_text()).get("simulations") or []
    n_infra = sum(s.get("termination_reason") == "infrastructure_error" for s in sims)
    return n_infra, len(sims)


@app.command()
def run(
    config: Path,
    task_ids: Optional[str] = typer.Option(None, help="comma-separated tau2 task ids"),
    num_tasks: Optional[int] = None,
    num_trials: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    audio: Optional[bool] = typer.Option(None, "--audio/--text", help="toggle STT/TTS loopback"),
    name: Optional[str] = typer.Option(None, help="override run name"),
):
    """Run the LiveKit agent on tau2 and save results.json."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain
    from tau2.utils.utils import DATA_DIR

    from .config import RunCfg
    from .register import register

    cfg = RunCfg.load(
        config,
        task_ids=task_ids.split(",") if task_ids else None,
        num_tasks=num_tasks,
        num_trials=num_trials,
        max_concurrency=max_concurrency,
        name=name,
    )
    if audio is not None:
        cfg.audio_loop.enabled = audio
    agent_name = register(cfg)
    _preregister_plugins(cfg)  # must happen before run_domain() spawns worker threads

    run_name = f"{cfg.name}{'_audio' if cfg.audio_loop.enabled else ''}"
    llm_label = f"{cfg.llm.provider}/{cfg.llm.model}"

    run_domain(
        TextRunConfig(
            domain=cfg.domain,
            agent=agent_name,
            llm_agent=llm_label,  # label only; the bridge builds its own LLM from cfg
            llm_user=cfg.user_llm,
            llm_args_user=cfg.user_llm_args,
            task_split_name=None if cfg.task_ids else cfg.task_split,
            task_ids=cfg.task_ids,
            num_tasks=cfg.num_tasks,
            num_trials=cfg.num_trials,
            max_steps=cfg.max_steps,
            max_concurrency=cfg.max_concurrency,
            seed=cfg.seed,
            save_to=run_name,
        )
    )
    src = DATA_DIR / "simulations" / run_name / "results.json"
    dst = RESULTS / run_name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / "results.json")
    (dst / "config.json").write_text(cfg.model_dump_json(indent=2))
    con.print(f"[green]saved[/] {dst/'results.json'}  (tau2 view: {src})")
    report(dst / "results.json")

    n_infra, n = _infra_error_counts(dst / "results.json")
    if n and n_infra == n:
        con.print(f"[red]all {n} simulations ended as infrastructure_error; see info.error in results.json[/]")
        raise typer.Exit(code=1)
    if n_infra:
        con.print(f"[yellow]{n_infra}/{n} simulations ended as infrastructure_error; rerun those task ids[/]")


@app.command()
def report(results: Path, show_sims: bool = False):
    """Metrics + failure taxonomy for one run."""
    from .report import summarize

    s = summarize(results)
    t = Table(title=str(results))
    t.add_column("metric")
    t.add_column("value")
    for k, v in s.items():
        if k in ("per_sim", "failure_tags"):
            continue
        t.add_row(k, f"{v:.3f}" if isinstance(v, float) else json.dumps(v))
    con.print(t)
    ft = Table(title="failure tags (failed sims only)")
    ft.add_column("tag")
    ft.add_column("count")
    for k, v in s["failure_tags"].items():
        ft.add_row(k, str(v))
    con.print(ft)
    if show_sims:
        for r in s["per_sim"]:
            con.print(r)
    (results.parent / "summary.json").write_text(json.dumps(s, indent=2))


@app.command()
def compare(a: Path, b: Path):
    """Per-task diff between two runs (e.g. prompt v1 vs v2)."""
    from .report import summarize

    sa, sb = summarize(a), summarize(b)
    def per_task(s):
        acc = {}
        for r in s["per_sim"]:
            acc.setdefault(r["task_id"], []).append(r["reward"])
        return {k: sum(v) / len(v) for k, v in acc.items()}

    ra, rb = per_task(sa), per_task(sb)
    fixed = sorted(k for k in ra if k in rb and ra[k] < 1 <= rb[k])
    broke = sorted(k for k in ra if k in rb and ra[k] >= 1 > rb[k])
    con.print(f"avg reward {sa['avg_reward']:.3f} -> {sb['avg_reward']:.3f}")
    con.print(f"[green]fixed ({len(fixed)})[/]: {fixed}")
    con.print(f"[red]regressed ({len(broke)})[/]: {broke}")


if __name__ == "__main__":
    app()
