"""`voice-tau run` must register LiveKit plugins on the main thread before tau2's workers start.

LiveKit raises "Plugins must be registered on the main thread" when a plugin module is
first imported on another thread, and the bridge builds STT/LLM/TTS on worker threads.
The earlier offline tests missed this because they inject ScriptedLLM and never call
build_llm().

Plugin registration is process-wide and one-way, so each case runs `voice-tau run` in a
fresh subprocess. tau2's run_domain() is replaced with a stand-in that does what tau2's
workers do first: build the configured STT/LLM/TTS on a non-main thread.

The config uses cartesia for TTS because tau2's preregister_livekit_plugins() never
imports cartesia, so the cartesia case exercises part (2) of the fix,
_preimport_configured_plugins().
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN_THREAD_ERR = "Plugins must be registered on the main thread"

if not (ROOT / "vendor" / "tau2-bench").exists():  # pragma: no cover
    pytest.skip("vendor/tau2-bench not cloned", allow_module_level=True)

CONFIG = """\
name: plugtest
llm: {provider: openai, model: gpt-4.1-mini}
stt: {provider: deepgram, model: nova-3}
tts: {provider: cartesia}
"""

SCRIPT = textwrap.dedent(
    """
    import json, sys, threading
    from pathlib import Path

    sys.path.insert(0, {root!r})
    import tau2.runner
    import tau2.utils.utils as tu
    from tests.cli_fakes import write_fake_results
    from voice_tau import cli
    from voice_tau.config import RunCfg
    from voice_tau.pipeline import build_llm, build_stt, build_tts

    mode, cfg_path, tmp = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    tu.DATA_DIR = tmp  # run() reads this at call time
    if mode == "no_prereg":
        cli._preregister_plugins = lambda cfg: None
    elif mode == "tau2_only":
        cli._preimport_configured_plugins = lambda cfg: None

    def fake_run_domain(config):
        cfg = RunCfg.load(cfg_path)
        errors = {{}}

        def worker():  # what the bridge does on tau2's worker thread
            for build, comp in ((build_llm, cfg.llm), (build_stt, cfg.stt), (build_tts, cfg.tts)):
                try:
                    build(comp)
                    errors[comp.provider] = None
                except Exception as e:
                    errors[comp.provider] = f"{{type(e).__name__}}: {{e}}"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        print("WORKER=" + json.dumps(errors), flush=True)
        write_fake_results(tu.DATA_DIR, config.save_to, ["user_stop"])

    tau2.runner.run_domain = fake_run_domain
    cli.app(["run", cfg_path], standalone_mode=False)
    """
).format(root=str(ROOT))


def _run(mode: str, tmp_path: Path) -> dict:
    cfg = tmp_path / "plugtest.yaml"
    cfg.write_text(CONFIG)
    env = {**os.environ, "OPENAI_API_KEY": "x", "DEEPGRAM_API_KEY": "x", "CARTESIA_API_KEY": "x"}
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, mode, str(cfg), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180,
    )
    line = next((l for l in p.stdout.splitlines() if l.startswith("WORKER=")), None)
    assert line, f"worker never ran (exit {p.returncode})\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    return json.loads(line.removeprefix("WORKER="))


def test_run_registers_all_configured_plugins_before_workers(tmp_path):
    errors = _run("full", tmp_path)
    assert errors == {"openai": None, "deepgram": None, "cartesia": None}


def test_without_preregistration_worker_build_fails(tmp_path):
    """Control: proves the test detects the original bug rather than passing vacuously."""
    errors = _run("no_prereg", tmp_path)
    assert MAIN_THREAD_ERR in (errors["openai"] or "")


def test_tau2_preregister_alone_misses_cartesia(tmp_path):
    """Control for part (2): tau2's helper covers openai/deepgram but not cartesia."""
    errors = _run("tau2_only", tmp_path)
    assert errors["openai"] is None and errors["deepgram"] is None
    assert MAIN_THREAD_ERR in (errors["cartesia"] or "")

