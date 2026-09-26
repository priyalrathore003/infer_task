# voice-tau-eval

A **LiveKit cascaded voice agent** (pluggable STT → LLM → TTS), graded by **tau2-bench (τ³-bench) retail**.
tau2 does the grading (final DB state, gold actions, NL assertions). The agent under test is a real
LiveKit `Agent` running inside a real `AgentSession`.

```
             tau2 orchestrator (half-duplex, turn-based)
 ┌──────────────┐  UserMessage   ┌───────────────────────────────────────────────┐
 │ tau2 user    │ ─────────────► │ LiveKitTauAgent (bridge.py)                   │
 │ simulator    │ ◄───────────── │   [--audio] user_tts ─► agent STT ─► text     │
 └──────────────┘  AssistantMsg  │   AgentSession.run(text)  ◄── LiveKit Agent   │
                                 │     LLM plugin ─► proxy tool ─┐               │
 ┌──────────────┐  ToolCall      │                               │ parks on Future
 │ tau2 retail  │ ◄──────────────┼───────────────────────────────┘               │
 │ env + DB     │ ─────────────► │   resolve Future ─► LLM continues             │
 └──────────────┘  ToolMessage   └───────────────────────────────────────────────┘
        │
        ▼  evaluator: DB hash vs gold · action checks · NL assertions  ─►  reward
```

Tool calls go through tau2's orchestrator and are not run inside LiveKit. tau2 grades the trajectory and
its own DB, so if LiveKit ran the tools itself, tau2's grading would miss them.

## Modes

| mode | what's in the loop | cost | use for |
|---|---|---|---|
| `--text` (default) | LiveKit Agent + LLM plugin | LLM only | ground-truth tool-use / policy scores, prompt iteration |
| `--audio` | + user TTS → **agent STT** each turn (+ optional agent TTS) | + batch STT/TTS | measuring how ASR errors (IDs, emails, zips) hurt the score. Reports WER alongside reward |
| `worker console` / `dev` | full live STT/VAD/LLM/TTS in LiveKit | realtime | Loom demo; same agent, same tools, fresh tau2 DB |

No realtime or speech-to-speech API is used anywhere.

## Setup

Requires Python 3.12 (tau2 needs >=3.12) and [uv](https://docs.astral.sh/uv/).

```bash
cd voice-tau-eval

# 1. tau2-bench must be cloned INSIDE the project at vendor/tau2-bench.
#    pyproject.toml installs it from that path as an editable install, which is how
#    tau2 finds its data/ dir (tasks, db, policy). Installing tau2 from git instead
#    gives no data dir: tau2 warns "Data directory does not exist" and can't find tasks.
git clone https://github.com/sierra-research/tau2-bench vendor/tau2-bench
git -C vendor/tau2-bench checkout b7ea907   # the commit everything was tested on

# 2. Environment
uv venv -p 3.12
uv pip install -e ".[dev]"      # add ,anthropic / ,google / ,cartesia if your config uses them

# 3. Offline tests (no API keys). Pass `tests` explicitly: a bare `pytest` also
#    collects vendor/tau2-bench/tests and fails with 13 collection errors.
uv run pytest -q tests          # expect: 4 passed

# 4. Keys
cp .env.example .env            # OPENAI_API_KEY is required: agent LLM, user simulator, and
                                # tau2's NL-assertion judge (gpt-4.1-2025-04-14)
```

Check it works: `.venv/bin/voice-tau --help` should list `run`, `report` and `compare`.

Notes:
- `websockets` is a direct dependency because tau2 imports it at module load even in text mode. Without it, `import tau2` fails with `ModuleNotFoundError`.
- `uv run` writes a `uv.lock` on first use.
- Runs are saved by tau2 to `vendor/tau2-bench/data/simulations/<run_name>/results.json`, and `voice-tau run` copies them to `results/<run_name>/`.

## Run

```bash
voice-tau run configs/baseline.yaml --task-ids 0,1,2 --max-concurrency 1   # smoke (~cents)
voice-tau run configs/baseline.yaml                                        # retail test split, 40 tasks
voice-tau run configs/tau2_reference.yaml                                  # control: tau2's own prompt
voice-tau run configs/baseline.yaml --audio --task-ids 0,1,2               # ASR in the loop
voice-tau report results/baseline/results.json --show-sims
voice-tau compare results/tau2_reference/results.json results/baseline/results.json
tau2 view   # tau2's own trajectory browser also works on vendor/tau2-bench/data/simulations/*
```

To swap a component, edit `llm/stt/tts` in the YAML. To add a provider, add a branch in `pipeline.py`.
Prompts live in `src/voice_tau/prompts/*.md`, and `{domain_policy}` is replaced with tau2's `policy.md`.

## Improvement loop (prompt changes)

1. Run the baseline on the `train` split and read `summary.json` → `failure_tags`
   (`write_without_confirmation`, `write_before_auth`, `missed_action:*`, `db_mismatch`,
   `nl_assertion_failed`, `text_with_tool_call`, `unneeded_transfer`, …).
2. Copy the prompt to `prompts/v2_<fix>.md` and target the top tag. Add a config pointing at it.
3. Rerun on `train`, then `compare`. Report the final numbers on the held-out `test` split, with
   `num_trials: 3+` for pass^k.

## Files

- `agent.py`: the LiveKit `Agent`. Its tools are generated from tau2's OpenAI schemas.
- `bridge.py`: runs an `AgentSession` as a tau2 `HalfDuplexAgent` on a private event loop.
- `pipeline.py`: builds STT/LLM/TTS from the config.
- `audio_loop.py`: batch TTS→STT loopback for `--audio`.
- `report.py`: pass^k, latency, WER and failure taxonomy.
- `worker.py`: live mode.

## Known limits / notes for the write-up

- **tau2 ships its own cascaded LiveKit provider**: `tau2 run --audio-native --audio-native-provider livekit
  --cascaded-config default` (Deepgram STT → OpenAI/Anthropic → Deepgram/ElevenLabs TTS,
  full-duplex, tick-based). It is a real alternative for full-duplex audio numbers. It calls
  `livekit.plugins` directly rather than `Agent`/`AgentSession`, and its voice user simulator needs
  TTS on the user side too, so it costs more. This repo uses half-duplex so every score maps
  cleanly to a policy or tool decision.
- LiveKit's default `max_tool_steps=3` makes retail tasks fail silently. The config sets it to 25.
- `--audio` was built against the plugin APIs but hasn't been run with live keys yet. Smoke it on
  1–2 tasks first.
- Retail NL assertions are judged by `gpt-4.1-2025-04-14` (tau2 default), which also needs `OPENAI_API_KEY`.
- The agent's `cost` isn't filled into tau2 messages yet. Use provider dashboards, or add a
  `metrics_collected` listener.
