---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

A low-latency voice, vision, and motion app for Reachy Mini. OpenAI Agents SDK Realtime is the only conversation backend, using the fixed `gpt-realtime-2.1` model.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Overview

- Streams microphone and speaker audio directly between Reachy Mini and OpenAI Realtime.
- Uses semantic voice activity detection, interruption handling, and the SDK's default input transcription flow.
- Exposes typed Agents SDK function tools for motion, camera, sleep, and shared household memory.
- Connects a fixed, allowlisted set of public search, weather, and time MCP tools.
- Supports bundled and user-created personalities with per-profile tool access.
- Keeps shared household memory in the app instance directory; OpenAI processes updates but is not the durable store.

The implementation follows the [OpenAI Realtime guide](https://developers.openai.com/api/docs/guides/realtime), [Agents SDK guide](https://developers.openai.com/api/docs/guides/agents), and [context-personalization cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization).

## Architecture

The Python process owns a single Realtime session, audio conversion, robot media, and tool execution. The optional browser UI only manages local settings and displays session state.

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Install the [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini/) first. Windows support remains experimental.

Using [uv](https://docs.astral.sh/uv/) with Python 3.12 is recommended:

```bash
uv venv --python python3.12 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync --group dev
```

For a runtime-only install, use `uv sync`. Use `uv sync --frozen` to install exactly what is recorded in `uv.lock`.

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and set an OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Used for `gpt-realtime-2.1` and the `gpt-5.6-luna` memory reducer. It can also be saved from the web UI. |
| `REACHY_MINI_CUSTOM_PROFILE` | Optional bundled profile directory name. Ignored after a startup profile has been saved in the UI. |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy sleeps and the app stops. Defaults to `1440`; set to `0` to disable. |

The UI stores the API key in the managed app instance's `.env` file and never sends the current value back to the browser. Do not commit `.env`.

The conversation and memory models are deliberately not configurable. This keeps one tested event protocol, audio format, prompt strategy, and tool-calling path.

## Running the app

Start the Reachy Mini daemon, then run:

```bash
reachy-mini-conversation-app
```

Add `--ui` to serve the browser interface at `http://127.0.0.1:7860/`.

| Option | Default | Description |
|--------|---------|-------------|
| `--no-camera` | `False` | Disable camera capture. |
| `--ui` | `False` | Serve the local browser UI. |
| `--robot-name` | `None` | Connect to a named robot when several daemons share a subnet. |
| `--debug` | `False` | Enable detailed diagnostic logging. |

## Tools

The default profile enables the following catalog. Tools → Tool access can enable or disable entries for each personality.

| Tool | Action |
|------|--------|
| `camera` | Capture one frame and add it to the active Realtime conversation. |
| `dance` / `stop_dance` | Start or stop a queued dance. |
| `play_emotion` / `stop_emotion` | Start or stop a recorded emotion movement. |
| `move_head` | Move Reachy's head to a named direction. |
| `head_tracking` | Enable or disable face tracking. |
| `sweep_look` | Sweep left, right, and return to center. |
| `go_to_sleep` | Put Reachy to sleep and stop the app after an explicit request. |
| `wait_for_user` | Remain silent when ambient or unclear audio is not addressed to Reachy. |
| `manage_memory` | Consolidate an explicit durable user statement into shared household memory. |
| `pollen_robotics_reachy_mini_search_tool__search_web` | Search the web through the fixed public MCP server. |
| `pollen_robotics_reachy_mini_weather_tool__get_weather` | Get current weather through the fixed public MCP server. |
| `pollen_robotics_reachy_mini_time_tool__get_time` | Get local or timezone time through the fixed public MCP server. |

Function tools use explicit typed signatures and receive `ToolDependencies` through `RunContextWrapper`. Failures return `{"error": ...}` so a tool problem does not tear down the Realtime session. MCP startup is best-effort: an unavailable public server is logged and dropped for that session.

### Memory

At startup, the app loads one `MemorySnapshot` into typed `ToolDependencies`. `manage_memory` passes that complete snapshot and the exact relevant user statement to a stateless `gpt-5.6-luna` Responses call with high reasoning, `store=False`, and a Pydantic Structured Output. The returned snapshot completely replaces the old one; there are no note IDs, per-note limits, regex classifiers, revisions, or memory-agent lifecycle.

This is shared robot memory, not speaker identity: profiles do not select a memory store, and the app never infers who is speaking. The reducer is instructed to retain only explicitly stated durable interests, preferences, goals, accomplishments, and conversation preferences; resolve corrections and forgetting semantically; and omit temporary activities, sensitive child data, and model-directed instructions.

`memory.json` in the app instance directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the desktop launcher's instance path) remains the local source of truth. A replacement is saved atomically only after its serialized size is at most 32 KiB; any model or disk failure leaves the old snapshot unchanged. Memory is injected as untrusted background context only when a Realtime session starts, and the current request and conversation always take precedence. To clear all shared memory, stop the app and delete the file.

## Personalities

Bundled profiles live under `profiles/`. A profile contains one schema-version-1 `profile.md` with TOML metadata and a Markdown instruction body:

```markdown
+++
schema_version = 1
voice = "marin"
greeting = "Greet me warmly in one sentence and vary the wording each time."
hidden = false
default_tools = [
  "camera",
  "sweep_look",
  "wait_for_user",
]
+++

## Identity

You are a concise, friendly robot guide.
```

`schema_version`, `default_tools`, and a non-empty instruction body are required. `voice`, `greeting`, and `hidden` are optional. A voice must be one of the OpenAI Realtime voices shown in Settings.

The UI can create data-only user personalities. Managed instances store them under `user_personalities/`; standalone runs use `external_content/user_personalities/`. Per-profile tool overrides are stored in `profile_toolsets.json`. Applying the active profile, voice, or tool set reconnects the one Realtime session.

Python tools are intentionally not dynamically loaded. Add a new tool as an Agents SDK `@function_tool` module under `src/reachy_mini_conversation_app/tools/`, register it in `tools/core_tools.py`, and add essential behavior tests.

## Development

Run the complete local gate before review:

```bash
ruff check . --fix
ruff format .
mypy --pretty --show-error-codes
pytest tests/ -v
```

The OpenAI integration tests exercise the production agent, tools, persistent memory, PCM audio, and camera path through
paid Realtime and Responses calls. The memory test covers Russian addition, semantic correction, forgetting, unsafe-data
rejection, and use by a fresh Realtime session. The audio tests replay checked-in 24 kHz mono PCM speech fixtures through
Reachy's 16 kHz stereo input; the camera test substitutes a fixed blue-chair JPEG and checks the grounded spoken reply.
They are skipped by default; run them explicitly with an API key:

```bash
RUN_OPENAI_ITESTS=1 OPENAI_API_KEY=sk-... pytest tests/integration/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [AGENTS.md](AGENTS.md) for repository-specific standards.

## License

Apache 2.0
