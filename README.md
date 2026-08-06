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
- Exposes typed Agents SDK function tools for motion, camera, sleep, and local memory.
- Connects a fixed, allowlisted set of public search, weather, and time MCP tools.
- Supports bundled and user-created personalities with per-profile tool access.
- Keeps persistent user memories in the app instance directory; OpenAI session state is intentionally not treated as durable storage.

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
| `OPENAI_API_KEY` | Required. Used for the `gpt-realtime-2.1` session. It can also be saved from the web UI. |
| `REACHY_MINI_CUSTOM_PROFILE` | Optional bundled profile directory name. Ignored after a startup profile has been saved in the UI. |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy sleeps and the app stops. Defaults to `1440`; set to `0` to disable. |

The UI stores the API key in the managed app instance's `.env` file and never sends the current value back to the browser. Do not commit `.env`.

The model is deliberately not configurable. This keeps one tested event protocol, audio format, prompt strategy, and tool-calling path.

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
| `remember` / `forget` | Save or replace an explicit durable fact, or remove one by its exact memory ID. |
| `pollen_robotics_reachy_mini_search_tool__search_web` | Search the web through the fixed public MCP server. |
| `pollen_robotics_reachy_mini_weather_tool__get_weather` | Get current weather through the fixed public MCP server. |
| `pollen_robotics_reachy_mini_time_tool__get_time` | Get local or timezone time through the fixed public MCP server. |

Function tools use explicit typed signatures and receive `ToolDependencies` through `RunContextWrapper`. Failures return `{"error": ...}` so a tool problem does not tear down the Realtime session. MCP startup is best-effort: an unavailable public server is logged and dropped for that session.

### Memory

At startup, the app loads a bounded `MemoryState` into typed `ToolDependencies`. The Agents SDK shares that state with dynamic instructions and function tools through `RunContextWrapper`. The instructions render escaped notes inside `<user_memories>` at session start; after a memory tool changes the state's revision, the session refreshes them after the current response for subsequent turns. A failed refresh closes the stale session so the supervisor can reconnect with current state.

Saved notes are untrusted context, not policy. The current request takes precedence over the active conversation, which takes precedence over durable memory. Only explicit `remember` and `forget` calls mutate the state. Notes are deduplicated and limited to 20 entries of 280 characters. Prompt policy excludes secrets, identifiers, health data, and instructions; validation rejects common sensitive or secret patterns and instruction-shaped content before persistence.

`memory.v1.json` in the app instance directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the desktop launcher's instance path) is updated atomically and remains the cross-session source of truth. The connected Realtime conversation supplies short-term session context; there is no remote memory store, vector index, or second-model consolidation pass. To clear all memories, stop the app and delete the file.

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

The OpenAI integration tests make paid calls to `gpt-realtime-2.1` and are skipped by default. Run them explicitly with an API key:

```bash
RUN_OPENAI_ITESTS=1 OPENAI_API_KEY=sk-... pytest tests/integration/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [AGENTS.md](AGENTS.md) for repository-specific standards.

## License

Apache 2.0
