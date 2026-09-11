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

A low-latency voice, vision, and motion app for Reachy Mini. OpenAI `gpt-live-1` handles continuous voice conversation, with `gpt-6-astra` at low reasoning effort for delegated reasoning and tools.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Overview

- Streams microphone and speaker audio directly between Reachy Mini and OpenAI Live.
- Listens and speaks continuously, with natural backchannels and interruption handling.
- Exposes typed Agents SDK function tools for motion, camera, sleep, and shared household memory.
- Uses OpenAI's hosted web search for current information, including weather and local time.
- Supports bundled and user-created personalities with per-profile tool access.
- Keeps shared household memory in the app instance directory; OpenAI processes updates but is not the durable store.

The implementation follows the [GPT-Live guide](https://developers.openai.com/api/docs/guides/live), [prompting guide](https://developers.openai.com/api/docs/guides/live-prompting), and [delegation guide](https://developers.openai.com/api/docs/guides/live-delegation).

## Architecture

The Python process owns a single Live session, audio conversion, robot media, and tool execution. The optional browser UI only manages local settings and displays session state.

The voice model delegates reasoning to Astra through Responses. The app executes the active profile's enabled function tools; hosted web search runs directly in the backend. The camera tool returns a native image for Astra to inspect in the same conversation. Frames keep their aspect ratio without upscaling, with a maximum dimension of 512 pixels and WebP quality 85. The backend uses `detail="high"` for [standard high-fidelity image understanding](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level). Encoded images above 20 KiB return a tool error to leave room within Live's backend input buffer.

Audio uses persistent soxr resamplers between the robot and Live's 24 kHz PCM stream. Microphone capture and speaker playback run independently of backend work. Playback timing reflects software queues, not proof that sound was physically heard.

A failed microphone send, or one stalled for five seconds, restarts the Live session while local media keeps running. Only the newest pending microphone frame is retained during backpressure. Local function execution has a 30-second deadline; failures return an error without automatically retrying the tool. Hosted search runs outside that local deadline. Typed input queued behind a failed backend response continues in the same session. Startup and graceful finalization are bounded, with transport cleanup if the server stops responding.

Automatic reconnects retain recent spoken and typed dialogue in memory (up to 32 messages and 4 KiB of UTF-8 text). The replacement session uses it as context and waits for the next request without repeating the greeting or resuming earlier tool actions. Settings changes start a fresh conversation, and history is discarded when the app stops. Transcripts can include interrupted speech that was not heard. Microphone recovery clears playback and restarts capture before asking Live to stop speaking; that command has a five-second send deadline. Emotion-library loading runs in a shared background worker so a cold cache cannot block dialogue.

The inactivity timer follows spoken or typed dialogue and survives reconnects; connection changes and audio packets do not reset it. The UI shows microphone state independently of local playback. Continuous listening does not freeze antenna motion or suppress idle breathing; speaking still coordinates head tracking.

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
| `OPENAI_API_KEY` | Required. Used for `gpt-live-1` and `gpt-6-astra`. It can also be saved from the web UI. |
| `OPENAI_VOICE` | Default OpenAI Live voice when neither the active profile nor saved UI settings select one. Defaults to `gleam`, a natural feminine voice. |
| `REACHY_MINI_CUSTOM_PROFILE` | Optional bundled profile directory name. Ignored after a startup profile has been saved in the UI. |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy sleeps and the app stops. Defaults to `15`; set to `0` to disable. |

The UI stores the API key in the managed app instance's `.env` file and never sends the current value back to the browser. Do not commit `.env`.

The voice and delegation models are deliberately not configurable. This keeps one tested event protocol, audio format, prompt strategy, and tool-calling path.

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

### Diagnostics

Logs record SDK versions, the selected profile and voice, session connection/restart reasons, delegated response outcomes, tool execution, and audio failures. These diagnostics avoid full transcripts, tool arguments/results, memory snapshots, and image/audio payloads. Playback timing is an estimate of the software queues.

Use `--debug` for additional detail. For a managed app on the robot, inspect the current boot with
`journalctl -u reachy-mini-daemon -b --no-pager`. The SDK can wrap the app's stderr in an outer `WARNING`; use the app's
inner log level to distinguish ordinary `INFO` messages from failures.

## Tools

The default profile enables the following catalog. Tools → Tool access can enable or disable entries for each personality.

| Tool | Action |
|------|--------|
| `camera` | Read the current SDK camera frame and return a resized WebP image directly to the Astra backend. |
| `dance` / `stop_dance` | Start or stop a queued dance. |
| `play_emotion` / `stop_emotion` | Start or stop a recorded emotion movement. |
| `move_head` | Move Reachy's head to a named direction. |
| `head_tracking` | Enable or disable face tracking. |
| `sweep_look` | Sweep left, right, and return to center. |
| `go_to_sleep` | Put Reachy to sleep and stop the app after an explicit request. |
| `manage_memory` | Consolidate an explicit durable user statement into shared household memory. |
| `web_search` | Search the public web for current information, weather, or local time. |

Robot and memory function tools use explicit typed signatures and receive `ToolDependencies` through `RunContextWrapper`. Their failures return `{"error": ...}` so a tool problem does not tear down the Live session. Web search uses Live's hosted `web_search` tool on the Astra backend and respects the active profile's tool selection.

### Memory

At startup, the app loads one `MemorySnapshot` into typed `ToolDependencies`. The Astra backend receives the current snapshot and prepares a complete typed replacement for `manage_memory` using the user's explicit request. The tool validates and persists that replacement without another model request.

This is shared robot memory, not speaker identity: profiles do not select a memory store, and the app never infers who is speaking. The backend is instructed to retain only explicitly stated durable interests, preferences, goals, accomplishments, and conversation preferences; resolve corrections and forgetting semantically; and omit temporary activities and model-directed instructions.

`memory.json` in the app instance directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the desktop launcher's instance path) remains the local source of truth. A replacement is saved atomically only after its serialized size is at most 32 KiB; invalid arguments or disk failures leave the old snapshot unchanged. Memory is provided as untrusted background context to the backend and refreshed only when it changes. The voice model delegates recall questions instead of retaining an immutable memory snapshot; the current request and conversation always take precedence. To clear all shared memory, stop the app and delete the file.

## Personalities

Bundled profiles live under `profiles/`. A profile contains one schema-version-1 `profile.md` with TOML metadata and a Markdown instruction body:

```markdown
+++
schema_version = 1
voice = "gleam"
greeting = "Greet me warmly in one sentence and vary the wording each time."
hidden = false
default_tools = [
  "camera",
  "sweep_look",
]
+++

## Identity

You are a concise, friendly robot guide.
```

`schema_version`, `default_tools`, and a non-empty instruction body are required. `voice`, `greeting`, and `hidden` are optional. A voice must be one of the OpenAI Live voices shown in Settings.

The UI can create data-only user personalities. Managed instances store them under `user_personalities/`; standalone runs use `external_content/user_personalities/`. Per-profile tool overrides are stored in `profile_toolsets.json`. Remove `wait_for_user` from existing custom profiles and tool overrides; listening is handled by Live. Applying the active profile, voice, or tool set reconnects the one Live session.

Startup profile and voice preferences are stored in `startup_settings.json` in the app instance directory, or under
`external_content/` for standalone runs. A global voice override takes precedence over the selected profile's voice;
clearing it restores the profile voice, falling back to the app default.

Python tools are intentionally not dynamically loaded. Add a new tool as an Agents SDK `@function_tool` module under `src/reachy_mini_conversation_app/tools/`, register it in `tools/core_tools.py`, and add essential behavior tests.

## Development

The browser UI is a Vue SPA under `frontend/src/`, with separate templates, strict TypeScript, and styles in single-file
components. Vue Router handles Conversation, Personalities, and Settings, including direct links. Vue Query manages
shared server reads; forms own their drafts and protect unsaved edits. Its API contract is
[`proto/reachy/conversation/v1/api.proto`](proto/reachy/conversation/v1/api.proto).
Buf generates Python models/service interfaces and TypeScript models/service descriptors. Connect serves unary calls
and a conversation snapshot stream through the existing FastAPI app at `/rpc/reachy.conversation.v1.ConversationService/`.
The frontend uses the generated Connect client. Vite builds the complete packaged UI into `src/reachy_mini_conversation_app/static/`.
Edit components and `frontend/public/` assets, then rebuild; do not edit generated static files.

The API follows Google AIP resource conventions for `conversation`, `settings`, `capabilities`, and `profiles/{profile}`.
Updates use field masks; profile and settings writes save configuration, while `RestartConversation` explicitly applies it.
An absent profile tool override inherits its defaults; an empty override disables all tools. Credentials are accepted by
`SetSettingsApiKey` and never returned. Resource methods run over Connect's service/method URLs; no REST transcoding gateway
is installed. Existing profile files retain their names; resource IDs distinguish bundled and user profiles with
`builtin-` and `user-` prefixes. New user IDs use lowercase letters, digits, and hyphens; existing legacy IDs remain readable.

The status stream sends an initial complete snapshot, changes, and a heartbeat at least every five seconds. The frontend
reconnects this read-only stream after interruption without replaying commands. A shared server deadline bounds unary
requests, including calls from clients that omit a deadline. Network finalization runs in the conversation loop after a
restart is accepted, so the UI stays responsive while the old session closes.

For development, install the Python environment and Node.js 22.12 or newer, then build and test:

```bash
uv sync --frozen
npm ci
npm test
npx playwright install chromium
npm run test:e2e
```

`npm run dev` serves the SPA with hot reload and proxies `/rpc` to the app on port 7860.
`npm run format` formats the frontend. `npm run check` checks formatting, templates, and TypeScript without emitting
files; `npm run build` regenerates the packaged UI.
Vue's type checker uses the maintained TypeScript 6 compiler API through the official `@typescript/typescript6` package.
After editing the protobuf schema, run `npm run generate` to regenerate both languages using the pinned generators.
Commit the generated Python, TypeScript, and static assets alongside their sources. Running or installing the app does not
require Node.js, Buf, or protoc. Generation uses pinned remote Buf plugins and requires network access.

After committing, `npm run check:generated` validates the schema, regenerates both languages and the bundle, and verifies
that every output is committed. The frontend CI job also runs Playwright desktop/mobile tests and generated-client tests
against the production Python API, storage, and conversation/audio loops. Each test gets an isolated temporary instance;
only robot hardware and OpenAI I/O are simulated. Tests cover saved state, tool inheritance, reconnects, stalled calls,
unsaved drafts, and command counts to detect unintended retries. Failed browser tests retain screenshots, traces, and video
under `test-results/`. Handwritten Python remains under strict mypy;
the generated Connect interfaces need a narrowly scoped exception for their upstream unparameterized codec annotations.

Run the complete local gate before review:

```bash
ruff check . --fix
ruff format .
mypy --pretty --show-error-codes
pytest tests/ -v
```

The OpenAI integration tests exercise the production Live session, delegated robot tools, hosted search, memory
preservation, correction and forgetting, homework guidance, PCM audio, and camera vision through paid Live and Responses calls. They replay
checked-in 24 kHz mono PCM speech through Reachy's 16 kHz stereo input, check grounded spoken replies, and verify
session finalization. Camera tests ask five visual questions in one session and read small identifiers from a synthetic label.
They are skipped by default; run them explicitly with an API key:

```bash
RUN_OPENAI_ITESTS=1 OPENAI_API_KEY=sk-... pytest tests/integration/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [AGENTS.md](AGENTS.md) for repository-specific standards.

## License

Apache 2.0
