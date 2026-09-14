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

Talk with Reachy Mini. Ask questions, show it an object, or ask it to dance.
Choose a personality and teach it preferences you want it to remember.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Installation

You need a working [Reachy Mini and daemon](https://github.com/pollen-robotics/reachy_mini/),
an internet connection, and an OpenAI API key.

To install from source, install [uv](https://docs.astral.sh/uv/),
[Git LFS](https://git-lfs.com/), and Node.js 22.12 or newer, then run:

```bash
git clone https://github.com/SaveTheRbtz/reachy_mini_conversation_app.git
cd reachy_mini_conversation_app
git lfs pull
uv sync --frozen --python 3.12
```

Copy [.env.example](.env.example) to `.env` and fill in `OPENAI_API_KEY`.
Keep this file private. Windows support is experimental.

## Running the app

Start the Reachy Mini daemon, then run:

```bash
uv run --frozen reachy-mini-conversation-app --ui
```

Open [localhost:7860](http://127.0.0.1:7860/) and start talking.

- **Conversation:** type a message, mute the microphone, or stop Reachy speaking.
- **Personalities:** choose or create a personality and set its tool access.
- **Settings:** choose a voice, set the startup personality, or enter an API key.

Ask naturally for actions such as looking right, showing happiness, or following your face.
These requests use the active personality's enabled tools. Reachy reports when a capability
is unavailable or an action fails.

Changing the active personality or voice starts a fresh conversation. Reachy sleeps
and the app stops after 15 minutes without conversation; set `REACHY_MINI_APP_TIMEOUT_MINUTES=0`
in `.env` to disable this.

Omit `--ui` to run without the browser. Add `--no-camera` to disable camera requests,
`--robot-name NAME` to select a robot, or `--debug` for detailed logs.
See [.env.example](.env.example) for other startup settings.

For command-line installs, save your key in `.env`: a key entered in the web UI lasts
only until the app stops. The Reachy Mini app launcher saves UI keys for later runs.

## Memory and privacy

Conversation audio, text, and requested camera images go to OpenAI. Saved memories
stay on your machine and are sent to OpenAI as conversation context. Memory is
shared across personalities.

Ask Reachy to remember, correct, or forget a preference. To erase all saved memory,
stop the app and delete `memory.json` from its instance directory. For standalone
runs, the default is `~/.local/share/reachy_mini_conversation_app/` (`XDG_DATA_HOME`
overrides the data directory).

## Troubleshooting

If Reachy cannot connect, check that its daemon is running and that the robot is
reachable on your network. If it connects but cannot talk, check your internet
connection and API key. Run with `--debug` to see the cause.

For an app running on the robot, read daemon logs with
`journalctl -u reachy-mini-daemon -b --no-pager`.

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [testing guide](tests/README.md).

The backend is Python; the browser UI is Vue and TypeScript. The
[protobuf contract](proto/reachy/conversation/v1/api.proto) defines their Connect API.
`uv sync` generates the API code and builds the UI. Commit source and lockfiles;
generated code and browser assets are build outputs.

Run the app with `--ui`, then use `npm run dev` for frontend hot reload.
After changing a schema, run `npm run generate` and restart the Python app.
Use `uv build` to create installable packages; those packages need no Node.js to install.

### Architecture

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

[Apache 2.0 license](LICENSE)
