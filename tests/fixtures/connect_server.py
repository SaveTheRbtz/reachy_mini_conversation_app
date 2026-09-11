"""Run the packaged SPA and production backend with simulated robot and Live I/O."""

import os
import json
import socket
import asyncio
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from contextlib import asynccontextmanager
from dataclasses import field, dataclass
from unittest.mock import create_autospec
from collections.abc import AsyncIterator

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import Field, BaseModel, ConfigDict
from starlette.types import Send, Scope, ASGIApp, Message, Receive

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.api import profile_resource_name
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.config import (
    OPENAI_VOICE_ENV,
    OPENAI_API_KEY_ENV,
    config,
    set_instance_path,
    set_custom_profile,
)
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import PlaybackAudio, InputAudioFrame, LiveConversation
from reachy_mini_conversation_app.personality import list_personalities
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.profile_toolsets import read_profile_toolsets, read_profile_tool_names
from reachy_mini_conversation_app.startup_settings import StartupSettings, read_startup_settings
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_pb import Conversation


class Faults(BaseModel):
    """Replaceable failure controls used only by browser integration tests."""

    model_config = ConfigDict(strict=True, extra="forbid")
    rpc_unavailable_methods: list[str] = Field(default_factory=list)
    watch_stalled: bool = False
    connect_stalled: bool = False
    shutdown_stalled: bool = False
    send_stalled: bool = False
    send_unavailable: bool = False
    live_disconnected: bool = False
    playback_seconds: float = Field(default=0, ge=0, le=30)


@dataclass
class Harness:
    """Keep fault controls and observations outside the production application."""

    faults: Faults = field(default_factory=Faults)
    rpc_calls: dict[str, int] = field(default_factory=dict)
    active_watches: int = 0
    active_says: int = 0
    completed_says: int = 0
    session_starts: int = 0
    shutdowns: int = 0
    interrupts: int = 0
    microphone_frames: int = 0
    playback_frames: int = 0
    sent_texts: list[str] = field(default_factory=list)


class BackendState(BaseModel):
    """Expose actual runtime state and persisted files without credential values."""

    ready: bool
    worker_alive: bool
    conversation: Conversation
    startup_settings: StartupSettings
    credential_saved: bool
    profile_files: dict[str, str]
    toolsets: dict[str, list[str]]
    effective_tools: dict[str, list[str]]
    rpc_calls: dict[str, int]
    active_watches: int
    active_says: int
    completed_says: int
    session_starts: int
    shutdowns: int
    interrupts: int
    microphone_frames: int
    playback_frames: int
    sent_texts: list[str]


class SimulatedLive(LiveConversation):
    """Replace OpenAI I/O while retaining real dialogue and playback bookkeeping."""

    def __init__(self, dependencies: ToolDependencies, voice: str, harness: Harness) -> None:
        """Initialize a real conversation without opening an external connection."""
        super().__init__(dependencies, voice=voice, output_sample_rate=48_000)
        self._harness = harness
        self._ready = False

    @property
    def connected(self) -> bool:
        """Report the simulated transport's current connection state."""
        return self._ready

    async def start_up(self) -> None:
        """Let the production session owner handle failures and requested restarts."""
        self._harness.session_starts += 1
        try:
            while self._harness.faults.connect_stalled:
                await asyncio.sleep(0.02)
            if self._harness.faults.live_disconnected:
                raise ConnectionError("Simulated Live connection unavailable")
            self._ready = True
            self._notify_activity("connected")
            while not self._closed.is_set():
                if self._harness.faults.live_disconnected:
                    raise ConnectionError("Simulated Live connection lost")
                await asyncio.sleep(0.02)
        finally:
            self._ready = False
            self.clear_playback()
            self._notify_activity("disconnected")

    async def shutdown(self) -> None:
        """Make graceful finalization independently controllable from local API calls."""
        self._harness.shutdowns += 1
        while self._harness.faults.shutdown_stalled:
            await asyncio.sleep(0.02)
        self._closed.set()

    async def receive(self, frame: InputAudioFrame) -> None:
        """Pace simulated microphone capture through the real audio loop."""
        self._harness.microphone_frames += 1
        await asyncio.sleep(0.02)

    async def say(self, text: str) -> None:
        """Observe accepted text and expose cancellation of a stalled remote write."""
        self._harness.active_says += 1
        try:
            if not self.connected or self._harness.faults.send_unavailable:
                raise ConnectionError("Simulated Live write unavailable")
            while self._harness.faults.send_stalled:
                await asyncio.sleep(0.02)
            self._remember("user", text, new_message=True)
            self._harness.sent_texts.append(text)
            self._harness.completed_says += 1
            seconds = self._harness.faults.playback_seconds
            if seconds:
                self._playback_interrupted.clear()
                self.output_queue.put_nowait(PlaybackAudio(np.zeros(int(48_000 * seconds), dtype=np.float32)))
        finally:
            self._harness.active_says -= 1

    async def interrupt(self) -> None:
        """Run the real playback flush and speaking-state release."""
        self._harness.interrupts += 1
        await super().interrupt()


class ObservedApplication:
    """Count actual RPCs and inject transport failures around the production ASGI app."""

    def __init__(self, application: ASGIApp, harness: Harness) -> None:
        """Wrap the application without replacing any resource handler."""
        self._application = application
        self._harness = harness

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Observe production RPC lifetimes and controllable transport failures."""
        if scope["type"] != "http" or not scope["path"].startswith("/rpc/"):
            await self._application(scope, receive, send)
            return
        method = scope["path"].rsplit("/", 1)[-1]
        harness = self._harness
        harness.rpc_calls[method] = harness.rpc_calls.get(method, 0) + 1
        if method in harness.faults.rpc_unavailable_methods:
            await send(
                {"type": "http.response.start", "status": 503, "headers": [(b"content-type", b"application/json")]}
            )
            await send(
                {"type": "http.response.body", "body": b'{"code":"unavailable","message":"Simulated RPC failure"}'}
            )
            return
        watching = method == "WatchConversation"
        if watching:
            harness.active_watches += 1

        async def observed_send(message: Message) -> None:
            if watching and message["type"] == "http.response.body" and message.get("more_body"):
                while harness.faults.watch_stalled:
                    await asyncio.sleep(0.02)
            await send(message)

        try:
            await self._application(scope, receive, observed_send)
        finally:
            if watching:
                harness.active_watches -= 1


async def main() -> None:
    """Print the bound port and serve until the browser test stops this process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-key", action="store_true")
    options = parser.parse_args()
    instance_directory = TemporaryDirectory(prefix="reachy-browser-test-")
    with instance_directory, socket.socket() as listener:
        instance_path = Path(instance_directory.name)
        set_instance_path(instance_path)
        set_custom_profile(None)
        os.environ.pop(OPENAI_API_KEY_ENV, None)
        os.environ.pop(OPENAI_VOICE_ENV, None)
        config.OPENAI_API_KEY = None
        if not options.no_key:
            config.OPENAI_API_KEY = "sk-synthetic-fixture-key"
            os.environ[OPENAI_API_KEY_ENV] = config.OPENAI_API_KEY
        harness = Harness()
        robot = create_autospec(ReachyMini, instance=True)
        robot.media.get_input_audio_samplerate.return_value = 16_000
        robot.media.get_audio_sample.return_value = np.zeros(320, dtype=np.float32)

        def playback(samples: object) -> None:
            harness.playback_frames += 1

        robot.media.push_audio_sample.side_effect = playback
        dependencies = ToolDependencies(
            reachy_mini=robot,
            movement_manager=create_autospec(MovementManager, instance=True),
            memory=MemorySnapshot(memories=[]),
            instance_path=instance_path,
            camera_enabled=False,
        )

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[None]:
            worker.start()
            try:
                yield
            finally:
                harness.faults = Faults()
                stream.close()
                await asyncio.to_thread(worker.join, 5)
                if worker.is_alive():
                    raise RuntimeError("The simulated robot did not shut down")
                # Uvicorn re-raises termination signals after the application lifespan ends.
                instance_directory.cleanup()

        application = FastAPI(lifespan=lifespan)
        stream = LocalStream(
            robot,
            conversation_factory=lambda voice: SimulatedLive(dependencies, voice, harness),
            settings_app=application,
            instance_path=instance_path,
        )
        worker = Thread(target=stream.launch, name="simulated-robot")

        @application.put("/__test/faults")
        async def replace_faults(faults: Faults) -> Faults:
            harness.faults = faults
            return faults

        @application.get("/__test/state")
        def state() -> BackendState:
            conversation = stream.snapshot()
            credential_path = instance_path / ".env"
            return BackendState(
                ready=worker.is_alive()
                and conversation.connection_state
                in {Conversation.ConnectionState.CONNECTED, Conversation.ConnectionState.WAITING_FOR_CONFIG},
                worker_alive=worker.is_alive(),
                conversation=conversation,
                startup_settings=read_startup_settings(instance_path),
                credential_saved=credential_path.is_file() and OPENAI_API_KEY_ENV in credential_path.read_text(),
                profile_files={
                    path.relative_to(instance_path).as_posix(): path.read_text(encoding="utf-8")
                    for path in config.user_personalities_root().rglob("*.md")
                },
                toolsets=read_profile_toolsets(instance_path).profiles,
                effective_tools={
                    profile_resource_name(selector): read_profile_tool_names(selector, instance_path)
                    for selector in list_personalities()
                },
                rpc_calls=dict(harness.rpc_calls),
                active_watches=harness.active_watches,
                active_says=harness.active_says,
                completed_says=harness.completed_says,
                session_starts=harness.session_starts,
                shutdowns=harness.shutdowns,
                interrupts=harness.interrupts,
                microphone_frames=harness.microphone_frames,
                playback_frames=harness.playback_frames,
                sent_texts=list(harness.sent_texts),
            )

        stream.init_settings_ui()
        listener.bind(("127.0.0.1", options.port))
        print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
        await uvicorn.Server(
            uvicorn.Config(ObservedApplication(application, harness), log_level="warning", timeout_graceful_shutdown=2)
        ).serve(sockets=[listener])


if __name__ == "__main__":
    asyncio.run(main())
