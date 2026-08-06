import os
import time
import asyncio
import logging
from typing import TypeVar
from pathlib import Path
from collections.abc import Callable, Coroutine

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reachy_mini import ReachyMini
from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import (
    REALTIME_MODEL,
    OPENAI_API_KEY_ENV,
    config,
    get_default_voice,
    has_openai_api_key,
    set_custom_profile,
    get_available_voices,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.prompts import get_profile_instructions
from reachy_mini_conversation_app.realtime import RealtimeConversation
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, selected_tool_names
from reachy_mini_conversation_app.personality_routes import build_personality_ops, register_personality_methods
from reachy_mini_conversation_app.profile_tool_routes import register_profile_tool_methods
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config


logger = logging.getLogger(__name__)
ConversationFactory = Callable[[str], RealtimeConversation]
ResultT = TypeVar("ResultT")
RETRY_DELAY_SECONDS = 5.0


class LocalStream:
    """Connect one Realtime conversation to Reachy Mini media and settings."""

    def __init__(
        self,
        robot: ReachyMini,
        *,
        conversation_factory: ConversationFactory,
        settings_app: FastAPI | None = None,
        instance_path: str | Path | None = None,
        startup_voice: str | None = None,
    ) -> None:
        """Initialize media, lifecycle, and settings state."""
        self._robot = robot
        self._conversation_factory = conversation_factory
        self._settings_app = settings_app
        self._instance_path = Path(instance_path) if instance_path is not None else None
        self._voice = startup_voice or get_default_voice()
        self._conversation = conversation_factory(self._voice)
        self._stop_event = asyncio.Event()
        self._restart_requested = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._rpc: JsonRpcServer | None = None
        self._settings_initialized = False
        self._mic_muted = False
        self._connection_state = "not_started"
        self._connection_error: str | None = None
        self._install_conversation(self._conversation)

    @property
    def conversation(self) -> RealtimeConversation:
        """Return the currently installed conversation."""
        return self._conversation

    def _install_conversation(self, conversation: RealtimeConversation) -> None:
        self._conversation = conversation
        conversation.set_clear_player(self._clear_player)
        conversation.set_activity_observer(self._dispatch_activity)

    def seconds_since_activity(self) -> float:
        """Return seconds since the active conversation last changed state."""
        return time.monotonic() - self._conversation.last_activity_time

    def _dispatch_activity(self, reason: str) -> None:
        if self._rpc is not None:
            self._rpc.broadcast_threadsafe("conversation.activity", {"reason": reason})

    async def _run_on_stream_loop(self, coroutine: Coroutine[object, object, ResultT]) -> ResultT:
        loop = self._asyncio_loop
        if loop is None:
            coroutine.close()
            raise RuntimeError("Conversation loop is not running")
        if loop is asyncio.get_running_loop():
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return await asyncio.wrap_future(future)

    async def request_restart(self, reason: str) -> None:
        """Request one session restart from the owning stream loop."""
        logger.info("Realtime restart requested: %s", reason)
        self._connection_state = "connecting"
        self._restart_requested.set()
        await self._conversation.shutdown()

    async def apply_personality(self, profile: str | None) -> str:
        """Validate and apply a profile through a session restart."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            get_profile_instructions()
            get_function_tools(selected_tool_names(str(self._instance_path) if self._instance_path else None))
        except Exception:
            set_custom_profile(previous_profile)
            raise
        await self.request_restart("personality_changed")
        return "Applied personality and restarting the conversation."

    async def change_voice(self, voice: str) -> str:
        """Persist a supported voice and restart because session voices are immutable."""
        if voice not in get_available_voices():
            raise ValueError(f"Unsupported OpenAI voice: {voice}")
        self._voice = voice
        settings = read_startup_settings(self._instance_path)
        write_startup_settings(self._instance_path, profile=settings.profile, voice=voice)
        await self.request_restart("voice_changed")
        return f"Voice changed to {voice}; restarting the conversation."

    async def get_available_voices(self) -> list[str]:
        """Return supported OpenAI Realtime voices."""
        return get_available_voices()

    def get_current_voice(self) -> str:
        """Return the voice selected for the active session."""
        return self._voice

    def _persist_personality(self, profile: str | None, voice: str | None) -> None:
        write_startup_settings(self._instance_path, profile=profile, voice=voice)

    def _persist_openai_key(self, api_key: str) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("OpenAI API key is required")
        os.environ[OPENAI_API_KEY_ENV] = normalized_key
        refresh_runtime_config_from_env()
        if self._instance_path is None:
            return
        env_path = self._instance_path / ".env"
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        replacement = f"{OPENAI_API_KEY_ENV}={normalized_key}"
        for index, line in enumerate(lines):
            if line.strip().startswith(f"{OPENAI_API_KEY_ENV}="):
                lines[index] = replacement
                break
        else:
            lines.append(replacement)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Persisted %s to the app instance configuration", OPENAI_API_KEY_ENV)

    def _status(self) -> dict[str, object]:
        return {
            "model": REALTIME_MODEL,
            "has_key": has_openai_api_key(),
            "connected": self._conversation.connected,
            "connection_state": "connected" if self._conversation.connected else self._connection_state,
            "connection_error": None if self._conversation.connected else self._connection_error,
            "voice": self._voice,
        }

    def init_settings_ui(self) -> None:
        """Mount the small settings and conversation JSON-RPC surface."""
        if self._settings_initialized or self._settings_app is None:
            return
        app = self._settings_app
        static_directory = Path(__file__).parent / "static"
        app.router.routes[:] = [
            route
            for route in app.router.routes
            if getattr(route, "path", None) not in {"/", "/static", "/{path:path}"}
        ]
        app.mount("/static", StaticFiles(directory=str(static_directory)), name="static")

        @app.get("/")
        def root() -> FileResponse:
            """Serve the app UI."""
            return FileResponse(str(static_directory / "index.html"))

        @app.get("/favicon.ico")
        def favicon() -> Response:
            """Avoid a noisy missing favicon request."""
            return Response(status_code=204)

        rpc = JsonRpcServer()

        def status(_params: dict[str, object]) -> dict[str, object]:
            return self._status()

        async def say(params: dict[str, object]) -> dict[str, object]:
            text = str(params.get("text", "")).strip()
            if not text:
                raise JsonRpcError("say requires text", reason="invalid_params", code=-32602)
            await self._run_on_stream_loop(self._conversation.say(text))
            return {"ok": True}

        async def interrupt(_params: dict[str, object]) -> dict[str, object]:
            await self._run_on_stream_loop(self._conversation.interrupt())
            return {"ok": True}

        def microphone(params: dict[str, object]) -> dict[str, object]:
            if "muted" in params:
                self._mic_muted = bool(params["muted"])
            return {"muted": self._mic_muted}

        async def configure_openai(params: dict[str, object]) -> dict[str, object]:
            api_key = str(params.get("api_key", ""))
            try:
                self._persist_openai_key(api_key)
            except (OSError, ValueError) as error:
                raise JsonRpcError(str(error), reason="invalid_openai_key", code=-32602) from error
            await self._run_on_stream_loop(self.request_restart("openai_key_changed"))
            return self._status()

        rpc.register("conversation.status", status)
        rpc.register("conversation.say", say)
        rpc.register("conversation.interrupt", interrupt)
        rpc.register("conversation.mic", microphone)
        rpc.register("openai.config", configure_openai)
        rpc.mount(app)
        self._rpc = rpc

        personality_ops = build_personality_ops(
            self,
            lambda: self._asyncio_loop,
            persist_personality=self._persist_personality,
            get_persisted_personality=lambda: read_startup_settings(self._instance_path).profile,
        )
        register_personality_methods(rpc, personality_ops)
        register_profile_tool_methods(
            rpc,
            lambda: self._asyncio_loop,
            self.request_restart,
            instance_path=self._instance_path,
        )
        self._settings_initialized = True

    async def _run_session_loop(self) -> None:
        conversation = self._conversation
        while not self._stop_event.is_set():
            if not has_openai_api_key():
                self._connection_state = "waiting_for_config"
                self._connection_error = f"{OPENAI_API_KEY_ENV} is not configured"
                await self._wait_for_restart(0.5)
                continue
            self._restart_requested.clear()
            if conversation.voice != self._voice:
                conversation = self._conversation_factory(self._voice)
            self._install_conversation(conversation)
            self._connection_state = "connecting"
            self._connection_error = None
            try:
                await conversation.start_up()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._connection_state = "disconnected"
                self._connection_error = f"{type(error).__name__}: {error}"
                logger.warning("Realtime session failed: %s", error, exc_info=logger.isEnabledFor(logging.DEBUG))
            if self._stop_event.is_set():
                return
            conversation = self._conversation_factory(self._voice)
            if self._restart_requested.is_set():
                continue
            await self._wait_for_restart(RETRY_DELAY_SECONDS)

    async def _wait_for_restart(self, timeout: float) -> None:
        if self._restart_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._restart_requested.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return

    def launch(self) -> None:
        """Start media and run the realtime, capture, and playback loops."""
        self._stop_event.clear()
        if self._instance_path is not None:
            env_path = self._instance_path / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                refresh_runtime_config_from_env()
        self.init_settings_ui()
        if not has_openai_api_key() and self._settings_app is None:
            logger.error("%s is not configured", OPENAI_API_KEY_ENV)
            return

        self._robot.media.start_recording()
        self._robot.media.start_playing()

        async def run_streams() -> None:
            self._asyncio_loop = asyncio.get_running_loop()
            await asyncio.gather(
                asyncio.sleep(1.0),
                asyncio.to_thread(apply_audio_startup_config, self._robot, logger=logger),
            )
            self._tasks = [
                asyncio.create_task(self._run_session_loop(), name="realtime-session"),
                asyncio.create_task(self.record_loop(), name="audio-capture"),
                asyncio.create_task(self.play_loop(), name="audio-playback"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Conversation tasks cancelled")
            finally:
                await self._conversation.shutdown()

        asyncio.run(run_streams())

    def close(self) -> None:
        """Stop media and cancel stream-loop tasks safely from any thread."""
        logger.info("Stopping local conversation stream")
        try:
            self._robot.media.stop_recording()
        except Exception as error:
            logger.debug("Failed to stop recording cleanly: %s", error)
        try:
            self._robot.media.stop_playing()
        except Exception as error:
            logger.debug("Failed to stop playback cleanly: %s", error)
        loop = self._asyncio_loop
        if loop is None or not loop.is_running():
            self._stop_event.set()
            return
        loop.call_soon_threadsafe(self._stop_event.set)
        for task in self._tasks:
            if not task.done():
                loop.call_soon_threadsafe(task.cancel)

    def _clear_player(self) -> None:
        audio = self._robot.media.audio
        if audio is None:
            logger.warning("Cannot clear robot playback: audio output is unavailable")
            return
        try:
            audio.clear_player()
        except Exception as error:
            logger.warning("Failed to clear robot playback: %s", error)

    async def record_loop(self) -> None:
        """Forward Reachy microphone frames to the active realtime session."""
        sample_rate = self._robot.media.get_input_audio_samplerate()
        while not self._stop_event.is_set():
            audio = self._robot.media.get_audio_sample()
            if audio is not None and not self._mic_muted:
                samples = np.asarray(audio, dtype=np.float32)
                await self._conversation.receive((sample_rate, samples))
            await asyncio.sleep(0)

    async def play_loop(self) -> None:
        """Play assistant audio and report playback progress after it is heard."""
        while not self._stop_event.is_set():
            conversation = self._conversation
            try:
                audio = await asyncio.wait_for(conversation.emit(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if audio.samples.size == 0:
                continue
            self._robot.media.push_audio_sample(audio.samples)
            await conversation.acknowledge_after_playback(audio)
