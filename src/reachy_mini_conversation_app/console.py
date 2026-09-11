import os
import time
import asyncio
import logging
from typing import TypeVar, TypeAlias, cast
from pathlib import Path
from collections.abc import Callable, Coroutine

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from starlette.types import ASGIApp
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.api import RequestErrors, RequestLimits, ConversationService, profile_resource_name
from reachy_mini_conversation_app.config import (
    LIVE_MODEL,
    OPENAI_API_KEY_ENV,
    config,
    get_default_voice,
    has_openai_api_key,
    set_custom_profile,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_profile_instructions
from reachy_mini_conversation_app.realtime import PlaybackAudio, LiveConversation
from reachy_mini_conversation_app.startup_settings import read_startup_settings
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, selected_tool_names
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_pb import Conversation
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_connect import (
    ConversationServiceASGIApplication,
)


logger = logging.getLogger(__name__)
ConversationFactory = Callable[[str], LiveConversation]
PlaybackAcknowledgement: TypeAlias = tuple[LiveConversation, PlaybackAudio]
ResultT = TypeVar("ResultT")
RETRY_DELAY_SECONDS = 5.0
MICROPHONE_FRAME_TIMEOUT_SECONDS = 5.0
AUDIO_WARNING_INTERVAL_SECONDS = 60.0
MICROPHONE_RETRY_DELAY_SECONDS = 0.01


class LocalStream:
    """Connect one Live conversation to Reachy Mini media and settings."""

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
        self._playback_acknowledgements: asyncio.Queue[PlaybackAcknowledgement] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._settings_initialized = False
        self._mic_muted = False
        self._playing = False
        self._last_interaction_at = time.monotonic()
        self._connection_state = Conversation.ConnectionState.NOT_STARTED
        self._connection_error: str | None = None
        self._install_conversation(self._conversation)

    @property
    def conversation(self) -> LiveConversation:
        """Return the currently installed conversation."""
        return self._conversation

    def _install_conversation(self, conversation: LiveConversation) -> None:
        self._conversation = conversation
        self._playing = False
        conversation.set_clear_player(self._clear_player)
        conversation.set_activity_observer(self._dispatch_activity)

    def seconds_since_activity(self) -> float:
        """Return seconds since spoken or typed dialogue, across reconnects."""
        return time.monotonic() - self._last_interaction_at

    def _dispatch_activity(self, reason: str) -> None:
        if reason == "interaction":
            self._last_interaction_at = time.monotonic()
            return
        if reason == "playback_started":
            self._playing = True
        elif reason in {"playback_stopped", "disconnected"}:
            self._playing = False
        if reason == "disconnected" and not self._restart_requested.is_set():
            self._connection_state = Conversation.ConnectionState.DISCONNECTED

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
        logger.info("Live restart requested: %s", reason)
        self._connection_state = Conversation.ConnectionState.CONNECTING
        self._restart_requested.set()

    async def restart(self, profile: str) -> None:
        """Validate configuration and accept a restart on the conversation loop."""
        await self._run_on_stream_loop(self._apply_profile(profile))

    async def _apply_profile(self, profile: str) -> None:
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile or previous_profile)
        try:
            get_profile_instructions()
            get_function_tools(selected_tool_names(str(self._instance_path) if self._instance_path else None))
        except Exception:
            set_custom_profile(previous_profile)
            raise
        settings = read_startup_settings(self._instance_path)
        self._voice = settings.voice or get_session_voice(default=get_default_voice())
        await self.request_restart("configuration_changed")

    async def say(self, text: str) -> None:
        """Submit typed input on the conversation loop."""
        await self._run_on_stream_loop(self._conversation.say(text))

    async def interrupt(self) -> None:
        """Clear playback and interrupt speech on the conversation loop."""
        await self._run_on_stream_loop(self._conversation.interrupt())

    async def set_muted(self, muted: bool) -> None:
        """Set local microphone silence independently of network state."""
        if muted != self._mic_muted:
            self._mic_muted = muted
            logger.info("Microphone mute changed: muted=%s", muted)

    async def set_api_key(self, api_key: str) -> None:
        """Persist the credential without exposing it in status responses."""
        await asyncio.to_thread(self._persist_openai_key, api_key)

    def _persist_openai_key(self, api_key: str) -> None:
        normalized_key = api_key.strip()
        if not normalized_key or "\n" in normalized_key or "\r" in normalized_key:
            raise ValueError("OpenAI API key must be a non-empty single line")
        if self._instance_path is None:
            os.environ[OPENAI_API_KEY_ENV] = normalized_key
            refresh_runtime_config_from_env()
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
        os.environ[OPENAI_API_KEY_ENV] = normalized_key
        refresh_runtime_config_from_env()
        logger.info("Persisted %s to the app instance configuration", OPENAI_API_KEY_ENV)

    def snapshot(self) -> Conversation:
        """Return the current local conversation resource."""
        connected = self._conversation.connected and not self._restart_requested.is_set()
        return Conversation(
            name="conversation",
            profile=profile_resource_name(config.REACHY_MINI_CUSTOM_PROFILE),
            model=LIVE_MODEL,
            connection_state=(Conversation.ConnectionState.CONNECTED if connected else self._connection_state),
            connection_error="" if connected else self._connection_error or "",
            voice=self._voice,
            muted=self._mic_muted,
            playing=self._playing,
        )

    def init_settings_ui(self) -> None:
        """Mount the generated Connect service and packaged browser UI."""
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

        @app.get("/favicon.ico")
        def favicon() -> Response:
            """Avoid a noisy missing favicon request."""
            return Response(status_code=204)

        service = ConversationServiceASGIApplication(
            ConversationService(self, self._instance_path), interceptors=[RequestErrors(), RequestLimits()]
        )
        # Starlette and Connect annotate the same ASGI protocol with incompatible scope types.
        app.mount("/rpc", cast(ASGIApp, service))

        @app.get("/{path:path}")
        def frontend() -> FileResponse:
            """Serve the SPA for browser routes, including direct navigation."""
            return FileResponse(str(static_directory / "index.html"), headers={"Cache-Control": "no-cache"})

        self._settings_initialized = True

    async def _run_session_loop(self) -> None:
        conversation = self._conversation
        attempt = 0
        while not self._stop_event.is_set():
            if not has_openai_api_key():
                self._connection_state = Conversation.ConnectionState.WAITING_FOR_CONFIG
                self._connection_error = f"{OPENAI_API_KEY_ENV} is not configured"
                self._restart_requested.clear()
                await self._wait_for_restart(0.5)
                continue
            if self._restart_requested.is_set():
                conversation.history.clear()
            self._restart_requested.clear()
            if conversation.voice != self._voice:
                conversation = self._conversation_factory(self._voice)
            self._install_conversation(conversation)
            self._connection_state = Conversation.ConnectionState.CONNECTING
            self._connection_error = None
            attempt += 1
            started_at = time.monotonic()
            reason = "remote_close"
            logger.info("Live connection attempt: attempt=%d model=%s voice=%s", attempt, LIVE_MODEL, self._voice)
            session_task = asyncio.create_task(conversation.start_up())
            restart_task = asyncio.create_task(self._restart_requested.wait())
            try:
                finished, _ = await asyncio.wait({session_task, restart_task}, return_when=asyncio.FIRST_COMPLETED)
                if session_task in finished:
                    await session_task
                else:
                    await conversation.shutdown()
            except asyncio.CancelledError:
                reason = "stopped"
                raise
            except Exception as error:
                reason = "error"
                self._connection_state = Conversation.ConnectionState.DISCONNECTED
                self._connection_error = f"{type(error).__name__}: {error}"
                logger.warning(
                    "Live session failed: attempt=%d elapsed=%.1fs error=%s",
                    attempt,
                    time.monotonic() - started_at,
                    self._connection_error,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
            finally:
                session_task.cancel()
                restart_task.cancel()
                await asyncio.gather(session_task, restart_task, return_exceptions=True)
                if self._stop_event.is_set():
                    reason = "stopped"
                elif self._restart_requested.is_set():
                    reason = "requested_restart"
                logger.log(
                    logging.WARNING if reason == "remote_close" else logging.INFO,
                    "Live session ended: attempt=%d reason=%s elapsed=%.1fs",
                    attempt,
                    reason,
                    time.monotonic() - started_at,
                )
            if self._stop_event.is_set():
                return
            history = conversation.history.copy()
            conversation = self._conversation_factory(self._voice)
            if self._restart_requested.is_set():
                continue
            conversation.history = history
            logger.info("Live reconnect scheduled: delay=%.1fs", RETRY_DELAY_SECONDS)
            await self._wait_for_restart(RETRY_DELAY_SECONDS)

    async def _wait_for_restart(self, timeout: float) -> None:
        if self._restart_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._restart_requested.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return

    def launch(self) -> None:
        """Start media and run the Live, capture, and playback loops."""
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
                asyncio.create_task(self._run_session_loop(), name="live-session"),
                asyncio.create_task(self.record_loop(), name="audio-capture"),
                asyncio.create_task(self.play_loop(), name="audio-playback"),
                asyncio.create_task(self._acknowledge_playback_loop(), name="audio-playback-tracking"),
            ]
            try:
                completed, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in completed:
                    await task
            except asyncio.CancelledError:
                logger.info("Conversation tasks cancelled")
            finally:
                try:
                    await self._conversation.shutdown()
                finally:
                    for task in self._tasks:
                        task.cancel()
                    await asyncio.gather(*self._tasks, return_exceptions=True)

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

    def _clear_player(self) -> None:
        logger.info("Clearing robot playback: pending_acknowledgements=%d", self._playback_acknowledgements.qsize())
        while not self._playback_acknowledgements.empty():
            try:
                self._playback_acknowledgements.get_nowait()
            except asyncio.QueueEmpty:
                break
        audio = self._robot.media.audio
        if audio is None:
            logger.warning("Cannot clear robot playback: audio output is unavailable")
            return
        try:
            audio.clear_player()
        except Exception as error:
            logger.warning("Failed to clear robot playback: %s", error)

    async def record_loop(self) -> None:
        """Forward Reachy microphone frames to the active Live session."""
        try:
            sample_rate = self._robot.media.get_input_audio_samplerate()
        except Exception:
            logger.exception("Failed to read the microphone sample rate")
            raise
        logger.info("Microphone capture loop started: sample_rate=%d Hz", sample_rate)
        last_frame_at = time.monotonic()
        last_warning_at = float("-inf")
        capture_started = False
        capture_stalled = False
        while not self._stop_event.is_set():
            try:
                audio = self._robot.media.get_audio_sample()
            except Exception:
                logger.exception(
                    "Failed to read a microphone frame: sample_rate=%d Hz muted=%s live_connected=%s",
                    sample_rate,
                    self._mic_muted,
                    self._conversation.connected,
                )
                raise
            now = time.monotonic()
            samples = None if audio is None else np.asarray(audio, dtype=np.float32)
            if samples is None or samples.size == 0:
                missing_duration = now - last_frame_at
                if (
                    missing_duration >= MICROPHONE_FRAME_TIMEOUT_SECONDS
                    and now - last_warning_at >= AUDIO_WARNING_INTERVAL_SECONDS
                ):
                    logger.warning(
                        "No usable microphone frames received for %.1fs "
                        "(sample_rate=%d Hz, muted=%s, live_connected=%s); restarting robot media",
                        missing_duration,
                        sample_rate,
                        self._mic_muted,
                        self._conversation.connected,
                    )
                    last_warning_at = now
                    capture_stalled = True
                    self._conversation.clear_playback()
                    try:
                        self._robot.media.stop_recording()
                        self._robot.media.start_recording()
                        self._robot.media.start_playing()
                    except Exception as error:
                        logger.warning("Failed to restart robot media after microphone stall: %s", error)
                    try:
                        await self._conversation.interrupt()
                    except Exception as error:
                        logger.warning(
                            "Failed to interrupt Live after microphone recovery: %s: %s", type(error).__name__, error
                        )
                await asyncio.sleep(MICROPHONE_RETRY_DELAY_SECONDS)
                continue
            if capture_stalled:
                logger.info("Microphone capture recovered after %.1fs without frames", now - last_frame_at)
                capture_stalled = False
                last_warning_at = float("-inf")
            if not capture_started:
                logger.info(
                    "Microphone capture started: sample_rate=%d Hz frame_shape=%s dtype=%s peak=%.4f",
                    sample_rate,
                    samples.shape,
                    samples.dtype,
                    float(np.max(np.abs(samples))),
                )
                capture_started = True
            last_frame_at = now
            await self._conversation.receive((sample_rate, np.zeros_like(samples) if self._mic_muted else samples))
            await asyncio.sleep(0)

    async def play_loop(self) -> None:
        """Continuously queue assistant audio for robot playback."""
        while not self._stop_event.is_set():
            conversation = self._conversation
            try:
                audio = await asyncio.wait_for(conversation.emit(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if audio.samples.size == 0:
                continue
            try:
                self._robot.media.push_audio_sample(audio.samples)
            except Exception:
                logger.exception(
                    "Failed to push assistant audio to the robot player: samples=%d",
                    audio.samples.size,
                )
                raise
            self._playback_acknowledgements.put_nowait((conversation, audio))

    async def _acknowledge_playback_loop(self) -> None:
        while True:
            conversation, audio = await self._playback_acknowledgements.get()
            if conversation is not self._conversation:
                continue
            await conversation.acknowledge_after_playback(audio)
            if self._playback_acknowledgements.empty() and conversation.output_queue.empty():
                conversation.acknowledge_playback_end()
