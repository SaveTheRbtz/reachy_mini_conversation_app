import math
import time
import base64
import asyncio
import logging
from typing import TypeAlias
from dataclasses import dataclass
from collections.abc import Callable, Iterable

import numpy as np
from pydantic import ValidationError
from numpy.typing import NDArray
from scipy.signal import resample_poly
from agents.realtime import (
    RealtimeAgent,
    RealtimeAudio,
    RealtimeError,
    RealtimeRunner,
    RealtimeSession,
    RealtimeToolEnd,
    RealtimeAudioEnd,
    RealtimeRunConfig,
    RealtimeToolStart,
    RealtimeModelConfig,
    RealtimeSessionEvent,
    RealtimeRawModelEvent,
    RealtimePlaybackTracker,
    RealtimeAudioInterrupted,
    RealtimeModelSendRawMessage,
)
from openai.types.realtime import RealtimeError as OpenAIRealtimeError
from openai.types.realtime import ResponseDoneEvent

from reachy_mini_conversation_app.config import (
    REALTIME_MODEL,
    config,
)
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_greeting_prompt
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, selected_tool_names


logger = logging.getLogger(__name__)

OPENAI_SAMPLE_RATE = 24_000
AUDIO_WARNING_INTERVAL_SECONDS = 60.0
REALTIME_AUDIO_SEND_STALL_SECONDS = 1.0
CONTEXT_POST_INSTRUCTIONS_TOKENS = 64_000
CONTEXT_RETENTION_RATIO = 0.8
AudioSamples: TypeAlias = NDArray[np.float32] | NDArray[np.int16]
InputAudioFrame: TypeAlias = tuple[int, AudioSamples]
ActivityObserver: TypeAlias = Callable[[str], None]


def create_realtime_agent(
    enabled_tool_names: Iterable[str],
) -> RealtimeAgent[ToolDependencies]:
    """Build the production Realtime agent for the selected tools."""
    return RealtimeAgent[ToolDependencies](
        name="Reachy Mini",
        instructions=get_session_instructions,
        tools=get_function_tools(enabled_tool_names),
    )


@dataclass(frozen=True)
class PlaybackAudio:
    """One assistant audio chunk and its playback accounting metadata."""

    item_id: str
    content_index: int
    source_pcm16: bytes
    samples: NDArray[np.float32]


class StreamingAudioBridge:
    """Convert streaming Reachy audio to and from OpenAI's PCM16 format."""

    _OVERLAP_SAMPLES = 32

    def __init__(self, output_sample_rate: int) -> None:
        """Initialize persistent input and output resampling state."""
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        self.output_sample_rate = output_sample_rate
        self._microphone_tail = np.empty(0, dtype=np.float32)
        self._playback_tail = np.empty(0, dtype=np.float32)

    @staticmethod
    def _mono_float32(samples: AudioSamples) -> NDArray[np.float32]:
        if samples.ndim == 2:
            if samples.shape[0] < samples.shape[1]:
                samples = samples.T
            return np.asarray(samples.mean(axis=1), dtype=np.float32)
        if samples.dtype == np.int16:
            return np.asarray(samples, dtype=np.float32) / 32768.0
        if samples.dtype == np.float32:
            return np.asarray(samples, dtype=np.float32)
        raise TypeError(f"Unsupported audio dtype: {samples.dtype}")

    @staticmethod
    def _resample(
        samples: NDArray[np.float32],
        source_rate: int,
        target_rate: int,
        tail: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if source_rate <= 0:
            raise ValueError("source sample rate must be positive")
        if samples.size == 0:
            return samples, tail
        overlap_size = min(StreamingAudioBridge._OVERLAP_SAMPLES, samples.size)
        next_tail = samples[-overlap_size:].copy()
        if source_rate == target_rate:
            return samples, next_tail

        joined = np.concatenate((tail, samples)) if tail.size else samples
        divisor = math.gcd(source_rate, target_rate)
        converted = resample_poly(joined, target_rate // divisor, source_rate // divisor)
        converted = np.asarray(converted, dtype=np.float32)
        if tail.size:
            discard = round(tail.size * target_rate / source_rate)
            converted = converted[discard:]
        return converted, next_tail

    def microphone_to_pcm16(self, sample_rate: int, samples: AudioSamples) -> bytes:
        """Convert one microphone frame to 24 kHz mono PCM16 bytes."""
        mono = self._mono_float32(samples)
        converted, self._microphone_tail = self._resample(
            mono,
            sample_rate,
            OPENAI_SAMPLE_RATE,
            self._microphone_tail,
        )
        pcm16 = np.clip(converted, -1.0, 1.0) * 32767.0
        return np.asarray(pcm16, dtype="<i2").tobytes()

    def pcm16_to_playback(self, pcm16: bytes) -> NDArray[np.float32]:
        """Convert one OpenAI PCM16 chunk to the Reachy output sample rate."""
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        converted, self._playback_tail = self._resample(
            samples,
            OPENAI_SAMPLE_RATE,
            self.output_sample_rate,
            self._playback_tail,
        )
        return converted

    def reset_playback(self) -> None:
        """Discard output-side resampling history after interruption."""
        self._playback_tail = np.empty(0, dtype=np.float32)


class RealtimeConversation:
    """Run one OpenAI Realtime conversation for Reachy Mini."""

    def __init__(
        self,
        dependencies: ToolDependencies,
        *,
        voice: str,
        output_sample_rate: int,
    ) -> None:
        """Initialize the session-independent conversation state."""
        self.dependencies = dependencies
        self.voice = voice
        self.output_queue: asyncio.Queue[PlaybackAudio | None] = asyncio.Queue()
        self.last_activity_time = time.monotonic()
        self._session: RealtimeSession | None = None
        self._activity_observer: ActivityObserver | None = None
        self._clear_player: Callable[[], None] | None = None
        self._bridge = StreamingAudioBridge(output_sample_rate)
        self._playback_tracker = RealtimePlaybackTracker()
        self._playback_interrupted = asyncio.Event()
        self._microphone_forwarding_started = False
        self._assistant_audio_item_id: str | None = None
        self._response_started_at: dict[str, float] = {}
        self._tool_started_at: float | None = None
        self._last_audio_send_warning_at = float("-inf")

    @property
    def connected(self) -> bool:
        """Return whether a realtime session is active."""
        return self._session is not None

    def set_activity_observer(self, observer: ActivityObserver | None) -> None:
        """Attach an observer for coarse conversation state changes."""
        self._activity_observer = observer

    def set_clear_player(self, clear_player: Callable[[], None]) -> None:
        """Attach the Reachy playback flush callback."""
        self._clear_player = clear_player

    def _mark_activity(self, reason: str) -> None:
        self.last_activity_time = time.monotonic()
        if self._activity_observer is not None:
            self._activity_observer(reason)

    async def start_up(self) -> None:
        """Connect and process events until the session closes."""
        api_key = (config.OPENAI_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        enabled_tool_names = selected_tool_names(
            str(self.dependencies.instance_path) if self.dependencies.instance_path is not None else None
        )
        logger.info(
            "Realtime configuration: model=%s voice=%s tools=%s context_tokens=%d retention_ratio=%.2f",
            REALTIME_MODEL,
            self.voice,
            ",".join(enabled_tool_names),
            CONTEXT_POST_INSTRUCTIONS_TOKENS,
            CONTEXT_RETENTION_RATIO,
        )
        agent = create_realtime_agent(enabled_tool_names)
        run_config: RealtimeRunConfig = {
            "tracing_disabled": True,
            "async_tool_calls": False,
            "model_settings": {
                "reasoning": {"effort": "low"},
            },
        }
        model_config: RealtimeModelConfig = {
            "api_key": api_key,
            "playback_tracker": self._playback_tracker,
            "initial_model_settings": {
                "model_name": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "parallel_tool_calls": False,
                "audio": {
                    "input": {
                        "format": "pcm16",
                        "noise_reduction": {"type": "near_field"},
                        "turn_detection": {
                            "type": "semantic_vad",
                            "create_response": True,
                            "interrupt_response": True,
                            "eagerness": "auto",
                        },
                    },
                    "output": {"format": "pcm16", "voice": self.voice},
                },
            },
        }
        runner = RealtimeRunner(agent, config=run_config)
        async with await runner.run(context=self.dependencies, model_config=model_config) as session:
            await session.model.send_event(
                RealtimeModelSendRawMessage(
                    message={
                        "type": "session.update",
                        "other_data": {
                            "session": {
                                "type": "realtime",
                                "truncation": {
                                    "type": "retention_ratio",
                                    "retention_ratio": CONTEXT_RETENTION_RATIO,
                                    "token_limits": {
                                        "post_instructions": CONTEXT_POST_INSTRUCTIONS_TOKENS,
                                    },
                                },
                            },
                        },
                    }
                )
            )
            self._session = session
            self.dependencies.send_image = self._send_image
            self._mark_activity("connected")
            try:
                await session.send_message(get_session_greeting_prompt())
                async for event in session:
                    await self._handle_event(event)
            finally:
                self._clear_playback()
                self._response_started_at.clear()
                self._tool_started_at = None
                self.dependencies.send_image = None
                self.dependencies.movement_manager.set_listening(False)
                self.dependencies.movement_manager.set_speaking(False)
                self._session = None
                self._mark_activity("disconnected")

    async def shutdown(self) -> None:
        """Close the active session."""
        session = self._session
        if session is not None:
            await session.close()

    async def receive(self, frame: InputAudioFrame) -> None:
        """Send one Reachy microphone frame to OpenAI."""
        session = self._session
        if session is None:
            return
        sample_rate, samples = frame
        pcm16 = self._bridge.microphone_to_pcm16(sample_rate, samples)
        if pcm16:
            send_started_at = time.monotonic()
            try:
                await session.send_audio(pcm16)
            except Exception:
                logger.exception(
                    "Failed to forward microphone audio to OpenAI Realtime: input_rate=%d Hz pcm16_bytes=%d",
                    sample_rate,
                    len(pcm16),
                )
                raise
            send_finished_at = time.monotonic()
            if not self._microphone_forwarding_started:
                logger.info(
                    "Realtime microphone forwarding started: input_rate=%d Hz pcm16_bytes=%d",
                    sample_rate,
                    len(pcm16),
                )
                self._microphone_forwarding_started = True
            send_duration = send_finished_at - send_started_at
            if (
                send_duration >= REALTIME_AUDIO_SEND_STALL_SECONDS
                and send_finished_at - self._last_audio_send_warning_at >= AUDIO_WARNING_INTERVAL_SECONDS
            ):
                logger.warning(
                    "Realtime microphone forwarding delayed: send_duration=%.3fs pcm16_bytes=%d",
                    send_duration,
                    len(pcm16),
                )
                self._last_audio_send_warning_at = send_finished_at

    async def emit(self) -> PlaybackAudio | None:
        """Wait for the next assistant audio chunk or end marker."""
        return await self.output_queue.get()

    async def say(self, text: str) -> None:
        """Send a text turn to the active realtime session."""
        session = self._session
        if session is None:
            raise RuntimeError("No active realtime session")
        await session.send_message(text)

    async def interrupt(self) -> None:
        """Interrupt the response and clear pending playback."""
        session = self._session
        if session is None:
            self._clear_playback()
            return
        try:
            await session.interrupt()
        finally:
            self._clear_playback()

    async def acknowledge_after_playback(self, audio: PlaybackAudio) -> None:
        """Track one chunk across its estimated robot playback duration."""
        self.dependencies.movement_manager.set_listening(False)
        self.dependencies.movement_manager.set_speaking(True)
        self._mark_activity("speaking")
        duration = audio.samples.size / self._bridge.output_sample_rate
        try:
            await asyncio.wait_for(self._playback_interrupted.wait(), timeout=duration)
        except asyncio.TimeoutError:
            self._playback_tracker.on_play_bytes(audio.item_id, audio.content_index, audio.source_pcm16)

    def acknowledge_playback_end(self) -> None:
        """Mark listening only after all queued assistant audio has played."""
        self.dependencies.movement_manager.set_speaking(False)
        self.dependencies.movement_manager.set_listening(True)
        self._mark_activity("listening")

    async def _handle_event(self, event: RealtimeSessionEvent) -> None:
        if isinstance(event, RealtimeAudio):
            self._playback_interrupted.clear()
            playback_samples = self._bridge.pcm16_to_playback(event.audio.data)
            self.output_queue.put_nowait(
                PlaybackAudio(
                    item_id=event.item_id,
                    content_index=event.content_index,
                    source_pcm16=event.audio.data,
                    samples=playback_samples,
                )
            )
            if self._assistant_audio_item_id != event.item_id:
                response_started_at = self._response_started_at.get(event.audio.response_id)
                logger.info(
                    "Realtime assistant audio received: response_id=%s item_id=%s elapsed_since_response_ms=%s "
                    "pcm16_bytes=%d playback_samples=%d",
                    event.audio.response_id,
                    event.item_id,
                    round((time.monotonic() - response_started_at) * 1000)
                    if response_started_at is not None
                    else None,
                    len(event.audio.data),
                    playback_samples.size,
                )
                self._assistant_audio_item_id = event.item_id
        elif isinstance(event, RealtimeAudioInterrupted):
            logger.info(
                "Realtime audio interrupted: item_id=%s pending_output_chunks=%d",
                event.item_id,
                self.output_queue.qsize(),
            )
            self.dependencies.movement_manager.set_speaking(False)
            self.dependencies.movement_manager.set_listening(True)
            self._clear_playback()
            self._mark_activity("listening")
        elif isinstance(event, RealtimeRawModelEvent):
            if event.data.type == "connection_status":
                logger.info("Realtime transport: status=%s", event.data.status)
            elif event.data.type == "turn_started":
                if event.data.response_id is not None:
                    self._response_started_at[event.data.response_id] = time.monotonic()
                logger.info("Realtime response started: response_id=%s", event.data.response_id)
            elif event.data.type == "raw_server_event":
                raw_server_event: object = event.data.data
                if not isinstance(raw_server_event, dict):
                    return
                event_type: object = raw_server_event.get("type")
                if event_type in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
                    logger.info("Realtime VAD: event=%s item_id=%s", event_type, raw_server_event.get("item_id"))
                    self._mark_activity(
                        "listening" if event_type == "input_audio_buffer.speech_started" else "thinking"
                    )
                elif event_type == "session.created":
                    session_metadata: object = raw_server_event.get("session")
                    if isinstance(session_metadata, dict):
                        logger.info("Realtime session created: session_id=%s", session_metadata.get("id"))
                elif event_type == "response.done":
                    try:
                        response = ResponseDoneEvent.model_validate(raw_server_event).response
                    except ValidationError as error:
                        logger.warning(
                            "Cannot read Realtime response.done diagnostics: errors=%s",
                            error.errors(include_input=False, include_context=False, include_url=False),
                        )
                        return
                    response_started_at = self._response_started_at.pop(response.id, None) if response.id else None
                    details = response.status_details
                    response_error = details.error if details else None
                    usage = response.usage
                    input_details = usage.input_token_details if usage else None
                    logger.log(
                        logging.WARNING if response.status in ("failed", "incomplete") else logging.INFO,
                        "Realtime response finished: response_id=%s status=%s duration_ms=%s reason=%s "
                        "error_code=%s error_type=%s input_tokens=%s cached_tokens=%s output_tokens=%s",
                        response.id,
                        response.status,
                        round((time.monotonic() - response_started_at) * 1000)
                        if response_started_at is not None
                        else None,
                        details.reason if details else None,
                        response_error.code if response_error else None,
                        response_error.type if response_error else None,
                        usage.input_tokens if usage else None,
                        input_details.cached_tokens if input_details else None,
                        usage.output_tokens if usage else None,
                    )
        elif isinstance(event, RealtimeAudioEnd):
            self.output_queue.put_nowait(None)
        elif isinstance(event, RealtimeToolStart):
            self._tool_started_at = time.monotonic()
            logger.info("Tool started: %s", event.tool.name)
            self._mark_activity("thinking")
        elif isinstance(event, RealtimeToolEnd):
            output: object = event.output
            failed = isinstance(output, dict) and "error" in output
            logger.log(
                logging.WARNING if failed else logging.INFO,
                "Tool finished: %s outcome=%s duration_ms=%s error=%s",
                event.tool.name,
                "error" if failed else "returned",
                round((time.monotonic() - self._tool_started_at) * 1000)
                if self._tool_started_at is not None
                else None,
                output.get("error") if isinstance(output, dict) else None,
            )
            self._tool_started_at = None
            self._mark_activity("thinking")
        elif isinstance(event, RealtimeError):
            session_error: object = event.error
            match session_error:
                case OpenAIRealtimeError():
                    logger.error(
                        "Realtime session error: type=%s code=%s param=%s event_id=%s",
                        session_error.type,
                        session_error.code,
                        session_error.param,
                        session_error.event_id,
                    )
                case ValidationError():
                    logger.error(
                        "Realtime validation error: errors=%s",
                        session_error.errors(include_input=False, include_context=False, include_url=False),
                    )
                case {"message": str(message)}:
                    logger.error("Realtime SDK error: %s", message)
                case Exception():
                    logger.error(
                        "Realtime session error: type=%s message=%s", type(session_error).__name__, session_error
                    )
                case _:
                    logger.error("Realtime session error: unexpected_error_type=%s", type(session_error).__name__)

    def _clear_playback(self) -> None:
        self._playback_interrupted.set()
        self._bridge.reset_playback()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._clear_player is not None:
            self._clear_player()

    async def _send_image(self, question: str, jpeg_bytes: bytes) -> None:
        session = self._session
        if session is None:
            raise RuntimeError("No active realtime session")
        image_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('ascii')}"
        await session.model.send_event(
            RealtimeModelSendRawMessage(
                message={
                    "type": "conversation.item.create",
                    "other_data": {
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": question},
                                {"type": "input_image", "image_url": image_url},
                            ],
                        }
                    },
                }
            )
        )
