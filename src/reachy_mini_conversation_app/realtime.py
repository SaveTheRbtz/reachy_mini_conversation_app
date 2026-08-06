import math
import time
import base64
import asyncio
import logging
from typing import TypeAlias
from dataclasses import dataclass
from collections.abc import Callable, Iterable

import numpy as np
from agents.mcp import MCPServer
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
    RealtimeAgentEndEvent,
    RealtimePlaybackTracker,
    RealtimeAudioInterrupted,
    RealtimeModelSendRawMessage,
)

from reachy_mini_conversation_app.config import (
    REALTIME_MODEL,
    config,
)
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_greeting_prompt
from reachy_mini_conversation_app.mcp_servers import log_mcp_failures, create_mcp_manager
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, selected_tool_names


logger = logging.getLogger(__name__)

OPENAI_SAMPLE_RATE = 24_000
AudioSamples: TypeAlias = NDArray[np.float32] | NDArray[np.int16]
InputAudioFrame: TypeAlias = tuple[int, AudioSamples]
ActivityObserver: TypeAlias = Callable[[str], None]


def create_realtime_agent(
    enabled_tool_names: Iterable[str],
    *,
    mcp_servers: Iterable[MCPServer] = (),
) -> RealtimeAgent[ToolDependencies]:
    """Build the production Realtime agent for the selected tools."""
    return RealtimeAgent[ToolDependencies](
        name="Reachy Mini",
        instructions=get_session_instructions,
        tools=get_function_tools(enabled_tool_names),
        mcp_servers=list(mcp_servers),
        mcp_config={"include_server_in_tool_names": True},
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
        self._agent: RealtimeAgent[ToolDependencies] | None = None
        self._injected_memory_revision = dependencies.memory.revision
        self._activity_observer: ActivityObserver | None = None
        self._clear_player: Callable[[], None] | None = None
        self._bridge = StreamingAudioBridge(output_sample_rate)
        self._playback_tracker = RealtimePlaybackTracker()
        self._playback_interrupted = asyncio.Event()

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
        async with create_mcp_manager(enabled_tool_names) as mcp_manager:
            log_mcp_failures(mcp_manager)
            agent = create_realtime_agent(
                enabled_tool_names,
                mcp_servers=mcp_manager.active_servers,
            )
            run_config: RealtimeRunConfig = {
                "tracing_disabled": True,
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
                self._session = session
                self._agent = agent
                self._injected_memory_revision = self.dependencies.memory.revision
                self.dependencies.send_image = self._send_image
                self._mark_activity("connected")
                try:
                    await session.send_message(get_session_greeting_prompt())
                    async for event in session:
                        await self._handle_event(event)
                finally:
                    self.dependencies.send_image = None
                    self.dependencies.movement_manager.set_listening(False)
                    self.dependencies.movement_manager.set_speaking(False)
                    self._session = None
                    self._agent = None
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
            await session.send_audio(pcm16)

    async def emit(self) -> PlaybackAudio:
        """Wait for the next assistant audio chunk."""
        while True:
            audio = await self.output_queue.get()
            if audio is not None:
                return audio
            self.dependencies.movement_manager.set_speaking(False)
            self.dependencies.movement_manager.set_listening(True)
            self._mark_activity("listening")

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
            return
        await session.interrupt()
        self._clear_playback()

    async def acknowledge_after_playback(self, audio: PlaybackAudio) -> None:
        """Report a chunk only after its estimated robot playback duration."""
        duration = audio.samples.size / self._bridge.output_sample_rate
        try:
            await asyncio.wait_for(self._playback_interrupted.wait(), timeout=duration)
        except asyncio.TimeoutError:
            self._playback_tracker.on_play_bytes(audio.item_id, audio.content_index, audio.source_pcm16)

    async def _handle_event(self, event: RealtimeSessionEvent) -> None:
        if isinstance(event, RealtimeAudio):
            self._playback_interrupted.clear()
            self.dependencies.movement_manager.set_listening(False)
            self.dependencies.movement_manager.set_speaking(True)
            self.output_queue.put_nowait(
                PlaybackAudio(
                    item_id=event.item_id,
                    content_index=event.content_index,
                    source_pcm16=event.audio.data,
                    samples=self._bridge.pcm16_to_playback(event.audio.data),
                )
            )
            self._mark_activity("speaking")
        elif isinstance(event, RealtimeAudioInterrupted):
            self.dependencies.movement_manager.set_speaking(False)
            self.dependencies.movement_manager.set_listening(True)
            self._clear_playback()
            self._mark_activity("listening")
        elif isinstance(event, RealtimeAudioEnd):
            self.output_queue.put_nowait(None)
        elif isinstance(event, RealtimeToolStart):
            logger.info("Tool started: %s", event.tool.name)
            self._mark_activity("thinking")
        elif isinstance(event, RealtimeToolEnd):
            logger.info("Tool finished: %s", event.tool.name)
            self._mark_activity("thinking")
        elif isinstance(event, RealtimeAgentEndEvent):
            await self._refresh_memory_instructions()
        elif isinstance(event, RealtimeError):
            logger.error("Realtime session error: %s", event.error)

    async def _refresh_memory_instructions(self) -> None:
        revision = self.dependencies.memory.revision
        if revision == self._injected_memory_revision:
            return
        session = self._session
        agent = self._agent
        if session is None or agent is None:
            return
        try:
            await session.update_agent(agent)
        except Exception as error:
            logger.warning("Failed to refresh Realtime memory instructions; reconnecting: %s", error)
            await session.close()
            return
        self._injected_memory_revision = revision

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
