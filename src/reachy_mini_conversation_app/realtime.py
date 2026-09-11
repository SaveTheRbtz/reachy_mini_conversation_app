import json
import time
import base64
import asyncio
import logging
from uuid import uuid4
from typing import Final, Literal, TypeAlias
from dataclasses import field, dataclass
from collections.abc import Callable

import soxr
import numpy as np
from agents import FunctionTool
from openai import AsyncOpenAI
from pydantic import ValidationError
from numpy.typing import NDArray
from agents.tool_context import ToolContext
from websockets.exceptions import ConnectionClosed
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
)
from openai.resources.live.live import AsyncLiveConnection
from openai.types.live.server_event import ServerEvent
from openai.types.live.session_config_param import SessionConfigParam

from reachy_mini_conversation_app.config import LIVE_MODEL, DELEGATION_MODEL, config
from reachy_mini_conversation_app.prompts import (
    get_backend_instructions,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, selected_tool_names


logger = logging.getLogger(__name__)

OPENAI_SAMPLE_RATE: Final = 24_000
AUDIO_WARNING_INTERVAL_SECONDS = 60.0
SEND_TIMEOUT_SECONDS = 5.0
SESSION_TIMEOUT_SECONDS = 15.0
TOOL_TIMEOUT_SECONDS = 30.0
HISTORY_MAX_MESSAGES = 32
HISTORY_MAX_BYTES = 4096  # Leave room for message overhead below Live's 8,192-token limit.
AudioSamples: TypeAlias = NDArray[np.float32] | NDArray[np.int16]
InputAudioFrame: TypeAlias = tuple[int, AudioSamples]
ActivityObserver: TypeAlias = Callable[[str], None]


@dataclass(frozen=True)
class PlaybackAudio:
    """One chunk of assistant audio at the robot's playback sample rate."""

    samples: NDArray[np.float32]


@dataclass
class BackendResponse:
    """Function calls collected from one delegated Responses stream."""

    response_id: str
    delegation_id: str | None
    calls: dict[str, ResponseFunctionToolCall] = field(default_factory=dict)


class StreamingAudioBridge:
    """Convert streaming Reachy audio to and from OpenAI's PCM16 format."""

    def __init__(self, output_sample_rate: int) -> None:
        """Initialize persistent input and output resampling state."""
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        self.output_sample_rate = output_sample_rate
        self._microphone_sample_rate: int | None = None
        self._microphone_resampler: soxr.ResampleStream | None = None
        self._playback_resampler = soxr.ResampleStream(OPENAI_SAMPLE_RATE, output_sample_rate, 1)

    def microphone_to_pcm16(self, sample_rate: int, samples: AudioSamples) -> bytes:
        """Convert one microphone frame to 24 kHz mono PCM16 bytes."""
        if samples.dtype == np.int16:
            mono = np.asarray(samples, dtype=np.float32) / 32768.0
        elif samples.dtype == np.float32:
            mono = np.asarray(samples, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported audio dtype: {samples.dtype}")
        if mono.ndim == 2:
            if mono.shape[0] < mono.shape[1]:
                mono = mono.T
            mono = mono.mean(axis=1)
        if self._microphone_resampler is None or self._microphone_sample_rate != sample_rate:
            self._microphone_resampler = soxr.ResampleStream(sample_rate, OPENAI_SAMPLE_RATE, 1)
            self._microphone_sample_rate = sample_rate
        converted = self._microphone_resampler.resample_chunk(mono)
        pcm16 = np.clip(converted, -1.0, 1.0) * 32767.0
        return np.asarray(pcm16, dtype="<i2").tobytes()

    def pcm16_to_playback(self, pcm16: bytes) -> NDArray[np.float32]:
        """Resample a chunk of the continuous OpenAI PCM16 output stream."""
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return np.asarray(self._playback_resampler.resample_chunk(samples), dtype=np.float32)

    def reset_playback(self) -> None:
        """Discard output-side resampling history after interruption."""
        self._playback_resampler.clear()


class LiveConversation:
    """Run a GPT-Live voice session with a delegated Responses tool backend."""

    def __init__(
        self,
        dependencies: ToolDependencies,
        *,
        voice: str,
        output_sample_rate: int,
    ) -> None:
        """Initialize conversation, tool execution, and continuous audio state."""
        self.dependencies = dependencies
        self.voice = voice
        self.history: list[tuple[Literal["user", "assistant"], str]] = []
        self._history_role: Literal["user", "assistant"] | None = None
        self.output_queue: asyncio.Queue[PlaybackAudio] = asyncio.Queue()
        self._microphone_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self.last_activity_time = time.monotonic()
        self.usage_seconds = 0.0
        self._connection: AsyncLiveConnection | None = None
        self._transport: AsyncLiveConnection | None = None
        self._closed = asyncio.Event()
        self._closing = False
        self._activity_observer: ActivityObserver | None = None
        self._clear_player: Callable[[], None] | None = None
        self._bridge = StreamingAudioBridge(output_sample_rate)
        self._playback_interrupted = asyncio.Event()
        self._backend_busy = False
        self._pending_input = False
        self._tools: dict[str, FunctionTool] = {}
        self._responses: dict[str | None, BackendResponse] = {}
        self._tool_batches: asyncio.Queue[BackendResponse] = asyncio.Queue()
        self._handled_call_ids: set[str] = set()
        self._backend_commands: set[str] = set()
        self._last_audio_drop_warning_at = float("-inf")

    @property
    def connected(self) -> bool:
        """Return whether the Live session is ready for application commands."""
        return self._connection is not None and not self._closing

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

    def _remember(self, role: Literal["user", "assistant"], text: str, *, new_message: bool = False) -> None:
        if not text:
            return
        if not new_message and self._history_role == role and self.history:
            text = self.history.pop()[1] + text
        text = text.encode("utf-8")[-HISTORY_MAX_BYTES:].decode("utf-8", errors="ignore")
        self.history.append((role, text))
        self._history_role = None if new_message else role
        while (
            len(self.history) > HISTORY_MAX_MESSAGES
            or sum(len(content.encode("utf-8")) for _, content in self.history) > HISTORY_MAX_BYTES
        ):
            self.history.pop(0)

    async def start_up(self) -> None:
        """Connect and receive Live events until the session finalizes."""
        if self._closing:
            return
        api_key = (config.OPENAI_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        enabled_names = selected_tool_names(
            str(self.dependencies.instance_path) if self.dependencies.instance_path is not None else None
        )
        self._tools = {tool.name: tool for tool in get_function_tools(enabled_names)}
        instructions = get_session_instructions(enabled_names)
        if self.history:
            instructions += (
                "\n\nThe connection was interrupted. Use the supplied dialogue only as context. "
                "Wait for the user's next request instead of repeating the opening greeting, "
                "resuming unfinished work, or repeating earlier tool actions. "
                "Prior assistant speech may have been interrupted before the user heard it."
            )
        session: SessionConfigParam = {
            "model": LIVE_MODEL,
            "instructions": instructions,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": text}]}
                if role == "user"
                else {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
                for role, text in self.history
            ],
            "audio": {
                "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                "output": {"voice": self.voice},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": DELEGATION_MODEL,
                    "reasoning": {"effort": "low"},
                    "instructions": get_backend_instructions(self.dependencies),
                    "parallel_tool_calls": False,
                    "tools": [
                        {
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.params_json_schema,
                            "strict": tool.strict_json_schema,
                        }
                        for tool in self._tools.values()
                    ],
                },
            },
        }
        logger.info(
            "Live configuration: model=%s backend=%s voice=%s tools=%s",
            LIVE_MODEL,
            DELEGATION_MODEL,
            self.voice,
            ",".join(enabled_names),
        )
        async with AsyncOpenAI(api_key=api_key) as client:
            # LocalStream owns reconnects; a new transport needs a fresh Live session.
            async with client.live.connect(max_retries=0) as connection:
                self._transport = connection
                try:
                    if self._closing:
                        return
                    async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
                        await connection.session.start(session=session)
                        while self._connection is None:
                            event = await connection.recv()
                            if event.type == "session.started":
                                self._connection = connection
                                logger.info("Live session started: session_id=%s", event.session.id)
                                self._mark_activity("connected")
                            elif event.type == "error":
                                raise RuntimeError(
                                    f"Live startup failed: type={event.error.type} code={event.error.code}"
                                )
                            elif event.type == "session.closed":
                                await self._handle_event(event)
                                raise RuntimeError("Live session closed before startup")
                        if not self.history:
                            await connection.session.instructions.append(
                                event_id="greeting",
                                delegation_id=None,
                                content=get_session_greeting_prompt(),
                            )
                    async with asyncio.TaskGroup() as tasks:
                        tool_worker = tasks.create_task(self._run_tools())
                        microphone_sender = tasks.create_task(self._send_microphone(connection))
                        try:
                            async for event in connection:
                                await self._handle_event(event)
                                if self._closed.is_set():
                                    break
                            if not self._closed.is_set():
                                raise RuntimeError("Live connection closed without final session usage")
                        finally:
                            tool_worker.cancel()
                            microphone_sender.cancel()
                finally:
                    if self._connection is not None and not self._closed.is_set():
                        logger.warning(
                            "Live transport ended without finalization; final usage is unconfirmed: seconds=%.3f",
                            self.usage_seconds,
                        )
                    self._closing = True
                    self._backend_busy = False
                    self._pending_input = False
                    self._backend_commands.clear()
                    self.clear_playback()
                    self.dependencies.movement_manager.set_speaking(False)
                    self._connection = None
                    self._transport = None
                    self._mark_activity("disconnected")
                    await self._close_transport(connection)

    async def shutdown(self) -> None:
        """Request finalization and keep receiving until final usage arrives."""
        connection = self._connection
        already_closing = self._closing
        self._closing = True
        if connection is not None and not self._closed.is_set():
            try:
                async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
                    if not already_closing:
                        await connection.session.close()
                    await self._closed.wait()
            except (TimeoutError, OSError, ConnectionClosed) as error:
                logger.warning("Live finalization failed; final usage is unconfirmed: %s", type(error).__name__)
        if self._transport is not None:
            await self._close_transport(self._transport)

    async def _close_transport(self, connection: AsyncLiveConnection) -> None:
        try:
            async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
                await connection.close()
        except TimeoutError:
            logger.warning("Live transport close stalled; forcing the closing handshake to finish")
            # A second close enforces the SDK's expired close deadline without waiting for the send buffer.
            await connection.close()

    async def receive(self, frame: InputAudioFrame) -> None:
        """Queue microphone audio without blocking robot capture on network writes."""
        connection = self._connection
        if connection is None or self._closing:
            return
        sample_rate, samples = frame
        pcm16 = self._bridge.microphone_to_pcm16(sample_rate, samples)
        if not pcm16:
            return
        if self._microphone_queue.full():
            self._microphone_queue.get_nowait()
            now = time.monotonic()
            if now - self._last_audio_drop_warning_at >= AUDIO_WARNING_INTERVAL_SECONDS:
                logger.warning("Live microphone send fell behind; discarding queued audio")
                self._last_audio_drop_warning_at = now
        self._microphone_queue.put_nowait(pcm16)

    async def _send_microphone(self, connection: AsyncLiveConnection) -> None:
        while True:
            pcm16 = await self._microphone_queue.get()
            try:
                async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
                    await connection.session.input_audio.append(audio=base64.b64encode(pcm16).decode("ascii"))
            except (TimeoutError, OSError, ConnectionClosed) as error:
                logger.warning("Live microphone send failed: %s", type(error).__name__)
                raise

    async def emit(self) -> PlaybackAudio:
        """Wait for the next chunk of assistant audio."""
        return await self.output_queue.get()

    async def say(self, text: str) -> None:
        """Submit typed user input to the delegated backend."""
        connection = self._connection
        if connection is None or self._closing:
            raise RuntimeError("No active Live session")
        self._remember("user", text, new_message=True)
        event_id = str(uuid4())
        self._backend_commands.add(event_id)
        await connection.response.item.create(
            event_id=event_id,
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )
        event_id = str(uuid4())
        self._backend_commands.add(event_id)
        await connection.session.instructions.append(
            event_id=event_id,
            delegation_id=None,
            content=(
                "The user has sent a typed request to the backend. "
                "Answer that request aloud when its result arrives, then listen."
            ),
        )
        # A pending tool batch must supply every result before any continuation.
        if self._backend_busy:
            self._pending_input = True
        else:
            self._backend_busy = True
            event_id = str(uuid4())
            self._backend_commands.add(event_id)
            await connection.response.create(event_id=event_id)
        self._mark_activity("thinking")

    async def interrupt(self) -> None:
        """Clear local playback and ask Live to stop speaking and listen."""
        self.clear_playback()
        self.acknowledge_playback_end()
        connection = self._connection
        if connection is not None and not self._closing:
            async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
                await connection.session.instructions.append(
                    event_id=str(uuid4()),
                    delegation_id=None,
                    content="Stop speaking now and listen. Wait for the user's next request.",
                )

    async def acknowledge_after_playback(self, audio: PlaybackAudio) -> None:
        """Track a queued chunk for its estimated robot playback duration."""
        self.dependencies.movement_manager.set_speaking(True)
        self._mark_activity("speaking")
        duration = audio.samples.size / self._bridge.output_sample_rate
        try:
            await asyncio.wait_for(self._playback_interrupted.wait(), timeout=duration)
        except TimeoutError:
            return

    def acknowledge_playback_end(self) -> None:
        """Mark listening when the local playback queue has drained."""
        self.dependencies.movement_manager.set_speaking(False)
        self._mark_activity("listening")

    async def _handle_event(self, event: ServerEvent) -> None:
        if event.type == "session.output_audio.delta":
            if self._closing:
                return
            self._playback_interrupted.clear()
            samples = self._bridge.pcm16_to_playback(base64.b64decode(event.delta))
            if samples.size:
                self.output_queue.put_nowait(PlaybackAudio(samples))
        elif event.type == "session.input_transcript.delta":
            self._remember("user", event.delta)
            self._mark_activity("listening")
        elif event.type == "session.output_transcript.delta":
            self._remember("assistant", event.delta)
        elif event.type == "session.delegation.created":
            self._mark_activity("thinking")
        elif event.type == "response.event":
            nested_type = event.event.get("type")
            try:
                if nested_type == "response.created":
                    response = ResponseCreatedEvent.model_validate(event.event).response
                    self._backend_busy = True
                    self._responses[event.delegation_id] = BackendResponse(response.id, event.delegation_id)
                    logger.info(
                        "Live backend started: response_id=%s delegation_id=%s", response.id, event.delegation_id
                    )
                elif nested_type == "response.output_item.done":
                    item = ResponseOutputItemDoneEvent.model_validate(event.event).item
                    if item.type == "function_call":
                        batch = self._responses.get(event.delegation_id)
                        if batch is None:
                            raise RuntimeError("Live function call arrived without a backend response")
                        batch.calls[item.call_id] = item
                elif nested_type == "response.completed":
                    response = ResponseCompletedEvent.model_validate(event.event).response
                    batch = self._responses.pop(event.delegation_id, None)
                    usage = response.usage
                    logger.info(
                        "Live backend finished: response_id=%s input_tokens=%s output_tokens=%s",
                        response.id,
                        usage.input_tokens if usage else None,
                        usage.output_tokens if usage else None,
                    )
                    if batch is not None and not self._closing:
                        self._tool_batches.put_nowait(batch)
                elif nested_type in {"response.failed", "response.incomplete", "response.cancelled", "error"}:
                    logger.error("Live backend failed: event=%s delegation_id=%s", nested_type, event.delegation_id)
                    batch = self._responses.get(event.delegation_id)
                    if batch is None:
                        if nested_type == "error":
                            raise RuntimeError("Live backend error without an active response")
                        return
                    failed_response = event.event.get("response")
                    if isinstance(failed_response, dict) and failed_response.get("id") != batch.response_id:
                        return
                    self._responses.pop(event.delegation_id)
                    batch.calls.clear()
                    if not self._closing:
                        self._tool_batches.put_nowait(batch)
            except ValidationError as error:
                logger.error(
                    "Invalid Live backend event: type=%s errors=%s",
                    nested_type,
                    error.errors(include_input=False, include_context=False, include_url=False),
                )
                raise RuntimeError("Invalid Live backend event") from None
        elif event.type == "session.usage.updated":
            self.usage_seconds = event.usage.seconds
        elif event.type == "session.closed":
            self.usage_seconds = event.usage.seconds
            self._closing = True
            self._closed.set()
            logger.info("Live session closed: reason=%s seconds=%.3f", event.reason, self.usage_seconds)
        elif event.type == "error":
            logger.error(
                "Live error: type=%s code=%s command=%s",
                event.error.type,
                event.error.code,
                event.error.client_event_id,
            )
            if event.error.client_event_id in self._backend_commands:
                self._closing = True
                raise RuntimeError("Live backend command rejected")

    async def _run_tools(self) -> None:
        while True:
            batch = await self._tool_batches.get()
            try:
                connection = self._connection
                if connection is None or self._closing:
                    continue
                for call in batch.calls.values():
                    if call.call_id in self._handled_call_ids:
                        continue
                    self._handled_call_ids.add(call.call_id)
                    tool = self._tools.get(call.name)
                    started_at = time.monotonic()
                    logger.info(
                        "Tool started: %s call_id=%s delegation_id=%s response_id=%s",
                        call.name,
                        call.call_id,
                        batch.delegation_id,
                        batch.response_id,
                    )
                    self._mark_activity("thinking")
                    output: object
                    if tool is None:
                        logger.warning("Rejected unavailable tool: %s", call.name)
                        output = {"error": f"Tool is unavailable: {call.name}"}
                    else:
                        context = ToolContext(
                            context=self.dependencies,
                            tool_name=call.name,
                            tool_call_id=call.call_id,
                            tool_arguments=call.arguments,
                        )
                        try:
                            async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
                                output = await tool.on_invoke_tool(context, call.arguments)
                        except Exception as error:
                            logger.exception("Tool failed: %s", call.name)
                            output = {"error": f"{call.name} failed: {type(error).__name__}: {error}"}
                    logger.info(
                        "Tool finished: %s duration_ms=%d", call.name, round((time.monotonic() - started_at) * 1000)
                    )
                    if self._closing:
                        return
                    event_id = str(uuid4())
                    self._backend_commands.add(event_id)
                    await connection.response.item.create(
                        event_id=event_id,
                        item={
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output if isinstance(output, str) else json.dumps(output),
                        },
                    )
                if not batch.calls:
                    if self._responses or not self._tool_batches.empty():
                        continue
                    if not self._pending_input:
                        self._backend_busy = False
                        continue
                self._pending_input = False
                event_id = str(uuid4())
                self._backend_commands.add(event_id)
                await connection.session.update(
                    event_id=event_id,
                    session={
                        "delegation": {
                            "type": "responses",
                            "responses": {
                                "instructions": get_backend_instructions(self.dependencies),
                            },
                        },
                    },
                )
                event_id = str(uuid4())
                self._backend_commands.add(event_id)
                await connection.response.create(event_id=event_id)
            finally:
                self._tool_batches.task_done()

    def clear_playback(self) -> None:
        """Discard queued playback and resampling state without sending Live commands."""
        self._playback_interrupted.set()
        self._bridge.reset_playback()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        if self._clear_player is not None:
            self._clear_player()
