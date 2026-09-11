import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from collections.abc import Callable, AsyncIterator

import numpy as np
import pytest
from scipy.signal import resample_poly
from openai.types.live.server_event import ServerEvent

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.config import get_default_voice
from reachy_mini_conversation_app.memory import load_memory
from reachy_mini_conversation_app.realtime import OPENAI_SAMPLE_RATE, LiveConversation
from reachy_mini_conversation_app.tools.types import ToolDependencies


SESSION_TIMEOUT_SECONDS = 90
REACHY_SAMPLE_RATE = 16_000
MICROPHONE_CHUNK_SAMPLES = REACHY_SAMPLE_RATE // 10
FIXTURES = Path(__file__).parents[1] / "fixtures"


class ObservedConversation(LiveConversation):
    """Observe production events without replacing transport or tool execution."""

    def __init__(self, dependencies: ToolDependencies) -> None:
        """Collect externally visible conversation outcomes."""
        super().__init__(dependencies, voice=get_default_voice(), output_sample_rate=REACHY_SAMPLE_RATE)
        self.transcript = ""
        self.input_transcript = ""
        self.backend_transcript = ""
        self.backend_completions = 0
        self.hosted_searches = 0
        self.errors: list[str] = []
        self.played_samples = 0
        self.finalized = False

    async def _handle_event(self, event: ServerEvent) -> None:
        if event.type == "session.output_transcript.delta":
            self.transcript += event.delta
        elif event.type == "session.input_transcript.delta":
            self.input_transcript += event.delta
        elif event.type == "response.event" and event.event.get("type") == "response.completed":
            self.backend_completions += 1
        elif event.type == "response.event" and event.event.get("type") == "response.output_item.done":
            item = event.event.get("item")
            if isinstance(item, dict) and item.get("type") == "web_search_call":
                self.hosted_searches += 1
            elif isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        self.backend_transcript += content["text"]
        elif event.type == "error":
            self.errors.append(event.error.code)
        elif event.type == "session.closed":
            self.finalized = True
        await super()._handle_event(event)


async def _stream_microphone(conversation: LiveConversation, fixture: Path | None) -> None:
    silence = np.zeros((MICROPHONE_CHUNK_SAMPLES, 2), dtype=np.float32)
    frames = np.zeros((REACHY_SAMPLE_RATE // 2, 2), dtype=np.float32)
    if fixture is not None:
        pcm = np.frombuffer(fixture.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
        mono = np.asarray(resample_poly(pcm, REACHY_SAMPLE_RATE, OPENAI_SAMPLE_RATE), dtype=np.float32)
        frames = np.concatenate((frames, np.column_stack((mono, mono))))
    for offset in range(0, frames.shape[0], MICROPHONE_CHUNK_SAMPLES):
        frame = frames[offset : offset + MICROPHONE_CHUNK_SAMPLES]
        await conversation.receive((REACHY_SAMPLE_RATE, frame))
        await asyncio.sleep(frame.shape[0] / REACHY_SAMPLE_RATE)
    while True:
        await conversation.receive((REACHY_SAMPLE_RATE, silence))
        await asyncio.sleep(MICROPHONE_CHUNK_SAMPLES / REACHY_SAMPLE_RATE)


async def _play_audio(conversation: ObservedConversation) -> None:
    while True:
        audio = await conversation.emit()
        assert audio.samples.dtype == np.float32
        assert audio.samples.size > 0
        await conversation.acknowledge_after_playback(audio)
        conversation.played_samples += audio.samples.size
        if conversation.output_queue.empty():
            conversation.acknowledge_playback_end()


@asynccontextmanager
async def live_session(
    tmp_path: Path, fixture: Path | None = None, camera_frame: np.ndarray | None = None
) -> AsyncIterator[ObservedConversation]:
    """Run the real Live session with simulated hardware and joined audio workers."""
    robot = MagicMock(spec=ReachyMini)
    if camera_frame is not None:
        robot.media.get_frame.return_value = camera_frame
    dependencies = ToolDependencies(
        reachy_mini=robot,
        movement_manager=MagicMock(spec=MovementManager),
        memory=load_memory(tmp_path),
        instance_path=tmp_path,
    )
    conversation = ObservedConversation(dependencies)
    async with asyncio.TaskGroup() as tasks:
        session = tasks.create_task(conversation.start_up())
        workers: list[asyncio.Task[None]] = []
        try:
            async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
                while not conversation.connected:
                    if session.done():
                        await session
                        pytest.fail("Live session closed before startup")
                    await asyncio.sleep(0.01)
                workers = [
                    tasks.create_task(_stream_microphone(conversation, fixture)),
                    tasks.create_task(_play_audio(conversation)),
                ]
                yield conversation
                assert not conversation.errors
        except TimeoutError:
            pytest.fail(
                f"Live content timeout; input={conversation.input_transcript!r}; "
                f"output={conversation.transcript!r}; backend={conversation.backend_transcript!r}; "
                f"backend_completions={conversation.backend_completions}"
            )
        finally:
            for worker in workers:
                worker.cancel()
            await conversation.shutdown()
            await asyncio.wait_for(session, timeout=20)
    assert conversation.finalized


async def wait_for_content(conversation: ObservedConversation, predicate: Callable[[str], bool]) -> str:
    """Wait within the session deadline for grounded speech and audible playback."""
    while not predicate(conversation.transcript.casefold()) or conversation.played_samples == 0:
        assert not conversation.errors
        assert conversation.connected
        await asyncio.sleep(0.05)
    return conversation.transcript
