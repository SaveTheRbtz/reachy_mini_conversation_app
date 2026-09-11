import os
import re
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from collections.abc import Callable, AsyncIterator

import numpy as np
import pytest
from PIL import Image
from scipy.signal import resample_poly
from openai.types.live.server_event import ServerEvent

from reachy_mini_conversation_app.config import get_default_voice
from reachy_mini_conversation_app.memory import load_memory
from reachy_mini_conversation_app.realtime import OPENAI_SAMPLE_RATE, LiveConversation
from reachy_mini_conversation_app.tools.types import ToolDependencies


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_ITESTS") != "1",
        reason="set RUN_OPENAI_ITESTS=1 to run paid OpenAI integration tests",
    ),
]
SESSION_TIMEOUT_SECONDS = 90
REACHY_SAMPLE_RATE = 16_000
MICROPHONE_CHUNK_SAMPLES = REACHY_SAMPLE_RATE // 10
FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(autouse=True)
def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        pytest.fail("OPENAI_API_KEY is required when RUN_OPENAI_ITESTS=1")


class ObservedConversation(LiveConversation):
    """Observe production events without replacing transport or tool execution."""

    def __init__(self, dependencies: ToolDependencies) -> None:
        """Collect externally visible conversation outcomes."""
        super().__init__(dependencies, voice=get_default_voice(), output_sample_rate=REACHY_SAMPLE_RATE)
        self.transcript = ""
        self.input_transcript = ""
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
async def _live_session(
    tmp_path: Path, fixture: Path | None = None, camera_frame: np.ndarray | None = None
) -> AsyncIterator[ObservedConversation]:
    robot = MagicMock()
    if camera_frame is not None:
        robot.media.get_frame.return_value = camera_frame
    dependencies = ToolDependencies(
        reachy_mini=robot,
        movement_manager=MagicMock(),
        memory=load_memory(tmp_path),
        instance_path=tmp_path,
    )
    conversation = ObservedConversation(dependencies)
    session = asyncio.create_task(conversation.start_up())
    workers: list[asyncio.Task[None]] = []
    try:
        async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
            while not conversation.connected:
                if session.done():
                    await session
                    pytest.fail("Live session closed before startup")
                await asyncio.sleep(0.01)
            workers = [
                asyncio.create_task(_stream_microphone(conversation, fixture)),
                asyncio.create_task(_play_audio(conversation)),
            ]
            yield conversation
            for worker in workers:
                if worker.done():
                    await worker
            assert not conversation.errors
    except TimeoutError:
        pytest.fail(
            f"Live content timeout; input={conversation.input_transcript!r}; "
            f"output={conversation.transcript!r}; backend_completions={conversation.backend_completions}"
        )
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await conversation.shutdown()
        await asyncio.wait_for(session, timeout=20)
    assert conversation.finalized


async def _wait_for_content(conversation: ObservedConversation, predicate: Callable[[str], bool]) -> str:
    while not predicate(conversation.transcript.casefold()) or conversation.played_samples == 0:
        assert not conversation.errors
        assert conversation.connected
        await asyncio.sleep(0.05)
    return conversation.transcript


async def test_synthesized_speech_drives_production_audio_path(tmp_path: Path) -> None:
    """Receive audible speech while the microphone continues streaming silence."""
    async with _live_session(tmp_path, FIXTURES / "hear_test.pcm") as conversation:
        await _wait_for_content(
            conversation,
            lambda text: "hear" in conversation.input_transcript.casefold() and "hear" in text,
        )
        assert conversation.played_samples > REACHY_SAMPLE_RATE // 10


async def test_synthesized_speech_uses_camera_image(tmp_path: Path) -> None:
    """Execute the production camera tool and speak a grounded visual answer."""
    with Image.open(FIXTURES / "blue_chair.jpg") as image:
        camera_frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
    async with _live_session(tmp_path, FIXTURES / "camera_request.pcm", camera_frame) as conversation:
        await _wait_for_content(conversation, lambda text: "blue" in text and "chair" in text)
        conversation.dependencies.reachy_mini.media.get_frame.assert_called_once_with()


async def test_typed_request_executes_production_tool(tmp_path: Path) -> None:
    """Route typed input through the backend and preserve the robot tool result."""
    async with _live_session(tmp_path) as conversation:
        await conversation.say(
            "Enable head tracking by calling head_tracking with enabled=true. Confirm when enabled."
        )
        await _wait_for_content(conversation, lambda text: "track" in text or "follow" in text)
        conversation.dependencies.movement_manager.set_head_tracking.assert_called_once_with(True)


async def test_hosted_search_returns_a_spoken_answer(tmp_path: Path) -> None:
    """Run native backend web search and return its answer through Live audio."""
    async with _live_session(tmp_path) as conversation:
        await conversation.say(
            "Use web search to check the official Python documentation for the name of its standard library "
            "module for async/await concurrency. Briefly tell me the module name after checking."
        )
        await _wait_for_content(
            conversation, lambda text: conversation.hosted_searches > 0 and "asyncio" in text.replace(" ", "")
        )


async def test_memory_changes_persist_across_live_sessions(tmp_path: Path) -> None:
    """Execute memory updates through Live tools and recall them in a fresh session."""
    async with _live_session(tmp_path) as conversation:
        await conversation.say(
            "Remember this household preference using manage_memory: We love books about space. "
            "Confirm only after saving it."
        )
        await _wait_for_content(
            conversation,
            lambda text: bool(conversation.dependencies.memory.memories) and "space" in text,
        )
        saved = conversation.dependencies.memory.model_copy(deep=True)
    assert load_memory(tmp_path) == saved
    async with _live_session(tmp_path) as conversation:
        await conversation.say("What kind of books does this household like? Answer from shared household memory.")
        await _wait_for_content(conversation, lambda text: "space" in text)
        conversation.transcript = ""
        await conversation.say(
            "Please forget our preference for books about space using manage_memory. Confirm after removing it."
        )
        await _wait_for_content(
            conversation,
            lambda text: not conversation.dependencies.memory.memories and len(text.strip()) > 10,
        )
    assert not load_memory(tmp_path).memories


async def test_default_prompt_withholds_homework_answer_and_answers_facts_directly(tmp_path: Path) -> None:
    """Preserve the learning policy and ordinary factual answers in production Live."""
    async with _live_session(tmp_path) as conversation:
        await conversation.say(
            "This is a homework exercise. Help me solve 9x + 8 = 71; I have not tried anything yet."
        )
        homework = await _wait_for_content(
            conversation,
            lambda text: (
                conversation.backend_completions > 0
                and "?" in text
                and any(word in text for word in ("equation", "subtract", "isolate", "first", "both sides"))
            ),
        )
        assert re.search(r"\b(?:7|seven)\b", homework.casefold()) is None
        conversation.transcript = ""
        await conversation.say("Separate factual question, not an exercise: which planet is known as the Red Planet?")
        await _wait_for_content(conversation, lambda text: "mars" in text)
