import os
import re
import json
import asyncio
import logging
from typing import Literal
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest
from openai import AsyncOpenAI
from scipy.signal import resample_poly
from agents.realtime import (
    RealtimeAudio,
    RealtimeError,
    RealtimeRunner,
    RealtimeSession,
    RealtimeToolEnd,
    RealtimeAudioEnd,
    RealtimeRunConfig,
    RealtimeModelConfig,
    RealtimeAgentEndEvent,
    RealtimeRawModelEvent,
    RealtimeModelSendRawMessage,
    RealtimeModelTranscriptDeltaEvent,
)

from reachy_mini_conversation_app.config import REALTIME_MODEL
from reachy_mini_conversation_app.memory import MemoryState
from reachy_mini_conversation_app.realtime import OPENAI_SAMPLE_RATE, RealtimeConversation, create_realtime_agent
from reachy_mini_conversation_app.tools.types import ToolDependencies


logger = logging.getLogger(__name__)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_ITESTS") != "1",
        reason="set RUN_OPENAI_ITESTS=1 to run paid OpenAI integration tests",
    ),
]
TURN_TIMEOUT_SECONDS = 60
REACHY_SAMPLE_RATE = 16_000
MICROPHONE_CHUNK_SAMPLES = REACHY_SAMPLE_RATE // 10
ENABLED_TOOLS = ("head_tracking", "remember", "forget")
BLUE_CHAIR_FIXTURE = Path(__file__).parents[1] / "fixtures" / "blue_chair.jpg"
RUN_CONFIG: RealtimeRunConfig = {
    "tracing_disabled": True,
    "async_tool_calls": False,
    "model_settings": {"reasoning": {"effort": "low"}},
}


@pytest.fixture(autouse=True)
def _require_openai_api_key() -> None:
    """Require credentials for explicitly enabled OpenAI integration tests."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        pytest.fail("OPENAI_API_KEY is required when RUN_OPENAI_ITESTS=1")


def _model_config(*, tool_choice: Literal["auto", "none"] = "none") -> RealtimeModelConfig:
    return {
        "initial_model_settings": {
            "model_name": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "max_output_tokens": 512,
            "tool_choice": tool_choice,
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
                "output": {"format": "pcm16", "voice": "coral"},
            },
        },
    }


def _dependencies(path: Path, memory: MemoryState) -> tuple[ToolDependencies, MagicMock, MagicMock]:
    movement_manager = MagicMock()
    reachy_mini = MagicMock()
    return (
        ToolDependencies(
            reachy_mini=reachy_mini,
            movement_manager=movement_manager,
            memory=memory,
            instance_path=path,
        ),
        movement_manager,
        reachy_mini,
    )


async def _invoke_forced_tool(
    session: RealtimeSession,
    dependencies: ToolDependencies,
    prompt: str,
    tool_name: str,
) -> RealtimeToolEnd:
    tool_event: RealtimeToolEnd | None = None
    response_audio = bytearray()
    response_audio_ended = False
    agent_end_count = 0
    observed_events: deque[str] = deque(maxlen=20)

    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            await session.model.send_event(
                RealtimeModelSendRawMessage(
                    message={
                        "type": "conversation.item.create",
                        "other_data": {
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": prompt}],
                            }
                        },
                    }
                )
            )
            await session.model.send_event(
                RealtimeModelSendRawMessage(
                    message={
                        "type": "response.create",
                        "other_data": {
                            "response": {
                                "tool_choice": {"type": "function", "name": tool_name},
                            }
                        },
                    }
                )
            )
            async for event in session:
                observed_events.append(type(event).__name__)
                if isinstance(event, RealtimeError):
                    pytest.fail(f"Realtime API error: {type(event.error).__name__}")
                if isinstance(event, RealtimeToolEnd):
                    if event.tool.name != tool_name:
                        pytest.fail(
                            f"Realtime called unexpected tool {event.tool.name}; events={list(observed_events)}"
                        )
                    tool_event = event
                elif isinstance(event, RealtimeAudio) and tool_event is not None:
                    response_audio.extend(event.audio.data)
                elif isinstance(event, RealtimeAudioEnd) and tool_event is not None:
                    response_audio_ended = True
                elif isinstance(event, RealtimeAgentEndEvent):
                    agent_end_count += 1
                    if tool_event is not None and agent_end_count >= 2:
                        break
    except TimeoutError:
        pytest.fail(
            f"Realtime did not finish {tool_name} within {TURN_TIMEOUT_SECONDS}s; events={list(observed_events)}"
        )

    if tool_event is None:
        pytest.fail(f"Realtime session closed before calling {tool_name}; events={list(observed_events)}")
    assert tool_event.info.context.context is dependencies
    assert response_audio_ended
    assert response_audio
    assert len(response_audio) % 2 == 0
    return tool_event


async def _request_response(session: RealtimeSession, prompt: str) -> tuple[bytes, str]:
    audio = bytearray()
    transcript: list[str] = []
    audio_ended = False
    turn_ended = False
    observed_events: deque[str] = deque(maxlen=20)

    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            await session.send_message(prompt)
            async for event in session:
                observed_events.append(type(event).__name__)
                if isinstance(event, RealtimeError):
                    pytest.fail(f"Realtime API error: {type(event.error).__name__}")
                if isinstance(event, RealtimeToolEnd):
                    pytest.fail(f"Realtime called unexpected tool {event.tool.name}; events={list(observed_events)}")
                if isinstance(event, RealtimeAudio):
                    audio.extend(event.audio.data)
                elif isinstance(event, RealtimeAudioEnd):
                    audio_ended = True
                elif isinstance(event, RealtimeRawModelEvent) and isinstance(
                    event.data, RealtimeModelTranscriptDeltaEvent
                ):
                    transcript.append(event.data.delta)
                elif isinstance(event, RealtimeAgentEndEvent) and audio_ended:
                    turn_ended = True
                    break
    except TimeoutError:
        pytest.fail(
            f"Realtime audio turn did not finish within {TURN_TIMEOUT_SECONDS}s; events={list(observed_events)}"
        )

    assert turn_ended
    assert audio_ended
    assert audio
    assert len(audio) % 2 == 0
    return bytes(audio), "".join(transcript).strip()


async def _send_synthesized_speech(conversation: RealtimeConversation, text: str) -> None:
    async with AsyncOpenAI(timeout=20.0, max_retries=1) as client:
        speech = await client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak clearly and briskly without long pauses.",
            response_format="pcm",
        )
    microphone_pcm = np.frombuffer(speech.content, dtype="<i2").astype(np.float32) / 32768.0
    microphone_mono = np.asarray(
        resample_poly(microphone_pcm, REACHY_SAMPLE_RATE, OPENAI_SAMPLE_RATE),
        dtype=np.float32,
    )
    microphone_stereo = np.column_stack((microphone_mono, microphone_mono))
    assert microphone_stereo.size > 0

    leading_silence = np.zeros((REACHY_SAMPLE_RATE // 2, 2), dtype=np.float32)
    trailing_silence = np.zeros((REACHY_SAMPLE_RATE * 2, 2), dtype=np.float32)
    microphone_stream = np.concatenate((leading_silence, microphone_stereo, trailing_silence))
    for offset in range(0, microphone_stream.shape[0], MICROPHONE_CHUNK_SAMPLES):
        frame = microphone_stream[offset : offset + MICROPHONE_CHUNK_SAMPLES]
        await conversation.receive((REACHY_SAMPLE_RATE, frame))
        await asyncio.sleep(frame.shape[0] / REACHY_SAMPLE_RATE)


async def test_production_agent_tools_memory_and_audio(tmp_path: Path) -> None:
    """Exercise production agent assembly through real Realtime sessions."""
    memory = MemoryState.load(tmp_path)
    dependencies, movement_manager, _ = _dependencies(tmp_path, memory)
    agent = create_realtime_agent(ENABLED_TOOLS)
    assert {tool.name for tool in agent.tools} == set(ENABLED_TOOLS)

    runner = RealtimeRunner(agent, config=RUN_CONFIG)
    async with await runner.run(context=dependencies, model_config=_model_config()) as session:
        tracking_event = await _invoke_forced_tool(
            session,
            dependencies,
            "Enable head tracking. Pass enabled=true.",
            "head_tracking",
        )
        tracking_arguments: object = json.loads(tracking_event.arguments)
        assert tracking_arguments == {"enabled": True}
        assert isinstance(tracking_event.output, dict)
        assert tracking_event.output == {"status": "following"}
        movement_manager.set_head_tracking.assert_called_once_with(True)

        remember_event = await _invoke_forced_tool(
            session,
            dependencies,
            "Save this durable preference verbatim: "
            '"Prefers the integration-test color cobalt". Pass null as replaces_memory_id.',
            "remember",
        )
        remember_arguments: object = json.loads(remember_event.arguments)
        assert remember_arguments == {
            "fact": "Prefers the integration-test color cobalt",
            "replaces_memory_id": None,
        }
        assert isinstance(remember_event.output, dict)
        assert remember_event.output["status"] == "saved"
        assert [note.text for note in memory.notes] == ["Prefers the integration-test color cobalt"]
        assert MemoryState.load(tmp_path).notes == memory.notes

    reloaded_memory = MemoryState.load(tmp_path)
    memory_id = reloaded_memory.notes[0].id
    assert remember_event.output["memory_id"] == memory_id
    reloaded_dependencies, _, _ = _dependencies(tmp_path, reloaded_memory)
    reloaded_agent = create_realtime_agent(ENABLED_TOOLS)
    reloaded_runner = RealtimeRunner(reloaded_agent, config=RUN_CONFIG)

    async with await reloaded_runner.run(
        context=reloaded_dependencies,
        model_config=_model_config(),
    ) as reloaded_session:
        forget_event = await _invoke_forced_tool(
            reloaded_session,
            reloaded_dependencies,
            "Remove the saved integration-test color preference using its identifier from <user_memories>.",
            "forget",
        )
        forget_arguments: object = json.loads(forget_event.arguments)
        assert forget_arguments == {"memory_id": memory_id}
        assert isinstance(forget_event.output, dict)
        assert forget_event.output == {
            "removed": "Prefers the integration-test color cobalt",
            "memory_id": memory_id,
        }
        assert reloaded_memory.notes == []
        assert MemoryState.load(tmp_path).notes == []

        await reloaded_session.update_agent(reloaded_agent)
        audio, transcript = await _request_response(
            reloaded_session,
            "Say exactly: Realtime end-to-end test passed.",
        )
        assert audio
        assert transcript


async def test_default_prompt_withholds_homework_answer_and_answers_facts_directly(tmp_path: Path) -> None:
    """Exercise stable outcomes of the learning policy through a real Realtime session."""
    memory = MemoryState.load(tmp_path)
    dependencies, _, _ = _dependencies(tmp_path, memory)
    agent = create_realtime_agent(())
    runner = RealtimeRunner(agent, config=RUN_CONFIG)

    async with await runner.run(context=dependencies, model_config=_model_config()) as session:
        _, homework_response = await _request_response(
            session,
            "This is a homework exercise. Help me solve 9x + 8 = 71; I have not tried anything yet.",
        )
        _, factual_response = await _request_response(
            session,
            "Separate factual question, not an exercise: which planet is known as the Red Planet?",
        )

    assert homework_response
    assert re.search(r"\b(?:7|seven)\b", homework_response.lower()) is None
    assert "mars" in factual_response.lower()


async def test_synthesized_speech_drives_production_audio_path(tmp_path: Path) -> None:
    """Send synthesized microphone frames through the production Realtime bridge."""
    memory = MemoryState.load(tmp_path)
    dependencies, _, _ = _dependencies(tmp_path, memory)
    conversation = RealtimeConversation(dependencies, voice="coral", output_sample_rate=48_000)
    assistant_audio_bytes = 0
    assistant_audio_ended = False
    assistant_transcript: list[str] = []
    observed_events: deque[str] = deque(maxlen=20)
    phase = "Realtime connection"

    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            agent = create_realtime_agent(())
            runner = RealtimeRunner(agent, config=RUN_CONFIG)
            async with await runner.run(context=dependencies, model_config=_model_config()) as session:
                conversation._session = session
                try:
                    phase = "microphone upload"
                    await _send_synthesized_speech(conversation, "Can you hear this synthesized microphone test?")

                    phase = "Realtime response"
                    async for event in session:
                        observed_events.append(type(event).__name__)
                        await conversation._handle_event(event)
                        if isinstance(event, RealtimeError):
                            pytest.fail(f"Realtime API error: {type(event.error).__name__}")
                        if isinstance(event, RealtimeAudio):
                            assistant_audio_bytes += len(event.audio.data)
                        elif isinstance(event, RealtimeAudioEnd):
                            assistant_audio_ended = True
                        elif isinstance(event, RealtimeRawModelEvent) and isinstance(
                            event.data, RealtimeModelTranscriptDeltaEvent
                        ):
                            assistant_transcript.append(event.data.delta)
                        elif isinstance(event, RealtimeAgentEndEvent) and assistant_audio_ended:
                            break
                    phase = "Realtime cleanup"
                finally:
                    conversation._session = None
    except TimeoutError:
        logger.error(
            "Synthesized speech path timed out during %s; events=%s",
            phase,
            list(observed_events),
        )
        pytest.fail(
            f"Synthesized speech path timed out during {phase} within "
            f"{TURN_TIMEOUT_SECONDS}s; events={list(observed_events)}"
        )

    assert assistant_audio_ended
    assert assistant_audio_bytes > 0
    assert "".join(assistant_transcript).strip()
    playback = await conversation.emit()
    assert playback is not None
    assert playback.samples.size > 0


async def test_synthesized_speech_uses_camera_image(tmp_path: Path) -> None:
    """Exercise spoken camera selection and image-grounded audio output."""
    memory = MemoryState.load(tmp_path)
    dependencies, _, reachy_mini = _dependencies(tmp_path, memory)
    camera_capture = MagicMock(return_value=BLUE_CHAIR_FIXTURE.read_bytes())
    reachy_mini.media.get_frame_jpeg = camera_capture
    conversation = RealtimeConversation(dependencies, voice="coral", output_sample_rate=48_000)
    camera_event: RealtimeToolEnd | None = None
    assistant_audio = bytearray()
    assistant_audio_ended = False
    assistant_transcript: list[str] = []
    observed_events: deque[str] = deque(maxlen=20)
    phase = "Realtime connection"

    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            agent = create_realtime_agent(("camera",))
            runner = RealtimeRunner(agent, config=RUN_CONFIG)
            async with await runner.run(
                context=dependencies,
                model_config=_model_config(tool_choice="auto"),
            ) as session:
                conversation._session = session
                conversation._agent = agent
                dependencies.send_image = conversation._send_image
                try:
                    phase = "spoken camera request"
                    await _send_synthesized_speech(
                        conversation,
                        "Please use your camera. What is the main object you see, and what color is it?",
                    )

                    phase = "camera response"
                    async for event in session:
                        observed_events.append(type(event).__name__)
                        await conversation._handle_event(event)
                        if isinstance(event, RealtimeError):
                            pytest.fail(f"Realtime API error: {type(event.error).__name__}")
                        if isinstance(event, RealtimeToolEnd):
                            if event.tool.name != "camera":
                                pytest.fail(
                                    f"Realtime called unexpected tool {event.tool.name}; events={list(observed_events)}"
                                )
                            if camera_event is not None:
                                pytest.fail(f"Realtime called camera more than once; events={list(observed_events)}")
                            camera_event = event
                            assistant_audio.clear()
                            assistant_audio_ended = False
                            assistant_transcript.clear()
                        elif isinstance(event, RealtimeAudio):
                            assistant_audio.extend(event.audio.data)
                        elif isinstance(event, RealtimeAudioEnd):
                            assistant_audio_ended = True
                        elif isinstance(event, RealtimeRawModelEvent) and isinstance(
                            event.data, RealtimeModelTranscriptDeltaEvent
                        ):
                            assistant_transcript.append(event.data.delta)
                        elif isinstance(event, RealtimeAgentEndEvent):
                            if camera_event is not None and assistant_audio_ended:
                                break
                finally:
                    dependencies.send_image = None
                    conversation._session = None
                    conversation._agent = None
    except TimeoutError:
        logger.error(
            "Spoken camera path timed out during %s; events=%s",
            phase,
            list(observed_events),
        )
        pytest.fail(
            f"Spoken camera path timed out during {phase} within "
            f"{TURN_TIMEOUT_SECONDS}s; events={list(observed_events)}"
        )

    if camera_event is None:
        pytest.fail(f"Realtime session ended without calling camera; events={list(observed_events)}")
    assert isinstance(camera_event.output, dict)
    assert camera_event.output["status"] == "image submitted"
    camera_question = camera_event.output["question"]
    assert isinstance(camera_question, str)
    assert "blue" not in camera_question.lower()
    assert "chair" not in camera_question.lower()
    camera_capture.assert_called_once_with()
    assert assistant_audio_ended
    assert assistant_audio
    assert len(assistant_audio) % 2 == 0
    transcript = "".join(assistant_transcript).lower()
    assert "blue" in transcript or "chair" in transcript
