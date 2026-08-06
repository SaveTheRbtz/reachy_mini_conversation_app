import os
import json
import asyncio
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock

import pytest
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
    RealtimeModelSendRawMessage,
)

from reachy_mini_conversation_app.config import REALTIME_MODEL
from reachy_mini_conversation_app.memory import MemoryState
from reachy_mini_conversation_app.realtime import create_realtime_agent
from reachy_mini_conversation_app.tools.types import ToolDependencies


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_ITESTS") != "1",
        reason="set RUN_OPENAI_ITESTS=1 to run paid OpenAI integration tests",
    ),
]
TURN_TIMEOUT_SECONDS = 60
ENABLED_TOOLS = ("head_tracking", "remember", "forget")
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


def _model_config() -> RealtimeModelConfig:
    return {
        "initial_model_settings": {
            "model_name": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "max_output_tokens": 512,
            "tool_choice": "none",
            "parallel_tool_calls": False,
            "audio": {
                "output": {"format": "pcm16", "voice": "marin"},
            },
        },
    }


def _dependencies(path: Path, memory: MemoryState) -> tuple[ToolDependencies, MagicMock]:
    movement_manager = MagicMock()
    return (
        ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=movement_manager,
            memory=memory,
            instance_path=path,
        ),
        movement_manager,
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


async def _request_audio(session: RealtimeSession, prompt: str) -> bytes:
    audio = bytearray()
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
    return bytes(audio)


async def test_production_agent_tools_memory_and_audio(tmp_path: Path) -> None:
    """Exercise production agent assembly through real Realtime sessions."""
    memory = MemoryState.load(tmp_path)
    dependencies, movement_manager = _dependencies(tmp_path, memory)
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
    reloaded_dependencies, _ = _dependencies(tmp_path, reloaded_memory)
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
        audio = await _request_audio(
            reloaded_session,
            "Say exactly: Realtime end-to-end test passed.",
        )
        assert audio
