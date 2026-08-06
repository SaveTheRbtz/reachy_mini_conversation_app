import os
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents import FunctionTool
from agents.realtime import (
    RealtimeAgent,
    RealtimeAudio,
    RealtimeError,
    RealtimeRunner,
    RealtimeToolEnd,
    RealtimeAudioEnd,
    RealtimeRunConfig,
    RealtimeModelConfig,
    RealtimeAgentEndEvent,
    RealtimeModelSendRawMessage,
)

from reachy_mini_conversation_app.config import REALTIME_MODEL
from reachy_mini_conversation_app.memory import MemoryState
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.forget import forget
from reachy_mini_conversation_app.tools.remember import remember


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_ITESTS") != "1",
        reason="set RUN_OPENAI_ITESTS=1 to run paid OpenAI integration tests",
    ),
]
TURN_TIMEOUT_SECONDS = 60
RUN_CONFIG: RealtimeRunConfig = {
    "tracing_disabled": True,
    "async_tool_calls": False,
    "model_settings": {"reasoning": {"effort": "low"}},
}


@pytest.fixture
def openai_api_key() -> str:
    """Return credentials for explicitly enabled OpenAI integration tests."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.fail("OPENAI_API_KEY is required when RUN_OPENAI_ITESTS=1")
    return api_key


def _model_config(api_key: str) -> RealtimeModelConfig:
    return {
        "api_key": api_key,
        "initial_model_settings": {
            "model_name": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "max_output_tokens": 128,
            "audio": {
                "output": {"format": "pcm16", "voice": "marin"},
            },
        },
    }


async def _invoke_forced_tool(
    agent: RealtimeAgent[ToolDependencies],
    dependencies: ToolDependencies,
    api_key: str,
    prompt: str,
    tool_name: str,
) -> RealtimeToolEnd:
    runner = RealtimeRunner(agent, config=RUN_CONFIG)
    tool_event: RealtimeToolEnd | None = None
    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            async with await runner.run(
                context=dependencies,
                model_config=_model_config(api_key),
            ) as session:
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
                    if isinstance(event, RealtimeError):
                        pytest.fail(f"Realtime API error: {event.error}")
                    if isinstance(event, RealtimeToolEnd) and event.tool.name == tool_name:
                        tool_event = event
                        break
    except TimeoutError:
        pytest.fail(f"Realtime did not call {tool_name} within {TURN_TIMEOUT_SECONDS}s")
    if tool_event is None:
        pytest.fail(f"Realtime session closed before calling {tool_name}")
    return tool_event


def _memory_agent(tool: FunctionTool) -> RealtimeAgent[ToolDependencies]:
    return RealtimeAgent[ToolDependencies](
        name="Reachy Mini memory integration test",
        instructions=get_session_instructions,
        tools=[tool],
    )


def _memory_dependencies(path: Path, memory: MemoryState) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        memory=memory,
        instance_path=path,
    )


async def test_gpt_realtime_audio_response(openai_api_key: str) -> None:
    """Receive real PCM audio from the fixed Realtime model."""
    agent = RealtimeAgent(
        name="Reachy Mini Realtime integration test",
        instructions="Reply with one short sentence confirming the Realtime connection works.",
    )
    runner = RealtimeRunner(agent, config=RUN_CONFIG)
    audio = bytearray()
    audio_ended = False
    turn_ended = False

    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            async with await runner.run(model_config=_model_config(openai_api_key)) as session:
                await session.send_message("Confirm this connection in five words or fewer.")
                async for event in session:
                    if isinstance(event, RealtimeError):
                        pytest.fail(f"Realtime API error: {event.error}")
                    if isinstance(event, RealtimeAudio):
                        audio.extend(event.audio.data)
                    elif isinstance(event, RealtimeAudioEnd):
                        audio_ended = True
                    elif isinstance(event, RealtimeAgentEndEvent):
                        turn_ended = True
                        break
    except TimeoutError:
        pytest.fail(f"Realtime audio turn did not finish within {TURN_TIMEOUT_SECONDS}s")

    assert turn_ended
    assert audio_ended
    assert audio
    assert len(audio) % 2 == 0


async def test_memory_persists_across_realtime_sessions(openai_api_key: str, tmp_path: Path) -> None:
    """Save and forget durable context through real Realtime tool calls."""
    memory = MemoryState.load(tmp_path)
    remember_event = await _invoke_forced_tool(
        _memory_agent(remember),
        _memory_dependencies(tmp_path, memory),
        openai_api_key,
        "Save this durable preference verbatim: "
        '"Prefers the integration-test color cobalt". Pass null as replaces_memory_id.',
        "remember",
    )

    remember_arguments = json.loads(remember_event.arguments)
    assert remember_arguments == {
        "fact": "Prefers the integration-test color cobalt",
        "replaces_memory_id": None,
    }
    assert isinstance(remember_event.output, dict)
    assert remember_event.output["status"] == "saved"
    assert [note.text for note in memory.notes] == ["Prefers the integration-test color cobalt"]
    reloaded_memory = MemoryState.load(tmp_path)
    assert reloaded_memory.notes == memory.notes
    memory_id = reloaded_memory.notes[0].id
    assert remember_event.output["memory_id"] == memory_id

    forget_event = await _invoke_forced_tool(
        _memory_agent(forget),
        _memory_dependencies(tmp_path, reloaded_memory),
        openai_api_key,
        "Remove the saved integration-test color preference. Use its exact bracketed ID from <user_memories>.",
        "forget",
    )

    assert json.loads(forget_event.arguments) == {"memory_id": memory_id}
    assert isinstance(forget_event.output, dict)
    assert forget_event.output["removed"] == "Prefers the integration-test color cobalt"
    assert reloaded_memory.notes == []
    assert MemoryState.load(tmp_path).notes == []
