import base64
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from agents.realtime import (
    RealtimeAudio,
    RealtimeAudioEnd,
    RealtimeEventInfo,
    RealtimeAgentEndEvent,
    RealtimeModelAudioEvent,
    RealtimeAudioInterrupted,
    RealtimeModelSendRawMessage,
)

import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.memory import MemoryState
from reachy_mini_conversation_app.realtime import PlaybackAudio, RealtimeConversation, StreamingAudioBridge


def _conversation(output_rate: int = 48_000) -> RealtimeConversation:
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemoryState(),
        movement_manager=SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock()),
    )
    return RealtimeConversation(dependencies, voice="marin", output_sample_rate=output_rate)


def test_audio_bridge_converts_stereo_float_to_24khz_pcm16() -> None:
    """Convert robot stereo input to the Realtime PCM format."""
    bridge = StreamingAudioBridge(output_sample_rate=48_000)
    stereo = np.stack(
        (
            np.linspace(-0.5, 0.5, 160, dtype=np.float32),
            np.linspace(0.5, -0.5, 160, dtype=np.float32),
        )
    )

    pcm16 = bridge.microphone_to_pcm16(16_000, stereo)

    assert len(pcm16) == 240 * 2
    assert np.frombuffer(pcm16, dtype="<i2").dtype == np.dtype("int16")


def test_audio_bridge_converts_openai_pcm_to_robot_rate() -> None:
    """Convert Realtime PCM output to the configured robot rate."""
    bridge = StreamingAudioBridge(output_sample_rate=48_000)
    pcm16 = np.arange(240, dtype="<i2").tobytes()

    playback = bridge.pcm16_to_playback(pcm16)

    assert playback.dtype == np.float32
    assert playback.shape == (480,)


@pytest.mark.asyncio
async def test_audio_event_keeps_source_pcm_for_delayed_playback_accounting() -> None:
    """Keep source PCM metadata until robot playback is acknowledged."""
    conversation = _conversation()
    pcm16 = np.arange(120, dtype="<i2").tobytes()
    event = RealtimeAudio(
        audio=RealtimeModelAudioEvent(
            data=pcm16,
            response_id="response",
            item_id="item",
            content_index=2,
        ),
        item_id="item",
        content_index=2,
        info=RealtimeEventInfo(context=MagicMock()),
    )

    await conversation._handle_event(event)
    queued = await conversation.emit()

    assert queued.item_id == "item"
    assert queued.content_index == 2
    assert queued.source_pcm16 == pcm16
    assert queued.samples.size == 240


@pytest.mark.asyncio
async def test_audio_end_marks_listening_after_queued_playback() -> None:
    """Keep speaking state until all generated audio has been consumed."""
    conversation = _conversation()
    movement_manager = conversation.dependencies.movement_manager
    audio = PlaybackAudio("item", 0, b"\x00\x00", np.zeros(1, dtype=np.float32))
    conversation.output_queue.put_nowait(audio)

    await conversation._handle_event(
        RealtimeAudioEnd(
            info=RealtimeEventInfo(context=MagicMock()),
            item_id="item",
            content_index=0,
        )
    )

    assert await conversation.emit() is audio
    movement_manager.set_speaking.assert_not_called()

    next_audio = asyncio.create_task(conversation.emit())
    await asyncio.sleep(0)
    movement_manager.set_speaking.assert_called_once_with(False)
    movement_manager.set_listening.assert_called_once_with(True)
    assert not next_audio.done()
    next_audio.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_audio


@pytest.mark.asyncio
async def test_interruption_clears_pending_audio_and_robot_player() -> None:
    """Clear queued and device audio immediately on interruption."""
    conversation = _conversation()
    clear_player = MagicMock()
    conversation.set_clear_player(clear_player)
    conversation.output_queue.put_nowait(PlaybackAudio("item", 0, b"\x00\x00", np.zeros(1, dtype=np.float32)))

    await conversation._handle_event(
        RealtimeAudioInterrupted(
            info=RealtimeEventInfo(context=MagicMock()),
            item_id="item",
            content_index=0,
        )
    )

    assert conversation.output_queue.empty()
    clear_player.assert_called_once_with()


@pytest.mark.asyncio
async def test_changed_memory_refreshes_dynamic_instructions_after_response() -> None:
    """Refresh the agent once after a response when context memory changed."""
    conversation = _conversation()
    agent = MagicMock()
    session = SimpleNamespace(update_agent=AsyncMock())
    conversation._agent = agent
    conversation._session = session
    conversation.dependencies.memory.remember("Prefers concise answers")
    event = RealtimeAgentEndEvent(
        agent=agent,
        info=RealtimeEventInfo(context=MagicMock()),
    )

    await conversation._handle_event(event)
    await conversation._handle_event(event)

    session.update_agent.assert_awaited_once_with(agent)
    assert conversation._injected_memory_revision == conversation.dependencies.memory.revision


@pytest.mark.asyncio
async def test_memory_refresh_failure_closes_stale_session() -> None:
    """Reconnect rather than retain deleted memory in stale instructions."""
    conversation = _conversation()
    agent = MagicMock()
    session = SimpleNamespace(
        update_agent=AsyncMock(side_effect=RuntimeError("refresh failed")),
        close=AsyncMock(),
    )
    conversation._agent = agent
    conversation._session = session
    conversation.dependencies.memory.remember("Prefers concise answers")

    await conversation._handle_event(
        RealtimeAgentEndEvent(
            agent=agent,
            info=RealtimeEventInfo(context=MagicMock()),
        )
    )

    session.close.assert_awaited_once_with()
    assert conversation._injected_memory_revision != conversation.dependencies.memory.revision


@pytest.mark.asyncio
async def test_camera_image_is_sent_as_one_raw_conversation_item() -> None:
    """Send camera text and image in one ordered conversation item."""
    conversation = _conversation()
    model = SimpleNamespace(send_event=MagicMock())

    async def send_event(event: RealtimeModelSendRawMessage) -> None:
        model.event = event

    model.send_event = send_event
    conversation._session = SimpleNamespace(model=model)
    jpeg = b"jpeg bytes"

    await conversation._send_image("What is this?", jpeg)

    event = model.event
    assert isinstance(event, RealtimeModelSendRawMessage)
    assert event.message["type"] == "conversation.item.create"
    item = event.message["other_data"]["item"]
    assert item["content"][0] == {"type": "input_text", "text": "What is this?"}
    assert item["content"][1]["image_url"] == (f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('ascii')}")


@pytest.mark.asyncio
async def test_session_uses_fixed_model_and_sdk_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the session with the fixed model and no transcription override."""
    captured = {}

    class FakeMCPManager:
        active_servers = []
        errors = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeSession:
        def __init__(self) -> None:
            self.messages = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def send_message(self, message) -> None:
            self.messages.append(message)

    fake_session = FakeSession()

    class CapturingRunner:
        def __init__(self, agent, *, config) -> None:
            captured["agent"] = agent
            captured["run_config"] = config

        async def run(self, *, context, model_config):
            captured["context"] = context
            captured["model_config"] = model_config
            return fake_session

    enabled_mcp_tools = []

    def create_mcp_manager(tool_names):
        enabled_mcp_tools.extend(tool_names)
        return FakeMCPManager()

    movement_manager = SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock())
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemoryState(),
        movement_manager=movement_manager,
    )
    conversation = RealtimeConversation(dependencies, voice="marin", output_sample_rate=48_000)
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(realtime_module, "RealtimeRunner", CapturingRunner)
    monkeypatch.setattr(realtime_module, "create_mcp_manager", create_mcp_manager)
    monkeypatch.setattr(realtime_module, "log_mcp_failures", MagicMock())

    await conversation.start_up()

    model_settings = captured["model_config"]["initial_model_settings"]
    audio_input = model_settings["audio"]["input"]
    assert model_settings["model_name"] == "gpt-realtime-2.1"
    assert model_settings["output_modalities"] == ["audio"]
    assert audio_input["turn_detection"]["type"] == "semantic_vad"
    assert audio_input["turn_detection"]["create_response"] is True
    assert audio_input["turn_detection"]["interrupt_response"] is True
    assert "transcription" not in audio_input
    assert captured["run_config"]["model_settings"]["reasoning"] == {"effort": "low"}
    assert captured["agent"].mcp_config == {"include_server_in_tool_names": True}
    assert len(enabled_mcp_tools) == len(set(enabled_mcp_tools))
    assert fake_session.messages
    movement_manager.set_listening.assert_called_once_with(False)
    movement_manager.set_speaking.assert_called_once_with(False)


@pytest.mark.asyncio
async def test_playback_is_reported_only_when_not_interrupted() -> None:
    """Skip playback accounting for audio cleared by an interruption."""
    conversation = _conversation()
    tracker = MagicMock()
    conversation._playback_tracker = tracker
    audio = PlaybackAudio("item", 1, b"\x00\x00", np.zeros(1, dtype=np.float32))

    await conversation.acknowledge_after_playback(audio)
    tracker.on_play_bytes.assert_called_once_with("item", 1, b"\x00\x00")

    tracker.reset_mock()
    conversation._playback_interrupted.set()
    await conversation.acknowledge_after_playback(audio)
    tracker.on_play_bytes.assert_not_called()
