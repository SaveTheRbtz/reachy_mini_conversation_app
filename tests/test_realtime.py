import base64
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
from agents.realtime import (
    RealtimeAudio,
    RealtimeAudioEnd,
    RealtimeEventInfo,
    RealtimeAgentEndEvent,
    RealtimeRawModelEvent,
    RealtimeModelAudioEvent,
    RealtimeAudioInterrupted,
    RealtimeModelSendRawMessage,
)
from agents.realtime.model_events import RealtimeModelRawServerEvent

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

    assert queued is not None
    assert queued.item_id == "item"
    assert queued.content_index == 2
    assert queued.source_pcm16 == pcm16
    assert queued.samples.size == 240


@pytest.mark.asyncio
async def test_microphone_forwarding_logs_success_and_throttles_delay_warnings(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose successful but slow microphone forwarding without flooding logs."""
    conversation = _conversation()
    activity_observer = MagicMock()
    conversation.set_activity_observer(activity_observer)
    last_activity_time = conversation.last_activity_time
    session = SimpleNamespace(send_audio=AsyncMock())
    conversation._session = session
    monkeypatch.setattr(realtime_module, "REALTIME_AUDIO_SEND_STALL_SECONDS", 0.0)
    frame = (24_000, np.ones(240, dtype=np.float32))

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        await conversation.receive(frame)
        await conversation.receive(frame)

    assert sum("Realtime microphone forwarding started" in message for message in caplog.messages) == 1
    assert sum("Realtime microphone forwarding delayed" in message for message in caplog.messages) == 1
    assert session.send_audio.await_count == 2
    assert conversation.last_activity_time == last_activity_time
    activity_observer.assert_not_called()


@pytest.mark.asyncio
async def test_vad_speech_transitions_mark_user_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset inactivity only when server VAD detects user speech."""
    conversation = _conversation()
    activity_observer = MagicMock()
    conversation.set_activity_observer(activity_observer)
    monkeypatch.setattr(
        realtime_module,
        "time",
        SimpleNamespace(monotonic=MagicMock(side_effect=[100.0, 200.0])),
    )

    for event_type in (
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
    ):
        await conversation._handle_event(
            RealtimeRawModelEvent(
                data=RealtimeModelRawServerEvent(data={"type": event_type}),
                info=RealtimeEventInfo(context=MagicMock()),
            )
        )

    assert conversation.last_activity_time == 200.0
    assert activity_observer.call_args_list == [call("listening"), call("thinking")]


@pytest.mark.asyncio
async def test_audio_end_defers_listening_until_playback_tracking() -> None:
    """Keep the listening transition ordered behind generated audio."""
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
    assert await conversation.emit() is None
    movement_manager.set_speaking.assert_not_called()
    movement_manager.set_listening.assert_not_called()

    conversation.acknowledge_playback_end()

    movement_manager.set_speaking.assert_called_once_with(False)
    movement_manager.set_listening.assert_called_once_with(True)


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
async def test_session_uses_fixed_model_sdk_defaults_and_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the session with fixed model defaults and server-side truncation."""
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
            self.model = SimpleNamespace(send_event=AsyncMock())

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
    context_config = fake_session.model.send_event.await_args_list[0].args[0]
    assert isinstance(context_config, RealtimeModelSendRawMessage)
    truncation = context_config.message["other_data"]["session"]["truncation"]
    assert truncation == {
        "type": "retention_ratio",
        "retention_ratio": 0.8,
        "token_limits": {"post_instructions": 64_000},
    }
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
