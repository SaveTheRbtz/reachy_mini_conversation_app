import base64
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest
from pydantic import ValidationError
from agents.realtime import (
    RealtimeAudio,
    RealtimeError,
    RealtimeToolEnd,
    RealtimeAudioEnd,
    RealtimeEventInfo,
    RealtimeToolStart,
    RealtimeRawModelEvent,
    RealtimeModelAudioEvent,
    RealtimeAudioInterrupted,
    RealtimeModelSendRawMessage,
)
from openai.types.realtime import RealtimeError as OpenAIRealtimeError
from agents.realtime.model_events import RealtimeModelRawServerEvent, RealtimeModelTurnStartedEvent

import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.realtime import PlaybackAudio, RealtimeConversation, StreamingAudioBridge


def _conversation(output_rate: int = 48_000) -> RealtimeConversation:
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemorySnapshot(memories=[]),
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
async def test_audio_event_preserves_playback_and_logs_once_per_item(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve playback metadata and correlate first audio without per-chunk logs."""
    conversation = _conversation()
    now = 100.0
    monkeypatch.setattr(realtime_module, "time", SimpleNamespace(monotonic=lambda: now))
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

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        await conversation._handle_event(
            RealtimeRawModelEvent(
                data=RealtimeModelTurnStartedEvent(response_id="response"),
                info=RealtimeEventInfo(context=MagicMock()),
            )
        )
        for item_id in ("item", "next-item"):
            event.item_id = item_id
            event.audio.item_id = item_id
            now += 0.25
            await conversation._handle_event(event)
            await conversation._handle_event(event)
    queued = await conversation.emit()

    assert queued is not None
    assert queued.item_id == "item"
    assert queued.content_index == 2
    assert queued.source_pcm16 == pcm16
    assert queued.samples.size == 240
    audio_logs = [message for message in caplog.messages if "Realtime assistant audio received" in message]
    assert len(audio_logs) == 2
    assert "item_id=item" in audio_logs[0]
    assert "item_id=next-item" in audio_logs[1]
    assert all("response_id=response" in message for message in audio_logs)
    assert "elapsed_since_response_ms=250" in audio_logs[0]
    assert "elapsed_since_response_ms=500" in audio_logs[1]
    assert repr(pcm16) not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason", "error_code", "level"),
    [
        ("completed", None, None, logging.INFO),
        ("cancelled", "turn_detected", None, logging.INFO),
        ("failed", None, "server_error", logging.WARNING),
        ("incomplete", "max_output_tokens", None, logging.WARNING),
    ],
)
async def test_response_logs_outcome_timing_and_usage_without_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: str,
    reason: str | None,
    error_code: str | None,
    level: int,
) -> None:
    """Distinguish API failures from successful turns and normal barge-in."""
    conversation = _conversation()
    now = 100.0
    monkeypatch.setattr(realtime_module, "time", SimpleNamespace(monotonic=lambda: now))
    response = {
        "id": "resp_test",
        "status": status,
        "status_details": {
            "type": status,
            "reason": reason,
            "error": {"code": error_code, "type": "server_error"} if error_code else None,
        },
        "usage": {"input_tokens": 1234, "output_tokens": 99, "input_token_details": {"cached_tokens": 256}},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "PRIVATE RESPONSE CONTENT"}],
            }
        ],
    }

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        await conversation._handle_event(
            RealtimeRawModelEvent(
                data=RealtimeModelTurnStartedEvent(response_id="resp_test"),
                info=RealtimeEventInfo(context=MagicMock()),
            )
        )
        now = 102.0
        await conversation._handle_event(
            RealtimeRawModelEvent(
                data=RealtimeModelRawServerEvent(
                    data={"type": "response.done", "event_id": "event_test", "response": response}
                ),
                info=RealtimeEventInfo(context=MagicMock()),
            )
        )

    assert any(
        "Realtime response started" in message and "response_id=resp_test" in message for message in caplog.messages
    )
    finished = [record for record in caplog.records if "Realtime response finished" in record.getMessage()]
    assert len(finished) == 1
    assert finished[0].levelno == level
    message = finished[0].getMessage()
    for field in (
        "response_id=resp_test",
        f"status={status}",
        f"reason={reason}",
        f"error_code={error_code}",
        "duration_ms=2000",
        "input_tokens=1234",
        "cached_tokens=256",
        "output_tokens=99",
    ):
        assert field in message
    assert "PRIVATE RESPONSE CONTENT" not in caplog.text


@pytest.mark.asyncio
async def test_malformed_response_logging_does_not_leak_content_or_stop_playback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep diagnostics from exposing validation inputs or breaking audio handling."""
    conversation = _conversation()
    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        await conversation._handle_event(
            RealtimeRawModelEvent(
                data=RealtimeModelRawServerEvent(
                    data={"type": "response.done", "event_id": "event_test", "response": "PRIVATE INVALID RESPONSE"}
                ),
                info=RealtimeEventInfo(context=MagicMock()),
            )
        )
        await conversation._handle_event(
            RealtimeAudioEnd(info=RealtimeEventInfo(context=MagicMock()), item_id="item", content_index=0)
        )

    assert sum(record.levelno == logging.WARNING for record in caplog.records) == 1
    assert "PRIVATE INVALID RESPONSE" not in caplog.text
    assert not any("Realtime response finished" in message for message in caplog.messages)
    assert await conversation.emit() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "fields"),
    [
        (
            OpenAIRealtimeError(
                type="invalid_request_error",
                code="invalid_value",
                param="audio",
                event_id="event_client",
                message="PRIVATE API ERROR MESSAGE",
            ),
            ("type=invalid_request_error", "code=invalid_value", "param=audio", "event_id=event_client"),
        ),
        (
            ValidationError.from_exception_data(
                "RealtimeResponse",
                [{"type": "string_type", "loc": ("response",), "input": {"content": "PRIVATE VALIDATION INPUT"}}],
            ),
            ("string_type", "response"),
        ),
        (
            {"message": "Tool call task failed: ConnectionError", "output": "PRIVATE SDK OUTPUT"},
            ("Tool call task failed",),
        ),
        (ConnectionError("Connection closed by peer"), ("type=ConnectionError", "Connection closed by peer")),
    ],
)
async def test_realtime_errors_log_support_metadata_without_private_content(
    caplog: pytest.LogCaptureFixture,
    error: object,
    fields: tuple[str, ...],
) -> None:
    """Expose support identifiers without dumping API messages or validation inputs."""
    conversation = _conversation()
    await conversation._handle_event(RealtimeError(error=error, info=RealtimeEventInfo(context=MagicMock())))

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert all(field in caplog.text for field in fields)
    assert "PRIVATE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "outcome", "level"),
    [
        ({"error": "No frame available"}, "error", logging.WARNING),
        ("PRIVATE TOOL RESULT", "returned", logging.INFO),
    ],
)
async def test_tool_logs_timing_and_outcome_without_arguments_or_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    output: dict[str, str] | str,
    outcome: str,
    level: int,
) -> None:
    """Measure tool calls without treating returned strings as proven successes."""
    conversation = _conversation()
    now = 100.0
    monkeypatch.setattr(realtime_module, "time", SimpleNamespace(monotonic=lambda: now))
    tool = MagicMock()
    tool.name = "test_tool"
    started = RealtimeToolStart(
        agent=MagicMock(),
        tool=tool,
        arguments='{"query":"PRIVATE TOOL ARGUMENTS"}',
        info=RealtimeEventInfo(context=MagicMock()),
    )

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        await conversation._handle_event(started)
        now = 102.0
        await conversation._handle_event(
            RealtimeToolEnd(
                agent=started.agent,
                tool=tool,
                arguments=started.arguments,
                output=output,
                info=started.info,
            )
        )

    assert any("Tool started: test_tool" in message for message in caplog.messages)
    finished = [record for record in caplog.records if "Tool finished: test_tool" in record.getMessage()]
    assert len(finished) == 1
    assert finished[0].levelno == level
    assert f"outcome={outcome}" in finished[0].getMessage()
    assert "duration_ms=2000" in finished[0].getMessage()
    if isinstance(output, dict):
        assert "error=No frame available" in finished[0].getMessage()
    assert "PRIVATE" not in caplog.text


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
async def test_failed_microphone_send_does_not_log_forwarding_success(caplog: pytest.LogCaptureFixture) -> None:
    """Report microphone forwarding only after sending a frame succeeds."""
    conversation = _conversation()
    send_audio = AsyncMock(side_effect=ConnectionError("Connection closed"))
    conversation._session = SimpleNamespace(send_audio=send_audio)
    frame = (24_000, np.ones(240, dtype=np.float32))

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        with pytest.raises(ConnectionError):
            await conversation.receive(frame)
        assert not any("Realtime microphone forwarding started" in message for message in caplog.messages)
        send_audio.side_effect = None
        await conversation.receive(frame)

    assert sum("Realtime microphone forwarding started" in message for message in caplog.messages) == 1


@pytest.mark.asyncio
async def test_vad_speech_transitions_mark_user_activity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log server-detected speech and reset inactivity without logging microphone frames."""
    conversation = _conversation()
    activity_observer = MagicMock()
    conversation.set_activity_observer(activity_observer)
    now = 100.0
    monkeypatch.setattr(realtime_module, "time", SimpleNamespace(monotonic=lambda: now))

    with caplog.at_level(logging.INFO, logger=realtime_module.__name__):
        for event_type, now in (
            ("input_audio_buffer.speech_started", 100.0),
            ("input_audio_buffer.speech_stopped", 200.0),
        ):
            await conversation._handle_event(
                RealtimeRawModelEvent(
                    data=RealtimeModelRawServerEvent(data={"type": event_type, "item_id": "user_item"}),
                    info=RealtimeEventInfo(context=MagicMock()),
                )
            )

    assert conversation.last_activity_time == 200.0
    assert activity_observer.call_args_list == [call("listening"), call("thinking")]
    assert sum("item_id=user_item" in message for message in caplog.messages) == 2


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

    movement_manager = SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock())
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=movement_manager,
    )
    conversation = RealtimeConversation(dependencies, voice="marin", output_sample_rate=48_000)
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(realtime_module, "RealtimeRunner", CapturingRunner)

    await conversation.start_up()

    model_settings = captured["model_config"]["initial_model_settings"]
    audio_input = model_settings["audio"]["input"]
    assert model_settings["model_name"] == "gpt-realtime-2.1"
    assert model_settings["output_modalities"] == ["audio"]
    assert audio_input["turn_detection"]["type"] == "semantic_vad"
    assert audio_input["turn_detection"]["create_response"] is True
    assert audio_input["turn_detection"]["interrupt_response"] is True
    assert "transcription" not in audio_input
    assert model_settings["parallel_tool_calls"] is False
    assert captured["run_config"]["async_tool_calls"] is False
    assert captured["run_config"]["model_settings"]["reasoning"] == {"effort": "low"}
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
