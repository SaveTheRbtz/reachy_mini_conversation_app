import json
import base64
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Callable

import soxr
import numpy as np
import pytest
from pydantic import TypeAdapter
from openai.types.live.server_event import ServerEvent

import reachy_mini_conversation_app.console as console_module
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import PlaybackAudio, LiveConversation, StreamingAudioBridge


def _conversation(output_rate: int = 24_000) -> LiveConversation:
    dependencies = SimpleNamespace(
        instance_path=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=SimpleNamespace(set_speaking=MagicMock()),
    )
    return LiveConversation(dependencies, voice="marin", output_sample_rate=output_rate)


def test_audio_bridge_converts_stereo_float_to_24khz_pcm16() -> None:
    """Convert robot stereo input to the Live PCM format."""
    bridge = StreamingAudioBridge(output_sample_rate=48_000)
    stereo = np.stack(
        (
            np.linspace(-0.5, 0.5, 1600, dtype=np.float32),
            np.linspace(0.5, -0.5, 1600, dtype=np.float32),
        )
    )

    pcm16 = bridge.microphone_to_pcm16(16_000, stereo)

    assert 0 < len(pcm16) < 2400 * 2
    assert not np.frombuffer(pcm16, dtype="<i2").any()
    assert np.frombuffer(pcm16, dtype="<i2").dtype == np.dtype("int16")


def test_audio_bridge_converts_openai_pcm_to_robot_rate() -> None:
    """Convert Live PCM output to the configured robot rate."""
    bridge = StreamingAudioBridge(output_sample_rate=48_000)
    pcm16 = np.arange(2400, dtype="<i2").tobytes()

    playback = bridge.pcm16_to_playback(pcm16)

    assert playback.dtype == np.float32
    assert 0 < playback.size < 4800


@pytest.mark.parametrize("output_rate", [16_000, 48_000])
@pytest.mark.parametrize("chunk_samples", [137, 2400, 9600])
def test_playback_resampling_matches_continuous_audio(output_rate: int, chunk_samples: int) -> None:
    """Preserve waveform continuity and duration regardless of incoming chunk boundaries."""
    bridge = StreamingAudioBridge(output_sample_rate=output_rate)
    pcm16 = (np.sin(np.arange(24_000) * 2 * np.pi * 440 / 24_000) * 16_000).astype("<i2")
    pcm16 = np.concatenate((pcm16, np.zeros(4096, dtype="<i2")))
    chunks = [
        bridge.pcm16_to_playback(pcm16[offset : offset + chunk_samples].tobytes())
        for offset in range(0, pcm16.size, chunk_samples)
    ]
    playback = np.concatenate(chunks)
    expected = soxr.resample(pcm16.astype(np.float32) / 32768.0, 24_000, output_rate)

    assert output_rate < playback.size <= expected.size
    np.testing.assert_allclose(playback, expected[: playback.size], atol=1e-6)


def test_microphone_resampling_stays_continuous_through_silence() -> None:
    """Preserve input phase across live frames while silence carries the buffered speech tail."""
    bridge = StreamingAudioBridge(output_sample_rate=16_000)
    signal = np.sin(np.arange(16_000) * 2 * np.pi * 440 / 16_000).astype(np.float32) / 2
    microphone = np.concatenate((signal, np.zeros(4096, dtype=np.float32)))
    stereo = np.column_stack((microphone, microphone))
    pcm16 = b"".join(
        bridge.microphone_to_pcm16(16_000, stereo[offset : offset + 1024])
        for offset in range(0, stereo.shape[0], 1024)
    )
    actual = np.frombuffer(pcm16, dtype="<i2")
    expected = np.asarray(np.clip(soxr.resample(microphone, 16_000, 24_000), -1.0, 1.0) * 32767, dtype="<i2")

    assert 24_000 < actual.size <= expected.size
    np.testing.assert_allclose(actual, expected[: actual.size], atol=1)


class LiveTransport:
    """Feed typed SDK events while recording the production client's writes."""

    def __init__(self) -> None:
        """Initialize queued events and observable client commands."""
        self.events: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self.session = SimpleNamespace(
            start=AsyncMock(),
            close=AsyncMock(),
            update=AsyncMock(),
            input_audio=SimpleNamespace(append=AsyncMock()),
            instructions=SimpleNamespace(append=AsyncMock()),
        )
        self.response = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()), create=AsyncMock())
        self.close = AsyncMock()

    async def __aenter__(self) -> "LiveTransport":
        """Open the test transport."""
        return self

    async def __aexit__(self, *args: object) -> bool:
        """Release the test transport."""
        return False

    def __aiter__(self) -> "LiveTransport":
        """Read events in their queued order."""
        return self

    async def recv(self) -> ServerEvent:
        """Receive one typed server event."""
        return await self.events.get()

    async def __anext__(self) -> ServerEvent:
        """Wait for the next server event."""
        return await self.events.get()


def _event(event_type: str, **fields: object) -> ServerEvent:
    return TypeAdapter(ServerEvent).validate_python({"type": event_type, "event_id": "event_test", **fields})


def _backend_event(event_type: str, **fields: object) -> ServerEvent:
    return _event("response.event", delegation_id="delegation_test", event={"type": event_type, **fields})


def _response(status: str) -> dict[str, object]:
    return {
        "id": "response_test",
        "created_at": 1,
        "model": "gpt-5.6-luna",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }


async def _eventually(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@pytest.fixture
def live_transport(monkeypatch: pytest.MonkeyPatch) -> LiveTransport:
    """Replace only the official SDK connection for session lifecycle tests."""
    transport = LiveTransport()
    client = MagicMock()
    client.live.connect.return_value = transport
    client.__aenter__.return_value = client
    monkeypatch.setattr(realtime_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    return transport


SESSION = {"id": "session_test", "expires_at": 1000, "model": "gpt-live-1", "status": "active"}


@pytest.mark.asyncio
async def test_session_gates_microphone_until_started_and_waits_for_final_usage(live_transport: LiveTransport) -> None:
    """Start microphone transport only after readiness and receive graceful finalization."""
    conversation = _conversation()
    task = asyncio.create_task(conversation.start_up())
    try:
        await _eventually(lambda: live_transport.session.start.await_count == 1)
        microphone = (24_000, np.zeros(2400, dtype=np.float32))
        await conversation.receive(microphone)
        assert not conversation.connected
        live_transport.session.input_audio.append.assert_not_awaited()
        await live_transport.events.put(_event("session.started", session=SESSION))
        await _eventually(lambda: conversation.connected)
        await conversation.receive(microphone)
        await _eventually(lambda: live_transport.session.input_audio.append.await_count == 1)
        live_transport.session.input_audio.append.assert_awaited_once()
        config = live_transport.session.start.await_args.kwargs["session"]
        assert config["model"] == "gpt-live-1"
        assert config["audio"]["format"] == {"type": "audio/pcm", "rate": 24_000}
        assert config["delegation"]["type"] == "responses"
        assert config["delegation"]["responses"]["model"] == "gpt-6-astra"
        assert config["delegation"]["responses"]["reasoning"] == {"effort": "low"}
        assert "turn_detection" not in config["audio"]
        closing = asyncio.create_task(conversation.shutdown())
        await _eventually(lambda: live_transport.session.close.await_count == 1)
        assert not closing.done()
        await conversation.receive(microphone)
        live_transport.session.input_audio.append.assert_awaited_once()
        await live_transport.events.put(
            _event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.2})
        )
        await asyncio.wait_for(closing, timeout=2)
        await asyncio.wait_for(task, timeout=2)
        assert not conversation.connected
        assert conversation.usage_seconds == 1.2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_full_duplex_input_does_not_discard_speaking_audio(live_transport: LiveTransport) -> None:
    """Keep microphone and playback flowing when input transcripts overlap speech."""
    conversation = _conversation()
    activity = MagicMock()
    conversation.set_activity_observer(activity)
    task = asyncio.create_task(conversation.start_up())
    try:
        await live_transport.events.put(_event("session.started", session=SESSION))
        await _eventually(lambda: conversation.connected)
        pcm = np.arange(2400, dtype="<i2")
        await live_transport.events.put(_event("session.output_audio.delta", delta=base64.b64encode(pcm).decode()))
        await live_transport.events.put(_event("session.input_transcript.delta", delta="yes", start_ms=10, end_ms=20))
        await conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
        await _eventually(lambda: live_transport.session.input_audio.append.await_count == 1)
        playback = await asyncio.wait_for(conversation.emit(), timeout=1)
        np.testing.assert_allclose(playback.samples, pcm.astype(np.float32) / 32768)
        assert conversation.output_queue.empty()
        tracking = asyncio.create_task(conversation.acknowledge_after_playback(playback))
        await _eventually(lambda: any(call.args == ("playback_started",) for call in activity.call_args_list))
        activity.reset_mock()
        await conversation._handle_event(
            _event("session.input_transcript.delta", delta=" go on", start_ms=20, end_ms=30)
        )
        activity.assert_called_once_with("interaction")
        await tracking
        conversation.acknowledge_playback_end()
        assert activity.call_args.args == ("playback_stopped",)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failure", ["disconnect", "stall"])
def test_microphone_network_failure_reconnects_without_stopping_media(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace a failed session and keep the robot's capture and playback loops alive."""
    failed_transport = LiveTransport()
    recovered_transport = LiveTransport()
    for transport in (failed_transport, recovered_transport):
        transport.events.put_nowait(_event("session.started", session=SESSION))
    stalled_send: asyncio.Future[None] | None = None

    async def fail_send(**kwargs: object) -> None:
        nonlocal stalled_send
        if failure == "stall":
            stalled_send = asyncio.get_running_loop().create_future()
            await stalled_send
        raise ConnectionResetError("simulated network reset")

    failed_transport.session.input_audio.append.side_effect = fail_send
    client = MagicMock()
    client.__aenter__.return_value = client
    client.live.connect.side_effect = [failed_transport, recovered_transport]
    monkeypatch.setattr(realtime_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(realtime_module, "SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(console_module.asyncio, "to_thread", AsyncMock())
    robot = MagicMock()
    robot.media.get_input_audio_samplerate.return_value = 24_000
    robot.media.get_audio_sample.return_value = np.zeros(2400, dtype=np.float32)
    stream = LocalStream(robot, conversation_factory=lambda voice: _conversation(), startup_voice="marin")
    recovered_transport.session.input_audio.append.side_effect = lambda **kwargs: stream.close()
    recovered_transport.session.close.side_effect = lambda: recovered_transport.events.put_nowait(
        _event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.2})
    )

    stream.launch()

    assert client.live.connect.call_count == 2
    failed_transport.close.assert_awaited()
    assert recovered_transport.session.input_audio.append.await_count >= 1
    robot.media.start_recording.assert_called_once_with()
    robot.media.start_playing.assert_called_once_with()
    assert stream.conversation.usage_seconds == 1.2
    if stalled_send is not None:
        assert stalled_send.cancelled()


@pytest.mark.asyncio
async def test_microphone_backpressure_keeps_only_the_latest_pending_frame(
    live_transport: LiveTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep capture responsive without accumulating stale microphone audio."""
    conversation = _conversation()
    release_send = asyncio.Event()

    async def send_audio(**kwargs: object) -> None:
        await release_send.wait()

    live_transport.session.input_audio.append.side_effect = send_audio
    session = asyncio.create_task(conversation.start_up())
    try:
        await live_transport.events.put(_event("session.started", session=SESSION))
        await _eventually(lambda: conversation.connected)
        await conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
        await _eventually(lambda: live_transport.session.input_audio.append.await_count == 1)
        for amplitude in (0.25, 0.5, 0.75):
            await conversation.receive((24_000, np.full(2400, amplitude, dtype=np.float32)))
        assert live_transport.session.input_audio.append.await_count == 1
        release_send.set()
        await _eventually(lambda: live_transport.session.input_audio.append.await_count == 2)
        latest_audio = live_transport.session.input_audio.append.await_args.kwargs["audio"]
        np.testing.assert_allclose(np.frombuffer(base64.b64decode(latest_audio), dtype="<i2"), 0.75 * 32767, atol=1)
        assert sum("discarding queued audio" in message for message in caplog.messages) == 1
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_interrupt_clears_playback_and_redirects_live() -> None:
    """Flush received and device audio only for an explicit application interruption."""
    conversation = _conversation()
    conversation._connection = LiveTransport()
    clear_player = MagicMock()
    conversation.set_clear_player(clear_player)
    conversation.output_queue.put_nowait(PlaybackAudio(samples=np.ones(2400, dtype=np.float32)))
    await conversation.interrupt()
    assert conversation.output_queue.empty()
    clear_player.assert_called_once_with()
    conversation._connection.session.instructions.append.assert_awaited_once()
    assert "stop" in conversation._connection.session.instructions.append.await_args.kwargs["content"].lower()


@pytest.mark.asyncio
async def test_typed_text_is_backend_input() -> None:
    """Route typed user text to Responses instead of voice instructions."""
    conversation = _conversation()
    conversation._connection = LiveTransport()
    await conversation.say("My order number is A0042.")
    items = [entry.kwargs["item"] for entry in conversation._connection.response.item.create.await_args_list]
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "My order number is A0042."}]
    instructions = conversation._connection.session.instructions.append.await_args.kwargs
    assert instructions["delegation_id"] is None
    assert "Answer that request aloud" in instructions["content"]
    assert "A0042" not in instructions["content"]


@pytest.mark.asyncio
async def test_function_batch_survives_empty_terminal_output_and_duplicate_events() -> None:
    """Execute each collected call once and continue only after all function results."""
    conversation = _conversation()
    transport = LiveTransport()
    conversation._connection = transport

    def assert_complete_batch(**kwargs: object) -> None:
        assert transport.response.item.create.await_count == 2

    transport.response.create.side_effect = assert_complete_batch
    execute = AsyncMock(return_value={"status": "following"})
    conversation._tools = {"head_tracking": SimpleNamespace(name="head_tracking", on_invoke_tool=execute)}
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(
            _backend_event("response.created", sequence_number=0, response=_response("in_progress"))
        )
        for call_id in ("call_1", "call_2", "call_1"):
            await conversation._handle_event(
                _backend_event(
                    "response.output_item.done",
                    sequence_number=1,
                    output_index=0,
                    item={
                        "id": f"item_{call_id}",
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "head_tracking",
                        "arguments": '{"enabled":true}',
                        "status": "completed",
                    },
                )
            )
        transport.response.create.assert_not_awaited()
        completed = _backend_event("response.completed", sequence_number=2, response=_response("completed"))
        await conversation._handle_event(completed)
        await _eventually(lambda: transport.response.create.await_count == 1)
        await conversation._handle_event(completed)
        await asyncio.sleep(0)
        assert execute.await_count == 2
        assert transport.response.item.create.await_count == 2
        assert transport.response.create.await_count == 1
        results = [entry.kwargs["item"] for entry in transport.response.item.create.await_args_list]
        assert {item["call_id"] for item in results} == {"call_1", "call_2"}
        assert all(item["type"] == "function_call_output" for item in results)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_live_error_logs_metadata_without_private_content(caplog: pytest.LogCaptureFixture) -> None:
    """Keep session errors diagnosable without logging server-echoed private content."""
    conversation = _conversation()
    with caplog.at_level(logging.WARNING):
        await conversation._handle_event(
            _event(
                "error", error={"type": "invalid_request_error", "code": "invalid_value", "message": "private text"}
            )
        )
    assert "invalid_value" in caplog.text
    assert "private text" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_available", [True, False])
async def test_failed_tool_returns_error_and_continues_backend(tool_available: bool) -> None:
    """Return failures for unavailable or failing tools without ending the voice stream."""
    conversation = _conversation()
    transport = LiveTransport()
    conversation._connection = transport
    if tool_available:
        conversation._tools = {
            "head_tracking": SimpleNamespace(
                name="head_tracking", on_invoke_tool=AsyncMock(side_effect=ValueError("invalid arguments"))
            )
        }
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(
            _backend_event("response.created", sequence_number=0, response=_response("in_progress"))
        )
        await conversation._handle_event(
            _backend_event(
                "response.output_item.done",
                sequence_number=1,
                output_index=0,
                item={
                    "id": "item_failed",
                    "type": "function_call",
                    "call_id": "call_failed",
                    "name": "head_tracking",
                    "arguments": "{}",
                    "status": "completed",
                },
            )
        )
        await conversation._handle_event(
            _backend_event("response.completed", sequence_number=2, response=_response("completed"))
        )
        await _eventually(lambda: transport.response.create.await_count == 1)
        result = transport.response.item.create.await_args.kwargs["item"]
        assert result["call_id"] == "call_failed"
        assert "error" in json.loads(result["output"])
        assert conversation.connected
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


def test_stereo_pcm16_microphone_preserves_amplitude() -> None:
    """Normalize integer stereo input before averaging channels."""
    bridge = StreamingAudioBridge(output_sample_rate=24_000)
    stereo = np.full((2400, 2), 8192, dtype=np.int16)
    converted = np.frombuffer(bridge.microphone_to_pcm16(24_000, stereo), dtype="<i2")
    np.testing.assert_allclose(converted, 8192, atol=1)


@pytest.mark.asyncio
async def test_typed_input_waits_for_dequeued_tool_result_before_continuation() -> None:
    """Queue typed corrections while a tool executes without starting a competing response."""
    conversation = _conversation()
    transport = LiveTransport()
    conversation._connection = transport
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(context: object, arguments: str) -> dict[str, str]:
        tool_started.set()
        await release_tool.wait()
        return {"status": "following"}

    conversation._tools = {"head_tracking": SimpleNamespace(name="head_tracking", on_invoke_tool=execute)}
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(
            _backend_event("response.created", sequence_number=0, response=_response("in_progress"))
        )
        await conversation.say("Please keep the movement gentle.")
        transport.response.create.assert_not_awaited()
        await conversation._handle_event(
            _backend_event(
                "response.output_item.done",
                sequence_number=1,
                output_index=0,
                item={
                    "id": "item_slow",
                    "type": "function_call",
                    "call_id": "call_slow",
                    "name": "head_tracking",
                    "arguments": '{"enabled":true}',
                    "status": "completed",
                },
            )
        )
        await conversation._handle_event(
            _backend_event("response.completed", sequence_number=2, response=_response("completed"))
        )
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        assert conversation._tool_batches.empty()
        await conversation.say("Actually, stop tracking after this.")
        transport.response.create.assert_not_awaited()
        release_tool.set()
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        transport.response.create.assert_awaited_once()
        submitted = [entry.kwargs["item"] for entry in transport.response.item.create.await_args_list]
        assert [item["type"] for item in submitted] == ["message", "message", "function_call_output"]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_closes_transport_while_waiting_for_startup(live_transport: LiveTransport) -> None:
    """Close a connecting session without sending commands before it is ready."""
    conversation = _conversation()
    task = asyncio.create_task(conversation.start_up())
    try:
        await _eventually(lambda: live_transport.session.start.await_count == 1)
        await conversation.shutdown()
        live_transport.close.assert_awaited_once()
        live_transport.session.close.assert_not_awaited()
        live_transport.session.instructions.append.assert_not_awaited()
        assert not conversation.connected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rejected_backend_command_stops_session_without_leaking_input(caplog: pytest.LogCaptureFixture) -> None:
    """Abort a rejected backend command so a busy session cannot silently stall."""
    conversation = _conversation()
    conversation._connection = LiveTransport()
    await conversation.say("private typed request")
    command_id = conversation._connection.response.create.await_args.kwargs["event_id"]
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="Live backend command rejected"):
        await conversation._handle_event(
            _event(
                "error",
                error={
                    "type": "invalid_request_error",
                    "code": "response_input_buffer_full",
                    "message": "private typed request",
                    "client_event_id": command_id,
                },
            )
        )
    assert not conversation.connected
    assert "response_input_buffer_full" in caplog.text
    assert "private typed request" not in caplog.text


@pytest.mark.asyncio
async def test_malformed_backend_event_fails_without_logging_private_content(caplog: pytest.LogCaptureFixture) -> None:
    """Reject invalid protocol input without echoing nested private content."""
    conversation = _conversation()
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="Invalid Live backend event"):
        await conversation._handle_event(
            _backend_event("response.created", sequence_number=0, response={"instructions": "private instructions"})
        )
    assert "Invalid Live backend event" in caplog.text
    assert "private instructions" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("search_enabled", [False, True])
async def test_session_registers_hosted_search_only_when_enabled(
    live_transport: LiveTransport, monkeypatch: pytest.MonkeyPatch, search_enabled: bool
) -> None:
    """Keep hosted search selection consistent with local tools and frontend capabilities."""
    selected = ["head_tracking", "web_search"] if search_enabled else ["head_tracking"]
    monkeypatch.setattr(realtime_module, "selected_tool_names", lambda _: selected)
    conversation = _conversation()
    task = asyncio.create_task(conversation.start_up())
    try:
        await _eventually(lambda: live_transport.session.start.await_count == 1)
        session = live_transport.session.start.await_args.kwargs["session"]
        tools = session["delegation"]["responses"]["tools"]
        assert [tool.get("name") for tool in tools if tool["type"] == "function"] == ["head_tracking"]
        assert ({"type": "web_search"} in tools) is search_enabled
        assert ("- web_search:" in session["instructions"]) is search_enabled
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
