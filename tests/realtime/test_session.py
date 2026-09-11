import base64
import asyncio
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.support.realtime import (
    SESSION,
    LiveTransport,
    event,
    eventually,
    running_task,
    backend_event,
    make_conversation,
)
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.config import LIVE_MODEL, DELEGATION_MODEL
from reachy_mini_conversation_app.realtime import OPENAI_SAMPLE_RATE, LiveConversation


pytestmark = pytest.mark.asyncio


async def test_microphone_waits_for_readiness_and_shutdown_waits_for_final_usage(
    live_transport: LiveTransport,
) -> None:
    """Start microphone transport only after readiness and receive graceful finalization."""
    conversation = make_conversation()
    async with running_task(conversation.start_up()) as session:
        await eventually(lambda: live_transport.session.start.await_count == 1)
        microphone = (24_000, np.zeros(2400, dtype=np.float32))
        await conversation.receive(microphone)
        assert not conversation.connected
        live_transport.session.input_audio.append.assert_not_awaited()
        await live_transport.send_event(event("session.started", session=SESSION))
        await conversation.receive(microphone)
        await eventually(lambda: live_transport.session.input_audio.append.await_count == 1)

        async with running_task(conversation.shutdown()) as closing:
            await eventually(lambda: live_transport.session.close.await_count == 1)
            assert not closing.done()
            await conversation.receive(microphone)
            live_transport.session.input_audio.append.assert_awaited_once()
            live_transport.events.put_nowait(
                event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.2})
            )
            await asyncio.wait_for(closing, timeout=2)
        await asyncio.wait_for(session, timeout=2)

    assert not conversation.connected
    assert conversation.usage_seconds == 1.2


async def test_full_duplex_input_does_not_discard_speaking_audio(
    conversation: LiveConversation, live_transport: LiveTransport
) -> None:
    """Keep microphone and playback flowing when input transcripts overlap speech."""
    activity = MagicMock()
    conversation.set_activity_observer(activity)
    pcm = np.arange(2400, dtype="<i2")
    await live_transport.send_event(event("session.output_audio.delta", delta=base64.b64encode(pcm).decode()))
    await live_transport.send_event(event("session.input_transcript.delta", delta="yes", start_ms=10, end_ms=20))
    await conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
    await eventually(lambda: live_transport.session.input_audio.append.await_count == 1)
    playback = await asyncio.wait_for(conversation.emit(), timeout=2)
    np.testing.assert_allclose(playback.samples, pcm.astype(np.float32) / 32768)
    assert conversation.output_queue.empty()

    async with running_task(conversation.acknowledge_after_playback(playback)) as tracking:
        await eventually(lambda: any(call.args == ("playback_started",) for call in activity.call_args_list))
        activity.reset_mock()
        await live_transport.send_event(
            event("session.input_transcript.delta", delta=" go on", start_ms=20, end_ms=30)
        )
        activity.assert_called_once_with("interaction")
        await tracking
    conversation.acknowledge_playback_end()
    assert activity.call_args.args == ("playback_stopped",)


async def test_microphone_backpressure_keeps_only_the_latest_pending_frame(
    conversation: LiveConversation, live_transport: LiveTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep capture responsive without accumulating stale microphone audio."""
    release_send = asyncio.Event()

    async def send_audio(**kwargs: object) -> None:
        await release_send.wait()

    live_transport.session.input_audio.append.side_effect = send_audio
    await conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
    await eventually(lambda: live_transport.session.input_audio.append.await_count == 1)
    for amplitude in (0.25, 0.5, 0.75):
        await conversation.receive((24_000, np.full(2400, amplitude, dtype=np.float32)))
    assert live_transport.session.input_audio.append.await_count == 1
    release_send.set()
    await eventually(lambda: live_transport.session.input_audio.append.await_count == 2)

    latest_audio = live_transport.session.input_audio.append.await_args.kwargs["audio"]
    np.testing.assert_allclose(np.frombuffer(base64.b64decode(latest_audio), dtype="<i2"), 0.75 * 32767, atol=1)
    assert sum("discarding queued audio" in message for message in caplog.messages) == 1


async def test_explicit_interrupt_clears_playback_and_redirects_live(
    conversation: LiveConversation, live_transport: LiveTransport
) -> None:
    """Flush received and device audio only for an explicit application interruption."""
    clear_player = MagicMock()
    conversation.set_clear_player(clear_player)
    pcm = np.ones(2400, dtype="<i2").tobytes()
    await live_transport.send_event(event("session.output_audio.delta", delta=base64.b64encode(pcm).decode()))
    assert not conversation.output_queue.empty()

    await conversation.interrupt()

    assert conversation.output_queue.empty()
    clear_player.assert_called_once_with()
    live_transport.session.instructions.append.assert_awaited_once()
    assert live_transport.session.instructions.append.await_args.kwargs["delegation_id"] is None


async def test_shutdown_closes_transport_while_waiting_for_startup(live_transport: LiveTransport) -> None:
    """Close a connecting session without sending commands before it is ready."""
    conversation = make_conversation()
    async with running_task(conversation.start_up()):
        await eventually(lambda: live_transport.session.start.await_count == 1)
        await conversation.shutdown()
        live_transport.close.assert_awaited_once()
        live_transport.session.close.assert_not_awaited()
        live_transport.session.instructions.append.assert_not_awaited()
        assert not conversation.connected


async def test_uncorrelated_live_error_logs_metadata_without_private_content(
    conversation: LiveConversation, live_transport: LiveTransport, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep recoverable session errors diagnosable without logging server-echoed private content."""
    with caplog.at_level(logging.WARNING):
        await live_transport.send_event(
            event("error", error={"type": "invalid_request_error", "code": "invalid_value", "message": "private text"})
        )
    assert conversation.connected
    assert "invalid_value" in caplog.text
    assert "private text" not in caplog.text


@pytest.mark.parametrize("failure", ["rejected-command", "malformed-response"])
async def test_fatal_protocol_error_ends_session_without_logging_private_content(
    live_transport: LiveTransport, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """Let supervision recover from unusable protocol state without exposing private payloads."""
    conversation = make_conversation()
    with caplog.at_level(logging.ERROR), pytest.raises(ExceptionGroup) as raised:
        async with running_task(conversation.start_up()) as session:
            await live_transport.send_event(event("session.started", session=SESSION))
            await conversation.say("private typed request")
            if failure == "rejected-command":
                command_id = live_transport.response.create.await_args.kwargs["event_id"]
                failed_event = event(
                    "error",
                    error={
                        "type": "invalid_request_error",
                        "code": "response_input_buffer_full",
                        "message": "private typed request",
                        "client_event_id": command_id,
                    },
                )
                expected_error = "Live backend command rejected"
            else:
                failed_event = backend_event(
                    "response.created", sequence_number=0, response={"instructions": "private instructions"}
                )
                expected_error = "Invalid Live backend event"
            live_transport.events.put_nowait(failed_event)
            await asyncio.wait_for(session, timeout=2)

    assert any(expected_error in str(error) for error in raised.value.exceptions)
    assert not conversation.connected
    assert "private typed request" not in caplog.text
    assert "private instructions" not in caplog.text
    live_transport.close.assert_awaited_once()


@pytest.mark.parametrize("search_enabled", [False, True], ids=["local-tools", "hosted-search"])
async def test_startup_wires_current_models_audio_and_selected_tools(
    live_transport: LiveTransport, monkeypatch: pytest.MonkeyPatch, search_enabled: bool
) -> None:
    """Configure the protocol from application settings and expose only selected capabilities."""
    selected = ["head_tracking", "web_search"] if search_enabled else ["head_tracking"]
    monkeypatch.setattr(realtime_module, "selected_tool_names", lambda _: selected)
    conversation = make_conversation()
    async with running_task(conversation.start_up()):
        await eventually(lambda: live_transport.session.start.await_count == 1)
        session = live_transport.session.start.await_args.kwargs["session"]
        assert session["model"] == LIVE_MODEL
        assert session["audio"]["format"] == {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE}
        assert session["audio"]["output"]["voice"] == conversation.voice
        assert session["delegation"]["type"] == "responses"
        assert session["delegation"]["responses"]["model"] == DELEGATION_MODEL
        assert session["delegation"]["responses"]["parallel_tool_calls"] is False
        tools = session["delegation"]["responses"]["tools"]
        assert [tool.get("name") for tool in tools if tool["type"] == "function"] == ["head_tracking"]
        assert ({"type": "web_search"} in tools) is search_enabled
        assert ("- web_search:" in session["instructions"]) is search_enabled
