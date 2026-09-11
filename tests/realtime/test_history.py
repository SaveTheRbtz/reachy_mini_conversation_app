import asyncio
from threading import Event, Timer
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.support.realtime import SESSION, LiveTransport, event, eventually, running_task, make_conversation
import reachy_mini_conversation_app.console as console_module
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.prompts import get_session_greeting_prompt
from reachy_mini_conversation_app.realtime import LiveConversation


@pytest.fixture
def transports(monkeypatch: pytest.MonkeyPatch) -> list[LiveTransport]:
    """Supply replacement transports to the production session supervisor."""
    transports = [LiveTransport() for _ in range(3)]
    for transport in transports:
        transport.events.put_nowait(event("session.started", session=SESSION))
    client = MagicMock()
    client.__aenter__.return_value = client
    client.live.connect.side_effect = transports
    monkeypatch.setattr(realtime_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 0.0)
    return transports


@pytest.mark.asyncio
async def test_reconnect_preserves_dialogue_through_a_failed_startup(transports: list[LiveTransport]) -> None:
    """Retain transcript fragments and typed requests through consecutive network failures."""
    first, failed_startup, recovered = transports
    first.session.input_audio.append.side_effect = ConnectionResetError("simulated connection failure")
    failed_startup.session.start.side_effect = ConnectionResetError("simulated startup failure")
    stream = LocalStream(
        MagicMock(), conversation_factory=lambda voice: make_conversation(voice=voice), startup_voice="marin"
    )
    expected_history = [
        ("user", "How about autumn?"),
        ("assistant", "The leaves turn red."),
        ("user", "Show me"),
        ("user", "Use the second option."),
    ]
    async with running_task(stream._run_session_loop()):
        await eventually(lambda: stream.conversation.connected)
        for role, fragment in [
            ("input", "How about"),
            ("input", " autumn?"),
            ("output", "The leaves"),
            ("output", " turn red."),
            ("input", "Show me"),
        ]:
            first.events.put_nowait(event(f"session.{role}_transcript.delta", delta=fragment, start_ms=0, end_ms=100))
        await eventually(lambda: stream.conversation.history == expected_history[:-1])
        await stream.conversation.say(expected_history[-1][1])
        await stream.conversation.receive((24000, np.zeros(2400, dtype=np.float32)))
        await eventually(lambda: recovered.session.start.await_count == 1 and stream.conversation.connected)
        assert stream.conversation.history == expected_history
        first.session.instructions.append.assert_any_await(
            event_id="greeting", delegation_id=None, content=get_session_greeting_prompt()
        )
        for transport in (failed_startup, recovered):
            seeded = transport.session.start.await_args.kwargs["session"]["input"]
            assert [(message["role"], message["content"][0]["text"]) for message in seeded] == expected_history
            assert [message["content"][0]["type"] for message in seeded] == [
                "input_text",
                "output_text",
                "input_text",
                "input_text",
            ]
            assert all((message.get("type", "message") == "message" for message in seeded))
            assert all(
                (
                    call.kwargs.get("event_id") != "greeting"
                    for call in transport.session.instructions.append.await_args_list
                )
            )
            transport.response.create.assert_not_awaited()
            transport.response.item.create.assert_not_awaited()
        recovered.events.put_nowait(
            event("session.input_transcript.delta", delta="Yes, the second one.", start_ms=0, end_ms=100)
        )
        await eventually(lambda: stream.conversation.history == [*expected_history, ("user", "Yes, the second one.")])


@pytest.mark.asyncio
async def test_dialogue_history_bounds_messages_and_utf8_bytes(
    conversation: LiveConversation, live_transport: LiveTransport
) -> None:
    """Keep the newest context without unbounded growth or broken Unicode fragments."""
    for index in range(35):
        role = "input" if index % 2 == 0 else "output"
        await live_transport.send_event(
            event(f"session.{role}_transcript.delta", delta=f"Message {index}", start_ms=0, end_ms=100)
        )
    assert len(conversation.history) == 32
    assert conversation.history[0][1] == "Message 3"
    assert conversation.history[-1][1] == "Message 34"
    for fragment in ("🌱" * 1025, "最新"):
        await live_transport.send_event(
            event("session.input_transcript.delta", delta=fragment, start_ms=0, end_ms=100)
        )
    retained_text = "".join((message for _, message in conversation.history))
    assert len(retained_text.encode("utf-8")) <= 4096
    assert retained_text.endswith("🌱最新")
    assert "�" not in retained_text


@pytest.mark.parametrize("during_retry", [False, True])
@pytest.mark.asyncio
async def test_explicit_restart_discards_prior_dialogue(
    transports: list[LiveTransport], monkeypatch: pytest.MonkeyPatch, during_retry: bool
) -> None:
    """Start fresh after an explicit reset even when a reconnect is already pending."""
    first, restarted, _ = transports
    stream = LocalStream(
        MagicMock(), conversation_factory=lambda voice: make_conversation(voice=voice), startup_voice="marin"
    )
    first.session.close.side_effect = lambda: first.events.put_nowait(
        event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.0})
    )
    if during_retry:
        monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 600.0)
        first.session.input_audio.append.side_effect = ConnectionResetError("simulated connection failure")
    async with running_task(stream._run_session_loop()):
        await eventually(lambda: stream.conversation.connected)
        first.events.put_nowait(
            event("session.input_transcript.delta", delta="Remember my old topic.", start_ms=0, end_ms=100)
        )
        await eventually(lambda: bool(stream.conversation.history))
        if during_retry:
            await stream.conversation.receive((24000, np.zeros(2400, dtype=np.float32)))
            await eventually(lambda: first.close.await_count == 1)
            restarted.session.start.assert_not_awaited()
        await stream.request_restart("personality_changed")
        await eventually(lambda: restarted.session.instructions.append.await_count == 1)
        assert not stream.conversation.history
        assert not restarted.session.start.await_args.kwargs["session"].get("input")
        assert restarted.session.instructions.append.await_args.kwargs["event_id"] == "greeting"


@pytest.mark.parametrize("failure", ["disconnect", "stall"])
def test_microphone_network_failure_reconnects_without_stopping_media(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace a failed session and keep the robot's capture and playback loops alive."""
    failed_transport = LiveTransport()
    recovered_transport = LiveTransport()
    for transport in (failed_transport, recovered_transport):
        transport.events.put_nowait(event("session.started", session=SESSION))
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
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(realtime_module, "SEND_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(realtime_module, "SESSION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(asyncio, "to_thread", AsyncMock())
    robot = MagicMock()
    robot.media.get_input_audio_samplerate.return_value = 24000
    robot.media.get_audio_sample.return_value = np.zeros(2400, dtype=np.float32)
    stream = LocalStream(
        robot, conversation_factory=lambda voice: make_conversation(voice=voice), startup_voice="marin"
    )
    recovered_transport.session.input_audio.append.side_effect = lambda **kwargs: stream.close()
    recovered_transport.session.close.side_effect = lambda: recovered_transport.events.put_nowait(
        event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.2})
    )
    timed_out = Event()

    def stop_unresponsive_stream() -> None:
        timed_out.set()
        stream.close()

    deadline = Timer(5, stop_unresponsive_stream)
    deadline.start()
    try:
        stream.launch()
    finally:
        deadline.cancel()
        deadline.join()
    assert not timed_out.is_set(), "Session recovery did not finish before the test deadline"
    assert client.live.connect.call_count == 2
    failed_transport.close.assert_awaited()
    assert recovered_transport.session.input_audio.append.await_count >= 1
    robot.media.start_recording.assert_called_once_with()
    robot.media.start_playing.assert_called_once_with()
    assert stream.conversation.usage_seconds == 1.2
    if stalled_send is not None:
        assert stalled_send.cancelled()
