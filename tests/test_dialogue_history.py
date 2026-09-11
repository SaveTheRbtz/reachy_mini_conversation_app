import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_realtime import SESSION, LiveTransport, _event, _eventually, _conversation

import reachy_mini_conversation_app.console as console_module
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.console import LocalStream


pytestmark = pytest.mark.asyncio


@pytest.fixture
def transports(monkeypatch: pytest.MonkeyPatch) -> list[LiveTransport]:
    """Supply replacement transports to the production session supervisor."""
    transports = [LiveTransport() for _ in range(3)]
    for transport in transports:
        transport.events.put_nowait(_event("session.started", session=SESSION))
    client = MagicMock()
    client.__aenter__.return_value = client
    client.live.connect.side_effect = transports
    monkeypatch.setattr(realtime_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 0.0)
    return transports


async def test_reconnect_preserves_dialogue_through_a_failed_startup(transports: list[LiveTransport]) -> None:
    """Retain transcript fragments and typed requests through consecutive network failures."""
    first, failed_startup, recovered = transports
    first.session.input_audio.append.side_effect = ConnectionResetError("simulated connection failure")
    failed_startup.session.start.side_effect = ConnectionResetError("simulated startup failure")
    stream = LocalStream(MagicMock(), conversation_factory=lambda voice: _conversation(), startup_voice="marin")
    session_loop = asyncio.create_task(stream._run_session_loop())
    expected_history = [
        ("user", "How about autumn?"),
        ("assistant", "The leaves turn red."),
        ("user", "Show me"),
        ("user", "Use the second option."),
    ]
    try:
        await _eventually(lambda: stream.conversation.connected)
        for role, fragment in [
            ("input", "How about"),
            ("input", " autumn?"),
            ("output", "The leaves"),
            ("output", " turn red."),
            ("input", "Show me"),
        ]:
            first.events.put_nowait(_event(f"session.{role}_transcript.delta", delta=fragment, start_ms=0, end_ms=100))
        await _eventually(lambda: stream.conversation.history == expected_history[:-1])
        await stream.conversation.say(expected_history[-1][1])
        await stream.conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
        await _eventually(lambda: recovered.session.start.await_count == 1 and stream.conversation.connected)

        assert stream.conversation.history == expected_history
        first.session.instructions.append.assert_any_await(
            event_id="greeting", delegation_id=None, content=realtime_module.get_session_greeting_prompt()
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
            assert all(message.get("type", "message") == "message" for message in seeded)
            assert all(
                call.kwargs.get("event_id") != "greeting"
                for call in transport.session.instructions.append.await_args_list
            )
            transport.response.create.assert_not_awaited()
            transport.response.item.create.assert_not_awaited()
        recovered.events.put_nowait(
            _event("session.input_transcript.delta", delta="Yes, the second one.", start_ms=0, end_ms=100)
        )
        await _eventually(lambda: stream.conversation.history == [*expected_history, ("user", "Yes, the second one.")])
    finally:
        session_loop.cancel()
        await asyncio.gather(session_loop, return_exceptions=True)


async def test_dialogue_history_bounds_messages_and_utf8_bytes() -> None:
    """Keep the newest context without unbounded growth or broken Unicode fragments."""
    conversation = _conversation()
    for index in range(35):
        role = "input" if index % 2 == 0 else "output"
        await conversation._handle_event(
            _event(f"session.{role}_transcript.delta", delta=f"Message {index}", start_ms=0, end_ms=100)
        )

    assert len(conversation.history) == 32
    assert conversation.history[0][1] == "Message 3"
    assert conversation.history[-1][1] == "Message 34"
    for fragment in ("🌱" * 1025, "最新"):
        await conversation._handle_event(
            _event("session.input_transcript.delta", delta=fragment, start_ms=0, end_ms=100)
        )

    retained_text = "".join(message for _, message in conversation.history)
    assert len(retained_text.encode("utf-8")) <= 4096
    assert retained_text.endswith("🌱最新")
    assert "\ufffd" not in retained_text


@pytest.mark.parametrize("during_retry", [False, True])
async def test_explicit_restart_discards_prior_dialogue(
    transports: list[LiveTransport], monkeypatch: pytest.MonkeyPatch, during_retry: bool
) -> None:
    """Start fresh after an explicit reset even when a reconnect is already pending."""
    first, restarted, _ = transports
    stream = LocalStream(MagicMock(), conversation_factory=lambda voice: _conversation(), startup_voice="marin")
    first.session.close.side_effect = lambda: first.events.put_nowait(
        _event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.0})
    )
    if during_retry:
        monkeypatch.setattr(console_module, "RETRY_DELAY_SECONDS", 600.0)
        first.session.input_audio.append.side_effect = ConnectionResetError("simulated connection failure")
    session_loop = asyncio.create_task(stream._run_session_loop())
    try:
        await _eventually(lambda: stream.conversation.connected)
        first.events.put_nowait(
            _event("session.input_transcript.delta", delta="Remember my old topic.", start_ms=0, end_ms=100)
        )
        await _eventually(lambda: bool(stream.conversation.history))
        if during_retry:
            await stream.conversation.receive((24_000, np.zeros(2400, dtype=np.float32)))
            await _eventually(lambda: first.close.await_count == 1)
            restarted.session.start.assert_not_awaited()
        await stream.request_restart("personality_changed")
        await _eventually(lambda: restarted.session.instructions.append.await_count == 1)

        assert not stream.conversation.history
        assert not restarted.session.start.await_args.kwargs["session"].get("input")
        assert restarted.session.instructions.append.await_args.kwargs["event_id"] == "greeting"
    finally:
        session_loop.cancel()
        await asyncio.gather(session_loop, return_exceptions=True)
