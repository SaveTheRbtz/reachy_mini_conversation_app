import os
import re
import time
import base64
import asyncio
import logging
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, create_autospec

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from numpy.typing import NDArray
from websockets.asyncio.client import ClientConnection
from openai.resources.live.live import AsyncLiveConnection
from openai.types.live.output_audio_delta_event import OutputAudioDeltaEvent

from tests.support.console import make_robot, make_conversation
from tests.support.realtime import running_task
from tests.support.realtime import make_conversation as make_live_conversation
import reachy_mini_conversation_app.console as console_module
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.config import OPENAI_API_KEY_ENV
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import PlaybackAudio
from reachy_mini_conversation_app.startup_settings import write_startup_settings
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_pb import Conversation


@pytest.mark.asyncio
async def test_spa_routes_keep_static_files_and_rpc_separate(tmp_path: Path) -> None:
    """Browser routes load the packaged SPA without intercepting assets or API requests."""
    application = FastAPI()

    @application.get("/{path:path}")
    def sdk_page() -> dict[str, str]:
        return {"page": "SDK fallback"}

    stream = LocalStream(
        make_robot(),
        conversation_factory=lambda voice: make_conversation(),
        settings_app=application,
        instance_path=tmp_path,
    )
    stream.init_settings_ui()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
        root = await client.get("/")
        assert root.status_code == 200
        assert root.headers["cache-control"] == "no-cache"
        assert '<div id="app"></div>' in root.text
        for path in ("/settings", "/profiles/builtin-default"):
            response = await client.get(path)
            assert response.status_code == 200
            assert response.content == root.content

        assets = re.findall(r'(?:src|href)="(/static/[^\"]+)"', root.text)
        assert assets
        for asset in assets:
            response = await client.get(asset)
            assert response.status_code == 200
            assert "text/html" not in response.headers["content-type"]
        assert (await client.get("/static/missing.js")).status_code == 404

        response = await client.post(
            "/rpc/reachy.conversation.v1.ConversationService/GetConversation", json={"name": "conversation"}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "conversation"
        assert (await client.post("/rpc/missing", json={})).status_code == 404


def test_inactivity_survives_reconnect_and_ignores_playback_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only dialogue refreshes the sleep deadline, including after a session replacement."""
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    conversation = make_conversation()
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation)
    observer_call = conversation.set_activity_observer.call_args
    assert observer_call is not None
    activity = observer_call.args[0]
    now = 200.0
    activity("interaction")
    now = 300.0
    for reason in ("playback_started", "playback_stopped", "disconnected"):
        activity(reason)
    replacement = make_conversation()
    stream._install_conversation(replacement)
    observer_call = replacement.set_activity_observer.call_args
    assert observer_call is not None
    activity = observer_call.args[0]
    activity("connected")
    assert stream.seconds_since_activity() == 100.0
    now = 400.0
    activity("interaction")
    assert stream.seconds_since_activity() == 0.0


@pytest.mark.asyncio
async def test_status_reports_microphone_and_playback_independently() -> None:
    """A muted microphone does not imply that the robot stopped playing audio."""
    conversation = make_conversation()
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation)
    observer_call = conversation.set_activity_observer.call_args
    assert observer_call is not None
    activity = observer_call.args[0]
    await stream.set_muted(True)
    activity("playback_started")
    activity("interaction")
    status = stream.snapshot()
    assert status.muted is True
    assert status.playing is True
    activity("disconnected")
    assert stream.snapshot().playing is False


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_override, expected", [("coral", "coral"), (None, "shimmer")])
async def test_restart_resolves_voice_without_waiting_for_remote_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_override: str | None, expected: str
) -> None:
    """Accept a restart locally using the same voice precedence as app startup."""
    conversation = make_conversation()
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation, instance_path=tmp_path)
    stream._asyncio_loop = asyncio.get_running_loop()
    write_startup_settings(tmp_path, profile=None, voice=voice_override)
    monkeypatch.setattr(console_module, "get_profile_instructions", lambda: "Profile instructions")
    monkeypatch.setattr(console_module, "get_function_tools", lambda names: [])
    monkeypatch.setattr(console_module, "get_session_voice", lambda default: "shimmer")
    await stream.restart("")
    assert stream.snapshot().voice == expected
    assert stream.snapshot().connection_state == Conversation.ConnectionState.CONNECTING
    conversation.shutdown.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["remote_close", "error", "requested_restart", "stopped"])
async def test_session_logs_end_reason_and_reconnect_delay(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distinguish unexpected session endings from requested restarts and stops."""
    conversation = make_conversation()
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation)
    monkeypatch.setattr(console_module, "has_openai_api_key", lambda: True)
    retry = AsyncMock()
    monkeypatch.setattr(stream, "_wait_for_restart", retry)

    async def start_up() -> None:
        if conversation.start_up.await_count > 1 or reason == "stopped":
            stream._stop_event.set()
        elif reason == "requested_restart":
            await stream.request_restart("test")
        elif reason == "error":
            raise ConnectionError("connection lost")

    conversation.start_up = AsyncMock(side_effect=start_up)
    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        await stream._run_session_loop()

    assert any("Live connection attempt: attempt=1" in message for message in caplog.messages)
    assert any(f"reason={reason} elapsed=" in message for message in caplog.messages)
    if reason in {"remote_close", "error"}:
        retry.assert_awaited_once_with(console_module.RETRY_DELAY_SECONDS)
        assert any("Live reconnect scheduled" in message for message in caplog.messages)
    else:
        retry.assert_not_awaited()
        assert not any("Live reconnect scheduled" in message for message in caplog.messages)
    if reason == "error":
        assert any(
            "Live session failed: attempt=1 elapsed=" in message
            and "error=ConnectionError: connection lost" in message
            for message in caplog.messages
        )


@pytest.mark.asyncio
async def test_microphone_mute_logs_only_changes(caplog: pytest.LogCaptureFixture) -> None:
    """Local microphone control remains available before the network loop starts."""
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: make_conversation())
    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        for muted in (False, True, True, False):
            await stream.set_muted(muted)
            assert stream.snapshot().muted is muted

    assert [message for message in caplog.messages if "Microphone mute changed" in message] == [
        "Microphone mute changed: muted=True",
        "Microphone mute changed: muted=False",
    ]


@pytest.mark.asyncio
async def test_restart_without_api_key_waits_instead_of_spinning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart with no credential must leave the event loop responsive."""
    conversation = make_conversation()
    conversation.connected = False
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation)
    monkeypatch.setattr(console_module, "has_openai_api_key", lambda: False)
    await stream.request_restart("test")

    async def wait_for_configuration(timeout: float) -> None:
        assert not stream._restart_requested.is_set()
        stream._stop_event.set()

    monkeypatch.setattr(stream, "_wait_for_restart", wait_for_configuration)
    await stream._run_session_loop()
    assert stream.snapshot().connection_state == Conversation.ConnectionState.WAITING_FOR_CONFIG


@pytest.mark.asyncio
async def test_api_key_persistence_failure_keeps_runtime_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed credential save must not silently switch the running configuration."""
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: make_conversation(), instance_path=tmp_path)
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "existing-key")
    (tmp_path / ".env").mkdir()
    with pytest.raises(OSError):
        await stream.set_api_key("replacement-key")
    assert os.environ[OPENAI_API_KEY_ENV] == "existing-key"
    with pytest.raises(ValueError, match="single line"):
        await stream.set_api_key("replacement-key\nUNRELATED_SETTING=1")


@pytest.mark.asyncio
async def test_record_loop_warns_once_when_microphone_frames_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report a sustained capture failure without flooding logs."""
    conversation = make_conversation()
    robot = make_robot()
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    robot.media.get_audio_sample.return_value = None
    monkeypatch.setattr(console_module, "MICROPHONE_FRAME_TIMEOUT_SECONDS", 0.0)
    recovered = asyncio.Event()
    conversation.interrupt.side_effect = recovered.set

    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        async with running_task(stream.record_loop()) as capture:
            await asyncio.wait_for(recovered.wait(), timeout=2)
            robot.media.stop_recording.assert_called_once_with()
            robot.media.start_recording.assert_called_once_with()
            robot.media.start_playing.assert_called_once_with()
            conversation.clear_playback.assert_called_once_with()
            conversation.interrupt.assert_awaited_once_with()
            stream.close()
            await asyncio.wait_for(capture, timeout=2)

    assert sum("No usable microphone frames received" in message for message in caplog.messages) == 1
    conversation.receive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [False, True])
async def test_capture_recovery_restarts_media_before_bounded_live_interruption(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stalled: bool,
) -> None:
    """Stop remote speech after local recovery without blocking on a stalled write."""
    conversation = make_live_conversation(output_rate=48_000)
    conversation.output_queue.put_nowait(PlaybackAudio(np.ones(48_000, dtype=np.float32)))
    robot = make_robot()
    recovered_audio = np.ones((2, 1_600), dtype=np.float32)
    robot.media.get_audio_sample.side_effect = lambda: recovered_audio if robot.media.start_recording.called else None
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    monkeypatch.setattr(console_module, "MICROPHONE_FRAME_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(realtime_module, "SEND_TIMEOUT_SECONDS", 0.01)

    async def interrupt_live(**_kwargs: object) -> None:
        robot.media.start_recording.assert_called_once_with()
        robot.media.start_playing.assert_called_once_with()
        assert conversation.output_queue.empty()
        if stalled:
            await asyncio.Event().wait()

    instructions = AsyncMock(side_effect=interrupt_live)
    connection = AsyncLiveConnection(create_autospec(ClientConnection, instance=True))
    monkeypatch.setattr(connection.session.instructions, "append", instructions)
    conversation._connection = connection

    async with running_task(stream.record_loop()):
        microphone_pcm = await asyncio.wait_for(conversation._microphone_queue.get(), timeout=1.0)

    robot.media.audio.clear_player.assert_called()
    robot.media.stop_recording.assert_called_once_with()
    robot.media.start_recording.assert_called_once_with()
    robot.media.start_playing.assert_called_once_with()
    assert conversation.output_queue.empty()
    assert np.max(np.frombuffer(microphone_pcm, dtype="<i2")) > 30_000
    instructions.assert_awaited_once()
    assert instructions.await_args is not None
    assert "Stop speaking now and listen" in instructions.await_args.kwargs["content"]
    assert ("Failed to interrupt Live after microphone recovery" in caplog.text) is stalled


@pytest.mark.asyncio
async def test_record_loop_logs_and_forwards_first_microphone_frame(caplog: pytest.LogCaptureFixture) -> None:
    """Identify the capture and forwarding boundary from the first usable frame."""
    conversation = make_conversation()
    robot = make_robot()
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    robot.media.get_audio_sample.return_value = np.ones((2, 160), dtype=np.float32)
    forwarded = asyncio.Event()
    conversation.receive.side_effect = lambda frame: forwarded.set()

    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        async with running_task(stream.record_loop()) as capture:
            await asyncio.wait_for(forwarded.wait(), timeout=2)
            stream.close()
            await asyncio.wait_for(capture, timeout=2)

    assert any("Microphone capture started" in message for message in caplog.messages)
    assert conversation.receive.await_count > 0


@pytest.mark.asyncio
async def test_muted_microphone_forwards_silence_without_changing_capture() -> None:
    """Keep Live's audio clock running while withholding microphone contents."""
    conversation = make_conversation()
    robot = make_robot()
    captured = np.ones((2, 160), dtype=np.float32)
    robot.media.get_audio_sample.return_value = captured
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    await stream.set_muted(True)
    conversation.receive.side_effect = lambda _frame: stream._stop_event.set()

    await stream.record_loop()

    conversation.receive.assert_awaited_once()
    sample_rate, forwarded = conversation.receive.await_args.args[0]
    assert sample_rate == 16_000
    np.testing.assert_array_equal(forwarded, np.zeros_like(captured))
    np.testing.assert_array_equal(captured, np.ones_like(captured))


@pytest.mark.asyncio
async def test_play_loop_pushes_chunks_without_waiting_for_playback_tracking() -> None:
    """Keep the player fed and finish playback tracking only after queued audio drains."""
    conversation = make_conversation()
    chunks = [PlaybackAudio(np.ones(19_200, dtype=np.float32)) for _ in range(3)]
    for chunk in chunks:
        conversation.output_queue.put_nowait(chunk)
    conversation.emit.side_effect = conversation.output_queue.get
    tracking_release = asyncio.Event()
    playback_ended = asyncio.Event()
    pushed_third = asyncio.Event()
    tracked: list[PlaybackAudio] = []

    async def acknowledge(audio: PlaybackAudio) -> None:
        tracked.append(audio)
        if audio is chunks[0]:
            await tracking_release.wait()

    conversation.acknowledge_after_playback.side_effect = acknowledge
    conversation.acknowledge_playback_end.side_effect = playback_ended.set
    robot = make_robot()
    robot.media.push_audio_sample.side_effect = lambda samples: (
        pushed_third.set() if samples is chunks[2].samples else None
    )
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    async with running_task(stream.play_loop()), running_task(stream._acknowledge_playback_loop()):
        await asyncio.wait_for(pushed_third.wait(), timeout=1.0)
        assert len(tracked) == 1
        assert tracked[0] is chunks[0]
        assert not playback_ended.is_set()
        tracking_release.set()
        await asyncio.wait_for(playback_ended.wait(), timeout=1.0)

    assert len(tracked) == 3
    assert all(actual is expected for actual, expected in zip(tracked, chunks, strict=True))
    conversation.acknowledge_playback_end.assert_called_once_with()


@pytest.mark.asyncio
async def test_interruption_discards_old_tracking_before_new_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear queued playback and let fresh Live audio reach the player promptly."""
    conversation = make_live_conversation(output_rate=48_000)
    instructions = AsyncMock()
    connection = AsyncLiveConnection(create_autospec(ClientConnection, instance=True))
    monkeypatch.setattr(connection.session.instructions, "append", instructions)
    conversation._connection = connection
    for _ in range(2):
        conversation.output_queue.put_nowait(PlaybackAudio(np.ones(48_000, dtype=np.float32)))
    old_audio_pushed = asyncio.Event()
    fresh_audio_pushed = asyncio.Event()
    robot = make_robot()

    def push_audio(_samples: NDArray[np.float32]) -> None:
        if robot.media.push_audio_sample.call_count == 2:
            old_audio_pushed.set()
        elif robot.media.push_audio_sample.call_count == 3:
            fresh_audio_pushed.set()

    robot.media.push_audio_sample.side_effect = push_audio
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    async with running_task(stream.play_loop()), running_task(stream._acknowledge_playback_loop()):
        await asyncio.wait_for(old_audio_pushed.wait(), timeout=1.0)
        await conversation.interrupt()
        assert stream._playback_acknowledgements.empty()
        robot.media.audio.clear_player.assert_called_once_with()
        instructions.assert_awaited_once()
        await conversation._handle_event(
            OutputAudioDeltaEvent(
                type="session.output_audio.delta",
                delta=base64.b64encode(np.ones(4_800, dtype="<i2").tobytes()).decode("ascii"),
            )
        )
        await asyncio.wait_for(fresh_audio_pushed.wait(), timeout=1.0)


def test_close_finalizes_while_session_receiver_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the session receiver running until shutdown receives finalization."""
    conversation = make_conversation()
    robot = make_robot()
    stream = LocalStream(robot, conversation_factory=lambda voice: conversation)
    receiver_started = asyncio.Event()
    finalized = asyncio.Event()
    receiver_exited = False

    async def start_up() -> None:
        nonlocal receiver_exited
        receiver_started.set()
        try:
            await finalized.wait()
        finally:
            receiver_exited = True

    async def record() -> None:
        await receiver_started.wait()
        stream.close()

    async def shutdown() -> None:
        assert not receiver_exited
        finalized.set()
        await asyncio.sleep(0)

    async def pending() -> None:
        await asyncio.Event().wait()

    conversation.start_up = AsyncMock(side_effect=start_up)
    conversation.shutdown.side_effect = shutdown
    monkeypatch.setattr(stream, "record_loop", record)
    monkeypatch.setattr(stream, "play_loop", pending)
    monkeypatch.setattr(stream, "_acknowledge_playback_loop", pending)
    monkeypatch.setattr(console_module, "has_openai_api_key", lambda: True)
    monkeypatch.setattr(asyncio, "to_thread", AsyncMock())

    stream.launch()

    conversation.shutdown.assert_awaited_once_with()
    assert receiver_exited
    robot.media.stop_recording.assert_called_once_with()
    robot.media.stop_playing.assert_called_once_with()


@pytest.mark.asyncio
async def test_restart_cancels_stalled_startup_after_finalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local controls responsive while the session owner finalizes a stalled connection."""
    connecting = make_conversation()
    replacement = make_conversation()
    connecting.history = [("user", "Old dialogue")]
    started = asyncio.Event()
    shutdown_started = asyncio.Event()
    finish_shutdown = asyncio.Event()
    startup_cancelled = asyncio.Event()
    conversations = iter((connecting, replacement))
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: next(conversations))
    monkeypatch.setattr(console_module, "has_openai_api_key", lambda: True)

    async def connect() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            startup_cancelled.set()

    async def shutdown() -> None:
        shutdown_started.set()
        await finish_shutdown.wait()

    async def start_replacement() -> None:
        stream._stop_event.set()

    connecting.start_up = AsyncMock(side_effect=connect)
    connecting.shutdown.side_effect = shutdown
    replacement.start_up = AsyncMock(side_effect=start_replacement)
    async with running_task(stream._run_session_loop()) as session:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await stream.request_restart("configuration_changed")
        await asyncio.wait_for(shutdown_started.wait(), timeout=1.0)
        await stream.set_muted(True)
        assert stream.snapshot().muted is True
        assert stream.snapshot().connection_state == Conversation.ConnectionState.CONNECTING
        assert not startup_cancelled.is_set()
        finish_shutdown.set()
        await asyncio.wait_for(session, timeout=1.0)
    assert startup_cancelled.is_set()
    replacement.start_up.assert_awaited_once_with()
    assert replacement.history == []


@pytest.mark.asyncio
async def test_cancelled_rpc_cancels_work_on_the_audio_loop() -> None:
    """A caller deadline must cancel its remote send across the ASGI and audio event loops."""
    conversation = make_conversation()
    stream = LocalStream(make_robot(), conversation_factory=lambda voice: conversation)
    caller_loop = asyncio.get_running_loop()
    owner_loop = asyncio.new_event_loop()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def say(text: str) -> None:
        assert asyncio.get_running_loop() is owner_loop
        caller_loop.call_soon_threadsafe(started.set)
        try:
            await asyncio.Event().wait()
        finally:
            caller_loop.call_soon_threadsafe(cancelled.set)

    conversation.say = AsyncMock(side_effect=say)
    stream._asyncio_loop = owner_loop
    owner = Thread(target=owner_loop.run_forever, daemon=True)
    owner.start()
    request = asyncio.create_task(stream.say("hello"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        await asyncio.to_thread(owner.join, 2)
        assert not owner.is_alive(), "The audio event loop did not stop"
        owner_loop.close()
