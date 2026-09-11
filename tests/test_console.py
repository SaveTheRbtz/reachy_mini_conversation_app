import base64
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from openai.types.live.output_audio_delta_event import OutputAudioDeltaEvent

import reachy_mini_conversation_app.console as console_module
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import PlaybackAudio, LiveConversation


def _conversation() -> SimpleNamespace:
    return SimpleNamespace(
        voice="gleam",
        history=[],
        last_activity_time=0.0,
        connected=True,
        shutdown=AsyncMock(),
        receive=AsyncMock(),
        emit=AsyncMock(),
        interrupt=AsyncMock(),
        clear_playback=MagicMock(),
        acknowledge_after_playback=AsyncMock(),
        acknowledge_playback_end=MagicMock(),
        set_clear_player=MagicMock(),
        set_activity_observer=MagicMock(),
        output_queue=asyncio.Queue(),
    )


def _robot() -> SimpleNamespace:
    return SimpleNamespace(
        media=SimpleNamespace(
            get_input_audio_samplerate=MagicMock(return_value=16_000),
            get_audio_sample=MagicMock(),
            push_audio_sample=MagicMock(),
            audio=SimpleNamespace(clear_player=MagicMock()),
            stop_recording=MagicMock(),
            start_recording=MagicMock(),
            start_playing=MagicMock(),
            stop_playing=MagicMock(),
        )
    )


@pytest.mark.asyncio
async def test_voice_change_persists_and_requests_one_session_restart(tmp_path) -> None:
    """Apply an immutable session voice through the stream's restart owner."""
    conversation = SimpleNamespace(
        voice="gleam",
        last_activity_time=0.0,
        connected=True,
        shutdown=AsyncMock(),
        set_clear_player=MagicMock(),
        set_activity_observer=MagicMock(),
    )
    robot = SimpleNamespace()
    stream = LocalStream(
        robot,
        conversation_factory=MagicMock(return_value=conversation),
        instance_path=tmp_path,
    )

    status = await stream.change_voice("coral")

    assert status == "Voice changed to coral; restarting the conversation."
    conversation.shutdown.assert_awaited_once_with()
    assert '"voice": "coral"' in (tmp_path / "startup_settings.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["remote_close", "error", "requested_restart", "stopped"])
async def test_session_logs_end_reason_and_reconnect_delay(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distinguish unexpected session endings from requested restarts and stops."""
    conversation = _conversation()
    stream = LocalStream(_robot(), conversation_factory=MagicMock(return_value=conversation))
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


def test_microphone_mute_logs_only_changes(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Expose intentional microphone silence without logging repeated settings reads."""
    settings_app = FastAPI()
    stream = LocalStream(
        _robot(), conversation_factory=MagicMock(return_value=_conversation()), settings_app=settings_app
    )
    rpc = MagicMock()
    monkeypatch.setattr(console_module, "JsonRpcServer", lambda: rpc)
    stream.init_settings_ui()
    microphone = next(call.args[1] for call in rpc.register.call_args_list if call.args[0] == "conversation.mic")
    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        for params in ({}, {"muted": True}, {"muted": True}, {"muted": False}):
            assert microphone(params) == {"muted": params.get("muted", False)}

    assert [message for message in caplog.messages if "Microphone mute changed" in message] == [
        "Microphone mute changed: muted=True",
        "Microphone mute changed: muted=False",
    ]


@pytest.mark.asyncio
async def test_record_loop_warns_once_when_microphone_frames_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report a sustained capture failure without flooding logs."""
    conversation = _conversation()
    robot = _robot()
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    robot.media.get_audio_sample.return_value = None
    monkeypatch.setattr(console_module, "MICROPHONE_FRAME_TIMEOUT_SECONDS", 0.0)

    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        capture_task = asyncio.create_task(stream.record_loop())
        while robot.media.start_playing.call_count == 0:
            await asyncio.sleep(0)
        robot.media.stop_recording.assert_called_once_with()
        robot.media.start_recording.assert_called_once_with()
        robot.media.start_playing.assert_called_once_with()
        conversation.clear_playback.assert_called_once_with()
        conversation.interrupt.assert_awaited_once_with()
        stream.close()
        await asyncio.wait_for(capture_task, timeout=1.0)

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
    dependencies = SimpleNamespace(
        instance_path=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock()),
    )
    conversation = LiveConversation(dependencies, voice="gleam", output_sample_rate=48_000)
    conversation.output_queue.put_nowait(PlaybackAudio(np.ones(48_000, dtype=np.float32)))
    robot = _robot()
    recovered_audio = np.ones((2, 1_600), dtype=np.float32)
    robot.media.get_audio_sample.side_effect = lambda: recovered_audio if robot.media.start_recording.called else None
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    monkeypatch.setattr(console_module, "MICROPHONE_FRAME_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(realtime_module, "SEND_TIMEOUT_SECONDS", 0.01)

    async def interrupt_live(**_kwargs: object) -> None:
        robot.media.start_recording.assert_called_once_with()
        robot.media.start_playing.assert_called_once_with()
        assert conversation.output_queue.empty()
        if stalled:
            await asyncio.Event().wait()

    instructions = AsyncMock(side_effect=interrupt_live)
    conversation._connection = SimpleNamespace(
        session=SimpleNamespace(instructions=SimpleNamespace(append=instructions))
    )

    capture_task = asyncio.create_task(stream.record_loop())
    try:
        microphone_pcm = await asyncio.wait_for(conversation._microphone_queue.get(), timeout=1.0)
    finally:
        capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)

    robot.media.audio.clear_player.assert_called()
    robot.media.stop_recording.assert_called_once_with()
    robot.media.start_recording.assert_called_once_with()
    robot.media.start_playing.assert_called_once_with()
    assert conversation.output_queue.empty()
    assert np.max(np.frombuffer(microphone_pcm, dtype="<i2")) > 30_000
    instructions.assert_awaited_once()
    assert "Stop speaking now and listen" in instructions.await_args.kwargs["content"]
    assert ("Failed to interrupt Live after microphone recovery" in caplog.text) is stalled


@pytest.mark.asyncio
async def test_record_loop_logs_and_forwards_first_microphone_frame(caplog: pytest.LogCaptureFixture) -> None:
    """Identify the capture and forwarding boundary from the first usable frame."""
    conversation = _conversation()
    robot = _robot()
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    robot.media.get_audio_sample.return_value = np.ones((2, 160), dtype=np.float32)

    with caplog.at_level(logging.INFO, logger=console_module.__name__):
        capture_task = asyncio.create_task(stream.record_loop())
        await asyncio.sleep(0.05)
        stream.close()
        await asyncio.wait_for(capture_task, timeout=1.0)

    assert any("Microphone capture started" in message for message in caplog.messages)
    assert conversation.receive.await_count > 0


@pytest.mark.asyncio
async def test_muted_microphone_forwards_silence_without_changing_capture() -> None:
    """Keep Live's audio clock running while withholding microphone contents."""
    conversation = _conversation()
    robot = _robot()
    captured = np.ones((2, 160), dtype=np.float32)
    robot.media.get_audio_sample.return_value = captured
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    stream._mic_muted = True
    conversation.receive.side_effect = lambda _frame: stream._stop_event.set()

    await stream.record_loop()

    conversation.receive.assert_awaited_once()
    sample_rate, forwarded = conversation.receive.await_args.args[0]
    assert sample_rate == 16_000
    np.testing.assert_array_equal(forwarded, np.zeros_like(captured))
    np.testing.assert_array_equal(captured, np.ones_like(captured))


@pytest.mark.asyncio
async def test_play_loop_pushes_chunks_without_waiting_for_playback_tracking() -> None:
    """Keep the player fed and mark listening only after queued playback drains."""
    conversation = _conversation()
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
    robot = _robot()
    robot.media.push_audio_sample.side_effect = lambda samples: (
        pushed_third.set() if samples is chunks[2].samples else None
    )
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    playback_task = asyncio.create_task(stream.play_loop())
    acknowledgement_task = asyncio.create_task(stream._acknowledge_playback_loop())
    try:
        await asyncio.wait_for(pushed_third.wait(), timeout=1.0)
        assert len(tracked) == 1
        assert tracked[0] is chunks[0]
        assert not playback_ended.is_set()
        tracking_release.set()
        await asyncio.wait_for(playback_ended.wait(), timeout=1.0)
    finally:
        playback_task.cancel()
        acknowledgement_task.cancel()
        await asyncio.gather(playback_task, acknowledgement_task, return_exceptions=True)

    assert len(tracked) == 3
    assert all(actual is expected for actual, expected in zip(tracked, chunks, strict=True))
    conversation.acknowledge_playback_end.assert_called_once_with()


@pytest.mark.asyncio
async def test_interruption_discards_old_tracking_before_new_audio() -> None:
    """Clear queued playback and let fresh Live audio reach the player promptly."""
    movement_manager = SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock())
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=movement_manager,
    )
    conversation = LiveConversation(dependencies, voice="gleam", output_sample_rate=48_000)
    instructions = AsyncMock()
    conversation._connection = SimpleNamespace(
        session=SimpleNamespace(instructions=SimpleNamespace(append=instructions))
    )
    for _ in range(2):
        conversation.output_queue.put_nowait(PlaybackAudio(np.ones(48_000, dtype=np.float32)))
    old_audio_pushed = asyncio.Event()
    fresh_audio_pushed = asyncio.Event()
    robot = _robot()

    def push_audio(_samples: np.ndarray) -> None:
        if robot.media.push_audio_sample.call_count == 2:
            old_audio_pushed.set()
        elif robot.media.push_audio_sample.call_count == 3:
            fresh_audio_pushed.set()

    robot.media.push_audio_sample.side_effect = push_audio
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    playback_task = asyncio.create_task(stream.play_loop())
    acknowledgement_task = asyncio.create_task(stream._acknowledge_playback_loop())
    try:
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
    finally:
        playback_task.cancel()
        acknowledgement_task.cancel()
        await asyncio.gather(playback_task, acknowledgement_task, return_exceptions=True)


def test_close_finalizes_while_session_receiver_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the session receiver running until shutdown receives finalization."""
    conversation = _conversation()
    robot = _robot()
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
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
    monkeypatch.setattr(console_module.asyncio, "to_thread", AsyncMock())

    stream.launch()

    conversation.shutdown.assert_awaited_once_with()
    assert receiver_exited
    robot.media.stop_recording.assert_called_once_with()
    robot.media.stop_playing.assert_called_once_with()
