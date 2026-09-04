import asyncio
import logging
from types import SimpleNamespace
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from agents.realtime import RealtimeAudio, RealtimeAudioEnd, RealtimeEventInfo, RealtimeModelAudioEvent

import reachy_mini_conversation_app.console as console_module
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import PlaybackAudio, RealtimeConversation


def _conversation() -> SimpleNamespace:
    return SimpleNamespace(
        voice="marin",
        last_activity_time=0.0,
        connected=True,
        shutdown=AsyncMock(),
        receive=AsyncMock(),
        emit=AsyncMock(),
        interrupt=AsyncMock(),
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
        voice="marin",
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

    assert any("Realtime connection attempt: attempt=1" in message for message in caplog.messages)
    assert any(f"reason={reason} elapsed=" in message for message in caplog.messages)
    if reason in {"remote_close", "error"}:
        retry.assert_awaited_once_with(console_module.RETRY_DELAY_SECONDS)
        assert any("Realtime reconnect scheduled" in message for message in caplog.messages)
    else:
        retry.assert_not_awaited()
        assert not any("Realtime reconnect scheduled" in message for message in caplog.messages)
    if reason == "error":
        assert any(
            "Realtime session failed: attempt=1 elapsed=" in message
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
        conversation.interrupt.assert_awaited_once_with()
        stream.close()
        await asyncio.wait_for(capture_task, timeout=1.0)

    assert sum("No usable microphone frames received" in message for message in caplog.messages) == 1
    conversation.receive.assert_not_awaited()


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
async def test_play_loop_pushes_chunks_without_waiting_for_playback_tracking(caplog: pytest.LogCaptureFixture) -> None:
    """Keep the robot player fed while playback accounting follows in order."""
    conversation = _conversation()
    first = PlaybackAudio("item", 0, np.ones(19_200, dtype=np.float32))
    second = PlaybackAudio("item", 0, np.ones(19_200, dtype=np.float32))
    third = PlaybackAudio("next-item", 0, np.ones(19_200, dtype=np.float32))
    pending_audio: deque[PlaybackAudio | None] = deque((first, second, third, None))
    tracking_release = asyncio.Event()
    playback_ended = asyncio.Event()
    pushed_third = asyncio.Event()
    tracked: list[PlaybackAudio] = []

    async def emit() -> PlaybackAudio | None:
        if pending_audio:
            return pending_audio.popleft()
        await asyncio.Event().wait()
        return None

    async def acknowledge(audio: PlaybackAudio) -> None:
        tracked.append(audio)
        if audio is first:
            await tracking_release.wait()

    conversation.emit.side_effect = emit
    conversation.acknowledge_after_playback.side_effect = acknowledge
    conversation.acknowledge_playback_end.side_effect = playback_ended.set
    robot = _robot()
    robot.media.push_audio_sample.side_effect = lambda samples: (
        pushed_third.set() if samples is third.samples else None
    )
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    caplog.set_level(logging.INFO, logger=console_module.__name__)

    playback_task = asyncio.create_task(stream.play_loop())
    acknowledgement_task = asyncio.create_task(stream._acknowledge_playback_loop())
    await asyncio.wait_for(pushed_third.wait(), timeout=1.0)

    assert tracked == [first]
    assert not tracking_release.is_set()

    tracking_release.set()
    await asyncio.wait_for(playback_ended.wait(), timeout=1.0)
    playback_task.cancel()
    acknowledgement_task.cancel()
    await asyncio.gather(playback_task, acknowledgement_task, return_exceptions=True)

    assert tracked == [first, second, third]
    assert [message for message in caplog.messages if "Robot audio submitted" in message] == [
        "Robot audio submitted: item_id=item samples=19200",
        "Robot audio submitted: item_id=next-item samples=19200",
    ]


@pytest.mark.asyncio
async def test_interruption_discards_old_tracking_before_new_audio(caplog: pytest.LogCaptureFixture) -> None:
    """Never report buffered chunks cleared by an interruption as played."""
    movement_manager = SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock())
    dependencies = SimpleNamespace(
        instance_path=None,
        send_image=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=movement_manager,
    )
    conversation = RealtimeConversation(dependencies, voice="marin", output_sample_rate=48_000)
    conversation._session = SimpleNamespace(interrupt=AsyncMock())
    old_first = PlaybackAudio("old", 0, np.ones(48_000, dtype=np.float32))
    old_second = PlaybackAudio("old", 0, np.ones(48_000, dtype=np.float32))
    conversation.output_queue.put_nowait(old_first)
    conversation.output_queue.put_nowait(old_second)
    old_audio_pushed = asyncio.Event()
    new_audio_tracked = asyncio.Event()
    tracker = MagicMock()
    tracker.on_play_ms.side_effect = lambda *_args: new_audio_tracked.set()
    conversation._playback_tracker = tracker
    robot = _robot()

    def push_audio(_samples: np.ndarray) -> None:
        if robot.media.push_audio_sample.call_count == 2:
            old_audio_pushed.set()

    robot.media.push_audio_sample.side_effect = push_audio
    stream = LocalStream(robot, conversation_factory=MagicMock(return_value=conversation))
    caplog.set_level(logging.INFO, logger=console_module.__name__)
    playback_task = asyncio.create_task(stream.play_loop())
    acknowledgement_task = asyncio.create_task(stream._acknowledge_playback_loop())
    await asyncio.wait_for(old_audio_pushed.wait(), timeout=1.0)
    await asyncio.sleep(0)

    await conversation.interrupt()
    new_pcm16 = np.ones(480, dtype="<i2").tobytes()
    await conversation._handle_event(
        RealtimeAudio(
            audio=RealtimeModelAudioEvent(
                data=new_pcm16,
                response_id="new-response",
                item_id="new",
                content_index=0,
            ),
            item_id="new",
            content_index=0,
            info=RealtimeEventInfo(context=MagicMock()),
        )
    )
    await conversation._handle_event(
        RealtimeAudioEnd(
            info=RealtimeEventInfo(context=MagicMock()),
            item_id="new",
            content_index=0,
        )
    )

    await asyncio.wait_for(new_audio_tracked.wait(), timeout=1.0)
    playback_task.cancel()
    acknowledgement_task.cancel()
    await asyncio.gather(playback_task, acknowledgement_task, return_exceptions=True)

    tracker.on_play_ms.assert_called_once_with("new", 0, 20.0)
    robot.media.audio.clear_player.assert_called_once_with()
    assert "Clearing robot playback: pending_acknowledgements=1" in caplog.messages
