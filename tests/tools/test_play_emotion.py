import json
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
from concurrent.futures import Future

import pytest
from agents.tool_context import ToolContext

import reachy_mini_conversation_app.tools.play_emotion as emotion_module


def test_cancelled_emotion_load_stays_off_loop_and_is_reused_after_relaunch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a single background load across cancellation and a replacement event loop."""
    started = threading.Event()
    release = threading.Event()
    recorded_moves = MagicMock()
    recorded_moves.list_moves.return_value = ["loving1"]

    def load_moves(_dataset: str) -> MagicMock:
        assert threading.current_thread() is not threading.main_thread()
        started.set()
        assert release.wait(timeout=3)
        return recorded_moves

    constructor = MagicMock(side_effect=load_moves)
    monkeypatch.setattr(emotion_module, "RecordedMoves", constructor)
    monkeypatch.setattr(emotion_module, "_recorded_moves", None)
    movement_manager = MagicMock()
    arguments = '{"emotion": "loving"}'
    context = ToolContext(
        SimpleNamespace(movement_manager=movement_manager),
        tool_name="play_emotion",
        tool_call_id="emotion-call",
        tool_arguments=arguments,
    )

    async def interrupted_call() -> None:
        task = asyncio.ensure_future(emotion_module.play_emotion.on_invoke_tool(context, arguments))
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            movement_manager.queue_move.assert_not_called()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def resumed_call() -> None:
        task = asyncio.ensure_future(emotion_module.play_emotion.on_invoke_tool(context, arguments))
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 1) == {"status": "queued", "emotion": "loving1"}
        assert await emotion_module.play_emotion.on_invoke_tool(context, arguments) == {
            "status": "queued",
            "emotion": "loving1",
        }

    try:
        asyncio.run(interrupted_call())
        asyncio.run(resumed_call())
    finally:
        release.set()
    constructor.assert_called_once_with("pollen-robotics/reachy-mini-emotions-library")
    assert movement_manager.queue_move.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_emotion_load_failure_is_logged_and_next_request_can_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, cancelled: bool
) -> None:
    """Report a download failure and allow a later explicit request to load again."""
    started = threading.Event()
    release = threading.Event()
    recorded_moves = MagicMock()
    recorded_moves.list_moves.return_value = ["loving1"]

    def load_moves(_dataset: str) -> MagicMock:
        if constructor.call_count == 1:
            started.set()
            assert release.wait(timeout=3)
            raise OSError("Download unavailable")
        return recorded_moves

    constructor = MagicMock(side_effect=load_moves)
    monkeypatch.setattr(emotion_module, "RecordedMoves", constructor)
    monkeypatch.setattr(emotion_module, "_recorded_moves", None)
    movement_manager = MagicMock()
    arguments = '{"emotion": "loving"}'
    context = ToolContext(
        SimpleNamespace(movement_manager=movement_manager),
        tool_name="play_emotion",
        tool_call_id="emotion-call",
        tool_arguments=arguments,
    )

    task = asyncio.ensure_future(emotion_module.play_emotion.on_invoke_tool(context, arguments))
    try:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0)
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        if not cancelled:
            assert await asyncio.wait_for(task, 1) == {
                "error": "Failed to play emotion: OSError: Download unavailable"
            }
        async with asyncio.timeout(1):
            while "Failed to load emotions: OSError: Download unavailable" not in caplog.text:
                await asyncio.sleep(0)
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    movement_manager.queue_move.assert_not_called()
    assert await emotion_module.play_emotion.on_invoke_tool(context, arguments) == {
        "status": "queued",
        "emotion": "loving1",
    }
    assert constructor.call_count == 2
    movement_manager.queue_move.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("emotion", ["unrecognized", "радость", ""])
async def test_unmatched_emotion_does_not_queue_random_movement(monkeypatch: pytest.MonkeyPatch, emotion: str) -> None:
    """Reject an unmatched request without substituting a random emotion."""
    recorded_moves = MagicMock()
    recorded_moves.list_moves.return_value = ["laughing2", "loving1"]
    loaded: Future[MagicMock] = Future()
    loaded.set_result(recorded_moves)
    monkeypatch.setattr(emotion_module, "_recorded_moves", loaded)
    movement_manager = MagicMock()
    arguments = json.dumps({"emotion": emotion})
    context = ToolContext(
        SimpleNamespace(movement_manager=movement_manager),
        tool_name="play_emotion",
        tool_call_id="emotion-call",
        tool_arguments=arguments,
    )

    output = await emotion_module.play_emotion.on_invoke_tool(context, arguments)

    assert "error" in output
    movement_manager.queue_move.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("emotion", ["happy", "random"])
async def test_requested_emotion_queues_matching_recording(monkeypatch: pytest.MonkeyPatch, emotion: str) -> None:
    """Queue the requested happy emotion or an explicitly random recording."""
    recorded_moves = MagicMock()
    recorded_moves.list_moves.return_value = ["laughing2"]
    loaded: Future[MagicMock] = Future()
    loaded.set_result(recorded_moves)
    monkeypatch.setattr(emotion_module, "_recorded_moves", loaded)
    movement_manager = MagicMock()
    arguments = json.dumps({"emotion": emotion})
    context = ToolContext(
        SimpleNamespace(movement_manager=movement_manager),
        tool_name="play_emotion",
        tool_call_id="emotion-call",
        tool_arguments=arguments,
    )

    assert await emotion_module.play_emotion.on_invoke_tool(context, arguments) == {
        "status": "queued",
        "emotion": "laughing2",
    }
    movement_manager.queue_move.assert_called_once()
    assert movement_manager.queue_move.call_args.args[0].emotion_name == "laughing2"
