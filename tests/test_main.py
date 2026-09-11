import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock
from collections.abc import Callable

import pytest

from tests.support.console import make_robot
import reachy_mini_conversation_app.main as main_module
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.realtime import LiveConversation


@dataclass
class App:
    """Observe launch configuration and resource cleanup at external boundaries."""

    robot: MagicMock
    movement: MagicMock
    stream: MagicMock
    conversation: MagicMock


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> App:
    """Keep startup configuration real while replacing hardware and the foreground loop."""
    harness = App(
        make_robot(),
        MagicMock(spec=MovementManager),
        MagicMock(spec=LocalStream),
        MagicMock(spec=LiveConversation),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "curious_kids_ru")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(
        main_module, "load_memory", lambda instance: MemorySnapshot(memories=["Private household memory"])
    )
    monkeypatch.setattr(app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(main_module, "MovementManager", lambda **kwargs: harness.movement)
    monkeypatch.setattr(main_module, "LiveConversation", harness.conversation)
    monkeypatch.setattr(main_module, "resolve_app_timeout_minutes", lambda: None)
    monkeypatch.setattr(main_module, "setup_logger", lambda debug: logging.getLogger(main_module.__name__))

    def stream(
        robot: object, *, conversation_factory: Callable[[str], object], startup_voice: str, **kwargs: object
    ) -> MagicMock:
        conversation_factory(startup_voice)
        return harness.stream

    monkeypatch.setattr(main_module, "LocalStream", stream)
    return harness


@pytest.mark.parametrize("no_camera", [False, True], ids=["camera-available", "camera-disabled"])
def test_launch_configures_camera_without_capturing_frames(
    app: App, caplog: pytest.LogCaptureFixture, no_camera: bool
) -> None:
    """The camera tool receives its permission without exposing images or memory at startup."""
    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        main_module.run(argparse.Namespace(debug=False, no_camera=no_camera, ui=False), robot=app.robot)

    dependencies = app.conversation.call_args.args[0]
    assert dependencies.camera_enabled is (not no_camera)
    assert dependencies.memory.memories == ["Private household memory"]
    app.stream.launch.assert_called_once_with()
    app.robot.media.get_frame.assert_not_called()
    app.robot.media.get_frame_jpeg.assert_not_called()
    assert "Private household memory" not in caplog.text


@pytest.mark.parametrize(
    "failure", [None, KeyboardInterrupt(), RuntimeError("audio failed")], ids=["normal", "interrupt", "error"]
)
def test_exit_releases_robot_resources(app: App, failure: BaseException | None) -> None:
    """Every foreground-loop exit stops movement and disconnects the SDK client."""
    app.stream.launch.side_effect = failure
    arguments = argparse.Namespace(debug=False, no_camera=True, ui=False)
    if isinstance(failure, RuntimeError):
        with pytest.raises(RuntimeError, match="audio failed"):
            main_module.run(arguments, robot=app.robot)
    else:
        main_module.run(arguments, robot=app.robot)

    app.movement.stop.assert_called_once_with(reset_to_neutral=False)
    app.robot.disable_wobbling.assert_called_once_with()
    app.robot.media.close.assert_called_once_with()
    app.robot.client.disconnect.assert_called_once_with()


def test_sleep_still_stops_app_when_movement_fails(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed sleep motion cannot prevent local app shutdown or cause duplicate requests."""
    app.robot.goto_sleep.side_effect = RuntimeError("motor unavailable")
    request_stop = MagicMock(return_value=False)
    monkeypatch.setattr(app_lifecycle, "request_stop_current_app", request_stop)
    results: list[dict[str, object]] = []

    def request_sleep() -> None:
        dependencies = app.conversation.call_args.args[0]
        results.extend([dependencies.go_to_sleep(), dependencies.go_to_sleep()])

    app.stream.launch.side_effect = request_sleep
    main_module.run(argparse.Namespace(debug=False, no_camera=True, ui=False), robot=app.robot)

    assert results[0]["status"] == "stop_requested"
    error = results[0]["error"]
    assert isinstance(error, str)
    assert "motor unavailable" in error
    assert results[1]["status"] == "already_requested"
    request_stop.assert_called_once()
    app.stream.close.assert_called_once_with()
    app.robot.goto_sleep.assert_called_once_with()
    app.robot.client.disconnect.assert_called_once_with()
