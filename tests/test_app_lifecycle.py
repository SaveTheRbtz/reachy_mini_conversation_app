import logging
import urllib.error
import urllib.request
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.support.console import make_robot
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app import app_lifecycle


@pytest.mark.parametrize("unavailable", [False, True], ids=["accepted", "daemon-unavailable"])
def test_stop_request_uses_connected_daemon_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    """A stop request targets the current daemon and degrades when it cannot be reached."""
    robot = make_robot()
    robot.client.host = "192.168.1.42"
    robot.client.port = 8000
    response = MagicMock()

    def urlopen(request: urllib.request.Request, timeout: float) -> MagicMock:
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert 0 < timeout <= 2
        if unavailable:
            raise urllib.error.URLError("daemon unavailable")
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert app_lifecycle.request_stop_current_app(robot, logging.getLogger(__name__)) is (not unavailable)


def test_sleeping_robot_enables_motors_before_waking() -> None:
    """Waking a sleeping robot requires powered motors before movement."""
    robot = make_robot()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    movements: list[str] = []
    robot.enable_motors.side_effect = lambda: movements.append("motors-enabled")
    robot.wake_up.side_effect = lambda: movements.append("wake-up")

    assert app_lifecycle.wake_up_if_sleeping(robot, logging.getLogger(__name__))
    assert movements == ["motors-enabled", "wake-up"]


def test_awake_robot_is_left_in_place() -> None:
    """Startup must not move an already-awake robot."""
    robot = make_robot()
    robot.get_current_head_pose.return_value = np.eye(4)

    assert not app_lifecycle.wake_up_if_sleeping(robot, logging.getLogger(__name__))
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


@pytest.mark.parametrize("failed_operation", ["get_current_head_pose", "enable_motors", "wake_up"])
def test_wake_failure_leaves_app_startup_available(failed_operation: str, caplog: pytest.LogCaptureFixture) -> None:
    """Unavailable hardware must be reported without preventing dialogue startup."""
    robot = make_robot()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    getattr(robot, failed_operation).side_effect = RuntimeError("motor unavailable")

    assert not app_lifecycle.wake_up_if_sleeping(robot, logging.getLogger(__name__))
    assert "motor unavailable" in caplog.text
