import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, EmotionQueueMove


class FixedMove(Move):
    """Supply a stable SDK move so scenarios can observe robot output without timing interpolation."""

    def __init__(self, head: NDArray[np.float64]) -> None:
        """Hold one head pose for the duration of a scenario."""
        self.head = head

    @property
    def duration(self) -> float:
        """Keep the move active until the test stops the manager."""
        return 60.0

    def evaluate(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return a full pose accepted by the SDK motion interface."""
        return self.head, np.zeros(2), 0.0


@pytest.fixture
def robot() -> MagicMock:
    """Provide valid sensor values and observable SDK movement commands."""
    robot = MagicMock(spec=ReachyMini)
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    robot.get_tracked_face.return_value.detected = True
    return robot


def test_stop_can_skip_neutral_reset(robot: MagicMock) -> None:
    """Stopping the production worker for sleep must preserve the robot's sleep pose."""
    manager = MovementManager(robot)
    commanded = threading.Event()
    robot.set_target.side_effect = lambda **_pose: commanded.set()
    manager.start()
    try:
        assert commanded.wait(timeout=2)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.goto_target.assert_not_called()


def test_queued_antennas_move_and_idle_breathing_resumes(robot: MagicMock) -> None:
    """Commanded antennas reach the robot and resume breathing when the move finishes."""
    head_pose = np.eye(4)
    antennas = (0.4, -0.4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, list(antennas))
    commanded = threading.Event()
    breathing = threading.Event()

    def observe_target(*, head: NDArray[np.float64], antennas: list[float], body_yaw: float) -> None:
        if np.allclose(antennas, (0.4, -0.4)):
            commanded.set()
        elif commanded.is_set():
            breathing.set()

    robot.set_target.side_effect = observe_target
    manager = MovementManager(robot)
    manager.idle_inactivity_delay = 0.0
    manager.queue_move(
        GotoQueueMove(
            target_head_pose=head_pose,
            start_head_pose=head_pose,
            target_antennas=antennas,
            start_antennas=antennas,
            duration=0.05,
        )
    )
    manager.start()
    try:
        assert commanded.wait(timeout=2)
        assert breathing.wait(timeout=2)
    finally:
        manager.stop(reset_to_neutral=False)


def test_head_tracking_follows_speaking(robot: MagicMock) -> None:
    """Tracking releases the head while speaking and resumes after playback finishes."""
    tracking = threading.Event()
    paused = threading.Event()

    def observe_tracking(*, weight: float) -> None:
        (tracking if weight else paused).set()

    robot.start_head_tracking.side_effect = observe_tracking
    manager = MovementManager(robot)
    manager.start()
    try:
        manager.set_head_tracking(True)
        assert tracking.wait(timeout=2)
        manager.set_speaking(True)
        assert paused.wait(timeout=2)
        tracking.clear()
        manager.set_speaking(False)
        assert tracking.wait(timeout=2)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


@pytest.mark.parametrize("movement", ["idle", "emotion", "dance"])
def test_speaking_preserves_face_anchor_except_for_dance(robot: MagicMock, movement: str) -> None:
    """Idle speech holds the face, emotions compose onto it, and dances use their own pose."""
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    motion_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    expected = {
        "idle": anchor,
        "emotion": compose_world_offset(anchor, motion_head),
        "dance": motion_head,
    }[movement]
    robot.get_current_head_pose.return_value = anchor
    commanded = threading.Event()
    paused = threading.Event()
    robot.start_head_tracking.side_effect = lambda *, weight: paused.set() if weight == 0.0 else None

    def observe_target(*, head: NDArray[np.float64], antennas: list[float], body_yaw: float) -> None:
        if np.allclose(head, expected):
            commanded.set()

    robot.set_target.side_effect = observe_target
    manager = MovementManager(robot)
    manager.idle_inactivity_delay = 600.0
    manager.start()
    try:
        manager.set_head_tracking(True)
        manager.set_speaking(True)
        assert paused.wait(timeout=2)
        if movement == "emotion":
            recorded = MagicMock()
            recorded.get.return_value = FixedMove(motion_head)
            manager.queue_move(EmotionQueueMove("happy", recorded))
        elif movement == "dance":
            manager.queue_move(FixedMove(motion_head))
        assert commanded.wait(timeout=2)
    finally:
        manager.stop(reset_to_neutral=False)
