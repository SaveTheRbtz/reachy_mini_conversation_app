import logging

import numpy as np
from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)


@function_tool(
    name_override="sweep_look",
    description_override="Sweep the robot's gaze left and right, then return to center.",
)
async def sweep_look_tool(context: RunContextWrapper[ToolDependencies]) -> ToolResult:
    """Queue one complete left-to-right sweep."""
    dependencies = context.context
    try:
        dependencies.movement_manager.clear_move_queue()
        current_head_pose = dependencies.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = dependencies.reachy_mini.get_current_joint_positions()
        body_yaw = head_joints[0]
        antennas = (antenna_joints[0], antenna_joints[1])
        max_angle = 0.9 * np.pi
        transition_duration = 3.0
        hold_duration = 1.0
        left = create_head_pose(0, 0, 0, 0, 0, max_angle, degrees=False)
        center = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)
        right = create_head_pose(0, 0, 0, 0, 0, -max_angle, degrees=False)
        waypoints = (
            (current_head_pose, left, body_yaw, body_yaw + max_angle, transition_duration),
            (left, left, body_yaw + max_angle, body_yaw + max_angle, hold_duration),
            (left, center, body_yaw + max_angle, body_yaw, transition_duration),
            (center, right, body_yaw, body_yaw - max_angle, transition_duration),
            (right, right, body_yaw - max_angle, body_yaw - max_angle, hold_duration),
            (right, center, body_yaw - max_angle, body_yaw, transition_duration),
        )
        for start_pose, target_pose, start_yaw, target_yaw, duration in waypoints:
            dependencies.movement_manager.queue_move(
                GotoQueueMove(
                    target_head_pose=target_pose,
                    start_head_pose=start_pose,
                    target_antennas=antennas,
                    start_antennas=antennas,
                    target_body_yaw=target_yaw,
                    start_body_yaw=start_yaw,
                    duration=duration,
                )
            )
    except Exception as error:
        logger.exception("Failed to queue gaze sweep")
        return {"error": f"Failed to sweep gaze: {type(error).__name__}: {error}"}
    logger.info("Queued gaze sweep")
    return {"status": "sweeping left, right, then center"}


sweep_look: FunctionTool = sweep_look_tool
