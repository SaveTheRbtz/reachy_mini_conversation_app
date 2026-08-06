import logging
from typing import Final, Literal

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)
Direction = Literal["left", "right", "up", "down", "front"]
HEAD_DELTAS: Final[dict[Direction, tuple[int, int, int, int, int, int]]] = {
    "left": (0, 0, 0, 0, 0, 40),
    "right": (0, 0, 0, 0, 0, -40),
    "up": (0, 0, 0, 0, -30, 0),
    "down": (0, 0, 0, 0, 30, 0),
    "front": (0, 0, 0, 0, 0, 0),
}


@function_tool(
    name_override="move_head",
    description_override="Move the robot's head left, right, up, down, or to the front.",
)
async def move_head_tool(context: RunContextWrapper[ToolDependencies], direction: Direction) -> ToolResult:
    """Queue a short directional head movement."""
    dependencies = context.context
    try:
        target = create_head_pose(*HEAD_DELTAS[direction], degrees=True)
        current_head_pose = dependencies.reachy_mini.get_current_head_pose()
        head_joints, current_antennas = dependencies.reachy_mini.get_current_joint_positions()
        movement = GotoQueueMove(
            target_head_pose=target,
            start_head_pose=current_head_pose,
            target_antennas=(0, 0),
            start_antennas=(current_antennas[0], current_antennas[1]),
            target_body_yaw=0,
            start_body_yaw=head_joints[0],
            duration=dependencies.motion_duration_s,
        )
        dependencies.movement_manager.queue_move(movement)
    except Exception as error:
        logger.exception("move_head failed")
        return {"error": f"move_head failed: {type(error).__name__}: {error}"}
    logger.info("Queued head movement direction=%s", direction)
    return {"status": f"looking {direction}"}


move_head: FunctionTool = move_head_tool
