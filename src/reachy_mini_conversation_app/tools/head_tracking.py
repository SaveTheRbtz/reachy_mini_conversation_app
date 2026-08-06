import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="head_tracking",
    description_override="Enable or disable following the user's face with the robot's head.",
)
async def head_tracking_tool(context: RunContextWrapper[ToolDependencies], enabled: bool) -> ToolResult:
    """Toggle head tracking."""
    try:
        context.context.movement_manager.set_head_tracking(enabled)
    except Exception as error:
        logger.exception("Head tracking update failed")
        return {"error": f"Head tracking failed: {type(error).__name__}: {error}"}
    logger.info("Head tracking enabled=%s", enabled)
    return {"status": "following" if enabled else "stopped following"}


head_tracking: FunctionTool = head_tracking_tool
