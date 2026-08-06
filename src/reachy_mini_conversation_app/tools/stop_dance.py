import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="stop_dance",
    description_override="Stop the current dance and clear queued robot movement.",
)
async def stop_dance_tool(context: RunContextWrapper[ToolDependencies]) -> ToolResult:
    """Stop the current dance."""
    context.context.movement_manager.clear_move_queue()
    logger.info("Stopped dance and cleared movement queue")
    return {"status": "stopped dance and cleared queue"}


stop_dance: FunctionTool = stop_dance_tool
