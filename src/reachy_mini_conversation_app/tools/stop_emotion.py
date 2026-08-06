import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="stop_emotion",
    description_override="Stop the current robot emotion and clear queued movement.",
)
async def stop_emotion_tool(context: RunContextWrapper[ToolDependencies]) -> ToolResult:
    """Stop the current emotion."""
    context.context.movement_manager.clear_move_queue()
    logger.info("Stopped emotion and cleared movement queue")
    return {"status": "stopped emotion and cleared queue"}


stop_emotion: FunctionTool = stop_emotion_tool
