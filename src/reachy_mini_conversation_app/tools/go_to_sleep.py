import asyncio
import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="go_to_sleep",
    description_override="Put Reachy to sleep and stop this app when the user clearly requests it.",
)
async def go_to_sleep_tool(context: RunContextWrapper[ToolDependencies]) -> ToolResult:
    """Put Reachy to sleep and request app shutdown."""
    callback = context.context.go_to_sleep
    if callback is None:
        return {"error": "go_to_sleep is unavailable in this runtime"}
    logger.info("Tool call: go_to_sleep")
    try:
        return await asyncio.to_thread(callback)
    except Exception as error:
        logger.exception("go_to_sleep failed")
        return {"error": f"go_to_sleep failed: {type(error).__name__}: {error}"}


go_to_sleep: FunctionTool = go_to_sleep_tool
