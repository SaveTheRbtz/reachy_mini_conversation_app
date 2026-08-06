from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


@function_tool(
    name_override="wait_for_user",
    description_override="Wait silently when the user has not addressed you or only background audio is present.",
)
async def wait_for_user_tool(context: RunContextWrapper[ToolDependencies]) -> ToolResult:
    """Acknowledge that no response should be spoken."""
    del context
    return {"status": "waiting"}


wait_for_user: FunctionTool = wait_for_user_tool
