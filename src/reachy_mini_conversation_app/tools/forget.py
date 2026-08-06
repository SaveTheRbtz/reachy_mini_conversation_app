import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="forget",
    description_override="Remove one saved memory using the exact bracketed memory ID in the instructions.",
)
async def forget_tool(context: RunContextWrapper[ToolDependencies], memory_id: str) -> ToolResult:
    """Forget one memory note by exact ID."""
    try:
        removed = context.context.memory.forget(memory_id)
    except (OSError, ValueError) as error:
        logger.warning("Failed to remove memory: %s", error)
        return {"error": f"Failed to remove memory: {error}"}
    if removed is None:
        return {"error": f"Unknown memory id: {memory_id}"}
    logger.info("Removed memory id=%s", removed.id)
    return {"removed": removed.text, "memory_id": removed.id}


forget: FunctionTool = forget_tool
