import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="remember",
    description_override=(
        "Save one durable, non-sensitive fact explicitly stated by the user. "
        "Pass replaces_memory_id=null for a new memory, or the exact ID shown inside square brackets without brackets "
        "when correcting one."
    ),
)
async def remember_tool(
    context: RunContextWrapper[ToolDependencies],
    fact: str,
    replaces_memory_id: str | None,
) -> ToolResult:
    """Save or replace one explicit long-term memory note."""
    try:
        change = context.context.memory.remember(fact, replaces_memory_id=replaces_memory_id)
    except (OSError, ValueError) as error:
        logger.warning("Rejected memory write: %s", error)
        return {"error": f"Failed to save memory: {error}"}
    logger.info("Memory %s id=%s", change.status, change.note.id)
    return {
        "status": change.status,
        "memory": change.note.text,
        "memory_id": change.note.id,
    }


remember: FunctionTool = remember_tool
