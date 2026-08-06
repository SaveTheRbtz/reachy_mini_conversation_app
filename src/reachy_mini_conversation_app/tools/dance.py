import random
import logging

from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove


logger = logging.getLogger(__name__)


@function_tool(
    name_override="dance",
    description_override="Queue a named dance, or a random available dance when no name is supplied.",
)
async def dance_tool(
    context: RunContextWrapper[ToolDependencies],
    move: str | None = None,
    repeat: int = 1,
) -> ToolResult:
    """Queue a dance without blocking the conversation."""
    available_names = list(AVAILABLE_MOVES)
    if not available_names:
        return {"error": "No dances are available"}
    if repeat < 1 or repeat > 5:
        return {"error": "repeat must be between 1 and 5"}
    move_name = move or random.choice(available_names)
    if move_name not in AVAILABLE_MOVES:
        return {"error": f"Unknown dance move {move_name!r}", "available": available_names}
    try:
        for _ in range(repeat):
            context.context.movement_manager.queue_move(DanceQueueMove(move_name))
    except Exception as error:
        logger.exception("Failed to queue dance %s", move_name)
        return {"error": f"Failed to queue dance: {type(error).__name__}: {error}"}
    logger.info("Queued dance move=%s repeat=%d", move_name, repeat)
    return {"status": "queued", "move": move_name, "repeat": repeat}


dance: FunctionTool = dance_tool
