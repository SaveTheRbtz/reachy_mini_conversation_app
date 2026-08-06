from typing import Final, TypedDict
from collections.abc import Iterable

from agents import Tool, FunctionTool

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.dance import dance
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.move_head import move_head
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.tools.stop_dance import stop_dance
from reachy_mini_conversation_app.tools.sweep_look import sweep_look
from reachy_mini_conversation_app.tools.web_search import web_search
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.play_emotion import play_emotion
from reachy_mini_conversation_app.tools.stop_emotion import stop_emotion
from reachy_mini_conversation_app.tools.head_tracking import head_tracking
from reachy_mini_conversation_app.tools.manage_memory import manage_memory
from reachy_mini_conversation_app.tools.wait_for_user import wait_for_user


class ToolCatalogEntry(TypedDict):
    """Metadata used by the personality settings UI."""

    id: str
    description: str


TOOLS: Final[tuple[FunctionTool, ...]] = (
    camera,
    dance,
    go_to_sleep,
    head_tracking,
    manage_memory,
    move_head,
    play_emotion,
    stop_dance,
    stop_emotion,
    sweep_look,
    wait_for_user,
    web_search,
)
TOOLS_BY_NAME: Final = {tool.name: tool for tool in TOOLS}


def selected_tool_names(instance_path: str | None = None) -> list[str]:
    """Return tool IDs enabled for the active profile."""
    return read_profile_tool_names(config.REACHY_MINI_CUSTOM_PROFILE, instance_path)


def get_function_tools(enabled_tool_names: Iterable[str]) -> list[Tool]:
    """Resolve enabled tools from the static registry."""
    enabled = set(enabled_tool_names)
    unknown = enabled - set(TOOLS_BY_NAME)
    if unknown:
        raise ValueError(f"Unknown profile tools: {', '.join(sorted(unknown))}")
    return [tool for tool in TOOLS if tool.name in enabled]


def available_tool_catalog() -> list[ToolCatalogEntry]:
    """Return the fixed tool catalog."""
    return sorted(
        [
            {
                "id": tool.name,
                "description": tool.description,
            }
            for tool in TOOLS
        ],
        key=lambda entry: entry["id"],
    )


__all__ = ["ToolDependencies", "available_tool_catalog", "get_function_tools", "selected_tool_names"]
