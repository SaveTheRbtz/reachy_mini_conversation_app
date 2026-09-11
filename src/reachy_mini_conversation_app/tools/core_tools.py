from typing import Final, TypedDict
from collections.abc import Iterable

from agents import FunctionTool

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.dance import dance
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.move_head import move_head
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.tools.stop_dance import stop_dance
from reachy_mini_conversation_app.tools.sweep_look import sweep_look
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.play_emotion import play_emotion
from reachy_mini_conversation_app.tools.stop_emotion import stop_emotion
from reachy_mini_conversation_app.tools.head_tracking import head_tracking
from reachy_mini_conversation_app.tools.manage_memory import manage_memory


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
)


def selected_tool_names(instance_path: str | None = None) -> list[str]:
    """Return tool IDs enabled for the active profile."""
    return read_profile_tool_names(config.REACHY_MINI_CUSTOM_PROFILE, instance_path)


def get_function_tools(enabled_tool_names: Iterable[str]) -> list[FunctionTool]:
    """Resolve enabled local function tools from the static registry."""
    enabled = {entry["id"] for entry in available_tool_catalog(enabled_tool_names)}
    return [tool for tool in TOOLS if tool.name in enabled]


def available_tool_catalog(enabled_tool_names: Iterable[str] | None = None) -> list[ToolCatalogEntry]:
    """Return tool metadata, optionally restricted to validated profile selections."""
    catalog: list[ToolCatalogEntry] = [{"id": tool.name, "description": tool.description} for tool in TOOLS]
    catalog.append(
        {
            "id": "web_search",
            "description": "Search the public web for current information, including weather and local time.",
        }
    )
    if enabled_tool_names is not None:
        enabled = set(enabled_tool_names)
        unknown = enabled - {entry["id"] for entry in catalog}
        if unknown:
            raise ValueError(f"Unknown profile tools: {', '.join(sorted(unknown))}")
        catalog = [entry for entry in catalog if entry["id"] in enabled]
    return sorted(catalog, key=lambda entry: entry["id"])


__all__ = ["ToolDependencies", "available_tool_catalog", "get_function_tools", "selected_tool_names"]
