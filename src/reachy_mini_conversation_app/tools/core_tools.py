from typing import Final, Literal, TypedDict
from collections.abc import Iterable

from agents import Tool, FunctionTool

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_servers import PUBLIC_MCP_TOOLS, PUBLIC_MCP_TOOL_IDS
from reachy_mini_conversation_app.tools.dance import dance
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.forget import forget
from reachy_mini_conversation_app.tools.remember import remember
from reachy_mini_conversation_app.tools.move_head import move_head
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_names
from reachy_mini_conversation_app.tools.stop_dance import stop_dance
from reachy_mini_conversation_app.tools.sweep_look import sweep_look
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.play_emotion import play_emotion
from reachy_mini_conversation_app.tools.stop_emotion import stop_emotion
from reachy_mini_conversation_app.tools.head_tracking import head_tracking
from reachy_mini_conversation_app.tools.wait_for_user import wait_for_user


class ToolCatalogEntry(TypedDict):
    """Metadata used by the personality settings UI."""

    id: str
    kind: Literal["local", "mcp"]
    source: str
    description: str


LOCAL_TOOLS: Final[tuple[FunctionTool, ...]] = (
    camera,
    dance,
    forget,
    go_to_sleep,
    head_tracking,
    move_head,
    play_emotion,
    remember,
    stop_dance,
    stop_emotion,
    sweep_look,
    wait_for_user,
)
LOCAL_TOOLS_BY_NAME: Final = {tool.name: tool for tool in LOCAL_TOOLS}


def selected_tool_names(instance_path: str | None = None) -> list[str]:
    """Return tool IDs enabled for the active profile."""
    return read_profile_tool_names(config.REACHY_MINI_CUSTOM_PROFILE, instance_path)


def get_function_tools(enabled_tool_names: Iterable[str]) -> list[Tool]:
    """Resolve enabled local tools from the static registry."""
    enabled = set(enabled_tool_names)
    unknown = enabled - set(LOCAL_TOOLS_BY_NAME) - PUBLIC_MCP_TOOL_IDS
    if unknown:
        raise ValueError(f"Unknown profile tools: {', '.join(sorted(unknown))}")
    return [tool for tool in LOCAL_TOOLS if tool.name in enabled]


def available_tool_catalog() -> list[ToolCatalogEntry]:
    """Return the fixed catalog of local and curated MCP tools."""
    local_entries: list[ToolCatalogEntry] = [
        {
            "id": tool.name,
            "kind": "local",
            "source": "Built-in",
            "description": tool.description,
        }
        for tool in LOCAL_TOOLS
    ]
    mcp_entries: list[ToolCatalogEntry] = [
        {
            "id": tool.profile_id,
            "kind": "mcp",
            "source": tool.server_name,
            "description": tool.description,
        }
        for tool in PUBLIC_MCP_TOOLS
    ]
    return sorted([*local_entries, *mcp_entries], key=lambda entry: entry["id"])


__all__ = ["ToolDependencies", "available_tool_catalog", "get_function_tools", "selected_tool_names"]
