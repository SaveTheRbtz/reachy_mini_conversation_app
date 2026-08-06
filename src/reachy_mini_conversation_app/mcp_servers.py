import logging
from typing import Final
from dataclasses import dataclass
from collections.abc import Iterable

from agents.mcp import MCPServerManager, MCPServerStreamableHttp, create_static_tool_filter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicMCPTool:
    """Describe one curated public MCP integration."""

    profile_id: str
    server_name: str
    remote_tool_name: str
    url: str
    description: str


PUBLIC_MCP_TOOLS: Final = (
    PublicMCPTool(
        profile_id="pollen_robotics_reachy_mini_search_tool__search_web",
        server_name="reachy_mini_search",
        remote_tool_name="reachy_mini_search_tool_search_web",
        url="https://pollen-robotics-reachy-mini-search-tool.hf.space/gradio_api/mcp/",
        description="Search the public web for current information.",
    ),
    PublicMCPTool(
        profile_id="pollen_robotics_reachy_mini_weather_tool__get_weather",
        server_name="reachy_mini_weather",
        remote_tool_name="reachy_mini_weather_tool_get_weather",
        url="https://pollen-robotics-reachy-mini-weather-tool.hf.space/gradio_api/mcp/",
        description="Get current weather for a place.",
    ),
    PublicMCPTool(
        profile_id="pollen_robotics_reachy_mini_time_tool__get_time",
        server_name="reachy_mini_time",
        remote_tool_name="reachy_mini_time_tool_get_time",
        url="https://pollen-robotics-reachy-mini-time-tool.hf.space/gradio_api/mcp/",
        description="Get current time for an IANA timezone.",
    ),
)
PUBLIC_MCP_TOOL_IDS: Final = frozenset(tool.profile_id for tool in PUBLIC_MCP_TOOLS)


def create_mcp_manager(enabled_tool_ids: Iterable[str]) -> MCPServerManager:
    """Build a resilient manager for the selected curated MCP tools."""
    enabled = set(enabled_tool_ids)
    servers = [
        MCPServerStreamableHttp(
            params={"url": tool.url, "timeout": 10.0},
            name=tool.server_name,
            cache_tools_list=True,
            tool_filter=create_static_tool_filter(allowed_tool_names=[tool.remote_tool_name]),
        )
        for tool in PUBLIC_MCP_TOOLS
        if tool.profile_id in enabled
    ]
    return MCPServerManager(
        servers,
        strict=False,
        drop_failed_servers=True,
        connect_in_parallel=True,
    )


def log_mcp_failures(manager: MCPServerManager) -> None:
    """Log unavailable optional MCP servers without failing the conversation."""
    for server, error in manager.errors.items():
        logger.warning("MCP server %s is unavailable: %s", server.name, error)
