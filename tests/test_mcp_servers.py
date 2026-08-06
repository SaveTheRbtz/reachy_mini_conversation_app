from reachy_mini_conversation_app.mcp_servers import PUBLIC_MCP_TOOLS, create_mcp_manager


def test_mcp_manager_contains_only_selected_curated_server() -> None:
    """Build only the allowlisted MCP server selected by a profile."""
    selected = PUBLIC_MCP_TOOLS[1]

    manager = create_mcp_manager([selected.profile_id, "unknown"])

    assert manager.strict is False
    assert manager.drop_failed_servers is True
    assert manager.connect_in_parallel is True
    assert len(manager.all_servers) == 1
    server = manager.all_servers[0]
    assert server.name == selected.server_name
    assert server.params["url"] == selected.url
    assert server.cache_tools_list is True
    assert server.tool_filter == {"allowed_tool_names": [selected.remote_tool_name]}
