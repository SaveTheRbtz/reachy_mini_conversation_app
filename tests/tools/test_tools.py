import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents.tool_context import ToolContext

from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.core_tools import get_function_tools
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.head_tracking import head_tracking


@pytest.mark.asyncio
async def test_camera_submits_image_without_returning_base64() -> None:
    """Submit a captured image without leaking encoded image data."""
    jpeg = b"current jpeg"
    image_sender = AsyncMock()
    robot = SimpleNamespace(media=SimpleNamespace(get_frame_jpeg=MagicMock(return_value=jpeg)))
    dependencies = SimpleNamespace(reachy_mini=robot, camera_enabled=True, send_image=image_sender)

    result = await camera.on_invoke_tool(
        ToolContext(
            dependencies,
            tool_name="camera_tool",
            tool_call_id="camera-call",
            tool_arguments='{"question": "What is the user holding?"}',
        ),
        json.dumps({"question": "What is the user holding?"}),
    )

    image_sender.assert_awaited_once_with("What is the user holding?", jpeg)
    assert result == {"status": "image submitted", "question": "What is the user holding?"}
    assert "base64" not in str(result)


@pytest.mark.asyncio
async def test_head_tracking_uses_typed_boolean_parameter() -> None:
    """Pass the declared boolean parameter to the movement manager."""
    movement_manager = SimpleNamespace(set_head_tracking=MagicMock())
    dependencies = SimpleNamespace(movement_manager=movement_manager)

    result = await head_tracking.on_invoke_tool(
        ToolContext(
            dependencies,
            tool_name="head_tracking_tool",
            tool_call_id="tracking-call",
            tool_arguments='{"enabled": true}',
        ),
        json.dumps({"enabled": True}),
    )

    movement_manager.set_head_tracking.assert_called_once_with(True)
    assert result == {"status": "following"}


@pytest.mark.asyncio
async def test_go_to_sleep_degrades_when_callback_is_unavailable() -> None:
    """Return a tool error when sleep control is unavailable."""
    dependencies = SimpleNamespace(go_to_sleep=None)

    result = await go_to_sleep.on_invoke_tool(
        ToolContext(
            dependencies,
            tool_name="go_to_sleep_tool",
            tool_call_id="sleep-call",
            tool_arguments="{}",
        ),
        "{}",
    )

    assert result == {"error": "go_to_sleep is unavailable in this runtime"}


def test_static_tool_registry_rejects_unknown_profile_entries() -> None:
    """Fail profile validation when a tool is outside the fixed catalog."""
    with pytest.raises(ValueError, match="Unknown profile tools: removed_tool"):
        get_function_tools(["camera", "removed_tool"])

    assert [tool.name for tool in get_function_tools(["camera", "wait_for_user"])] == [
        "camera",
        "wait_for_user",
    ]
