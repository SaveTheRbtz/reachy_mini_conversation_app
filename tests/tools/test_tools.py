import json
import logging
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image
from agents.tool_context import ToolContext

from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.core_tools import get_function_tools
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.head_tracking import head_tracking


@pytest.mark.asyncio
async def test_camera_submits_current_frame_without_leaking_image_data(caplog: pytest.LogCaptureFixture) -> None:
    """Encode each current BGR frame, including after a missing frame, without leaking image data."""
    red_frame = np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8)
    blue_frame = np.full((24, 32, 3), (255, 0, 0), dtype=np.uint8)
    red_frame.setflags(write=False)
    blue_frame.setflags(write=False)
    capture = MagicMock(side_effect=[red_frame, None, blue_frame])
    image_sender = AsyncMock()
    robot = SimpleNamespace(media=SimpleNamespace(get_frame=capture))
    dependencies = SimpleNamespace(reachy_mini=robot, camera_enabled=True, send_image=image_sender)
    arguments = json.dumps({"question": "What is the user holding?"})

    with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.tools.camera"):
        for index, expected_rgb in enumerate(((255, 0, 0), None, (0, 0, 255))):
            image_sender.reset_mock()
            result = await camera.on_invoke_tool(
                ToolContext(
                    dependencies,
                    tool_name="camera",
                    tool_call_id=f"camera-call-{index}",
                    tool_arguments=arguments,
                ),
                arguments,
            )
            if expected_rgb is None:
                assert result == {"error": "No frame available"}
                image_sender.assert_not_awaited()
                continue

            assert result == {"status": "image submitted", "question": "What is the user holding?"}
            image_sender.assert_awaited_once()
            question, jpeg_bytes = image_sender.await_args.args
            assert question == "What is the user holding?"
            with Image.open(BytesIO(jpeg_bytes)) as image:
                assert image.format == "JPEG"
                assert image.size == (32, 24)
                np.testing.assert_allclose(np.asarray(image).mean(axis=(0, 1)), expected_rgb, atol=3)
            assert f"Submitted camera frame: jpeg_bytes={len(jpeg_bytes)}" in caplog.messages
            assert "base64" not in str(result)

    assert capture.call_count == 3
    assert "What is the user holding?" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["capture", "encode"])
async def test_camera_reports_capture_and_encoding_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """Log failures and return a tool error without submitting an image."""
    capture = MagicMock(return_value=np.zeros((24, 32, 3), dtype=np.uint8))
    if failure == "capture":
        error = RuntimeError("Camera disconnected")
        capture.side_effect = error
    else:
        error = OSError("JPEG encoding failed")
        monkeypatch.setattr(Image.Image, "save", MagicMock(side_effect=error))
    image_sender = AsyncMock()
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)),
        camera_enabled=True,
        send_image=image_sender,
    )
    arguments = json.dumps({"question": "What is the user holding?"})

    result = await camera.on_invoke_tool(
        ToolContext(dependencies, tool_name="camera", tool_call_id="camera-call", tool_arguments=arguments),
        arguments,
    )

    assert result == {"error": f"Camera capture failed: {type(error).__name__}: {error}"}
    assert "Camera capture failed" in caplog.text
    image_sender.assert_not_awaited()


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

    assert [tool.name for tool in get_function_tools(["camera", "web_search", "wait_for_user"])] == [
        "camera",
        "wait_for_user",
        "web_search",
    ]
