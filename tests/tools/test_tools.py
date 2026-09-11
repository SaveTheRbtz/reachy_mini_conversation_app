import json
import base64
import logging
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from agents import ToolOutputImage
from agents.tool_context import ToolContext

import reachy_mini_conversation_app.tools.camera as camera_module
from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, available_tool_catalog
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.head_tracking import head_tracking


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_shape, expected_size", [((480, 640), (512, 384)), ((24, 32), (32, 24))])
async def test_camera_returns_current_resized_frame_without_logging_image_data(
    caplog: pytest.LogCaptureFixture, frame_shape: tuple[int, int], expected_size: tuple[int, int]
) -> None:
    """Preserve each BGR frame's colors and aspect ratio without upscaling or logging it."""
    red_frame = np.full((*frame_shape, 3), (0, 0, 255), dtype=np.uint8)
    blue_frame = np.full((*frame_shape, 3), (255, 0, 0), dtype=np.uint8)
    red_frame.setflags(write=False)
    blue_frame.setflags(write=False)
    capture = MagicMock(side_effect=[red_frame, None, blue_frame])
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)), camera_enabled=True
    )
    arguments = "{}"
    with caplog.at_level(logging.INFO, logger=camera_module.__name__):
        for index, expected_rgb in enumerate(((255, 0, 0), None, (0, 0, 255))):
            result = await camera.on_invoke_tool(
                ToolContext(
                    dependencies, tool_name="camera", tool_call_id=f"camera-{index}", tool_arguments=arguments
                ),
                arguments,
            )
            if expected_rgb is None:
                assert result == {"error": "No frame available"}
                continue
            assert isinstance(result, ToolOutputImage)
            assert result.detail == "high"
            assert result.image_url is not None
            assert result.image_url.startswith("data:image/webp;base64,")
            encoded = result.image_url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                assert image.format == "WEBP"
                assert image.size == expected_size
                np.testing.assert_allclose(np.asarray(image).mean(axis=(0, 1)), expected_rgb, atol=3)
            assert encoded not in caplog.text
    assert capture.call_count == 3
    assert "base64" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["capture", "encode"])
async def test_camera_reports_inspection_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """Return a tool error for capture or encoding failures."""
    capture = MagicMock(return_value=np.zeros((24, 32, 3), dtype=np.uint8))
    if failure == "capture":
        capture.side_effect = RuntimeError("Camera disconnected")
    else:
        monkeypatch.setattr(Image.Image, "save", MagicMock(side_effect=OSError("Image encoding failed")))
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)), camera_enabled=True
    )
    arguments = "{}"
    result = await camera.on_invoke_tool(
        ToolContext(dependencies, tool_name="camera", tool_call_id="camera-call", tool_arguments=arguments), arguments
    )
    assert "error" in result
    assert "Camera inspection failed" in result["error"]
    assert "Camera inspection failed" in caplog.text


@pytest.mark.asyncio
async def test_camera_rejects_oversized_frame_without_sending_it(caplog: pytest.LogCaptureFixture) -> None:
    """A detailed noisy image returns an error before it can exceed Live's input buffer."""
    frame = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=MagicMock(return_value=frame))),
        camera_enabled=True,
    )
    arguments = "{}"

    result = await camera.on_invoke_tool(
        ToolContext(dependencies, tool_name="camera", tool_call_id="camera-call", tool_arguments=arguments), arguments
    )

    assert result == {"error": "Camera image is too large to inspect"}
    assert "Camera image is too large" in caplog.text
    assert "base64" not in caplog.text


@pytest.mark.asyncio
async def test_disabled_camera_does_not_capture() -> None:
    """Respect camera availability without reading a frame."""
    capture = MagicMock()
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)), camera_enabled=False
    )
    arguments = "{}"

    result = await camera.on_invoke_tool(
        ToolContext(dependencies, tool_name="camera", tool_call_id="camera-call", tool_arguments=arguments), arguments
    )

    assert result == {"error": "Camera is disabled"}
    capture.assert_not_called()


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
    with pytest.raises(ValueError, match="Unknown profile tools: removed_tool"):
        available_tool_catalog(["web_search", "removed_tool"])

    assert [tool.name for tool in get_function_tools(["camera", "web_search", "move_head"])] == [
        "camera",
        "move_head",
    ]
