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
