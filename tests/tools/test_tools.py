import json
import base64
import logging
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image
from agents.tool_context import ToolContext

import reachy_mini_conversation_app.tools.camera as camera_module
from reachy_mini_conversation_app.tools.camera import camera
from reachy_mini_conversation_app.tools.core_tools import get_function_tools, available_tool_catalog
from reachy_mini_conversation_app.tools.go_to_sleep import go_to_sleep
from reachy_mini_conversation_app.tools.head_tracking import head_tracking


@pytest.fixture
def vision_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock only the vision API while retaining production camera encoding."""
    client = MagicMock()
    client.__aenter__.return_value = client
    client.responses.create = AsyncMock(return_value=SimpleNamespace(output_text="A visible object."))
    monkeypatch.setattr(camera_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(camera_module.config, "OPENAI_API_KEY", "test-key")
    return client


@pytest.mark.asyncio
async def test_camera_inspects_current_frame_without_leaking_image_data(
    vision_client: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Encode each current BGR frame without leaking images into tool results or logs."""
    red_frame = np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8)
    blue_frame = np.full((24, 32, 3), (255, 0, 0), dtype=np.uint8)
    red_frame.setflags(write=False)
    blue_frame.setflags(write=False)
    capture = MagicMock(side_effect=[red_frame, None, blue_frame])
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)), camera_enabled=True
    )
    arguments = json.dumps({"question": "What is the user holding?"})
    with caplog.at_level(logging.INFO, logger=camera_module.__name__):
        for index, expected_rgb in enumerate(((255, 0, 0), None, (0, 0, 255))):
            vision_client.responses.create.reset_mock()
            result = await camera.on_invoke_tool(
                ToolContext(
                    dependencies, tool_name="camera", tool_call_id=f"camera-{index}", tool_arguments=arguments
                ),
                arguments,
            )
            if expected_rgb is None:
                assert result == {"error": "No frame available"}
                vision_client.responses.create.assert_not_awaited()
                continue
            assert result == {"description": "A visible object."}
            request = vision_client.responses.create.await_args.kwargs
            assert request["model"] == "gpt-6-astra"
            assert request["reasoning"] == {"effort": "low"}
            assert request["store"] is False
            content = request["input"][0]["content"]
            assert content[0] == {"type": "input_text", "text": "What is the user holding?"}
            assert content[1]["detail"] == "high"
            encoded = content[1]["image_url"].split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                assert image.format == "JPEG"
                assert image.size == (32, 24)
                np.testing.assert_allclose(np.asarray(image).mean(axis=(0, 1)), expected_rgb, atol=3)
            assert encoded not in caplog.text
            assert "base64" not in str(result)
    assert capture.call_count == 3
    assert "What is the user holding?" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["capture", "encode", "vision", "empty"])
async def test_camera_reports_inspection_errors(
    monkeypatch: pytest.MonkeyPatch, vision_client: MagicMock, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """Return a tool error for capture, encoding, or vision failures."""
    capture = MagicMock(return_value=np.zeros((24, 32, 3), dtype=np.uint8))
    if failure == "capture":
        capture.side_effect = RuntimeError("Camera disconnected")
    elif failure == "encode":
        monkeypatch.setattr(Image.Image, "save", MagicMock(side_effect=OSError("JPEG encoding failed")))
    elif failure == "vision":
        vision_client.responses.create.side_effect = RuntimeError("Vision unavailable")
    else:
        vision_client.responses.create.return_value = SimpleNamespace(output_text=" ")
    dependencies = SimpleNamespace(
        reachy_mini=SimpleNamespace(media=SimpleNamespace(get_frame=capture)), camera_enabled=True
    )
    arguments = json.dumps({"question": "What is visible?"})
    result = await camera.on_invoke_tool(
        ToolContext(dependencies, tool_name="camera", tool_call_id="camera-call", tool_arguments=arguments), arguments
    )
    assert "error" in result
    assert "Camera inspection failed" in result["error"]
    assert "Camera inspection failed" in caplog.text
    if failure in {"capture", "encode"}:
        vision_client.responses.create.assert_not_awaited()


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
