import base64
import logging
from io import BytesIO

from PIL import Image
from agents import FunctionTool, ToolOutputImage, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="camera",
    description_override=(
        "Capture the current camera view when the user asks what you see, asks about their appearance, "
        "or wants you to inspect something in front of the robot."
    ),
)
async def camera_tool(context: RunContextWrapper[ToolDependencies]) -> ToolOutputImage | ToolResult:
    """Return the current camera frame to the delegated backend."""
    dependencies = context.context
    if not dependencies.camera_enabled:
        return {"error": "Camera is disabled"}
    try:
        frame = dependencies.reachy_mini.media.get_frame()
        if frame is None:
            logger.warning("No camera frame available")
            return {"error": "No frame available"}
        with Image.fromarray(frame[:, :, ::-1]) as image, BytesIO() as image_buffer:
            image.thumbnail((512, 512))
            image.save(image_buffer, format="WEBP", quality=85)
            image_bytes = image_buffer.getvalue()
        # Leave room for base64 expansion and context in Live's 32 KiB input buffer.
        if len(image_bytes) > 20 * 1024:
            logger.warning("Camera image is too large: image_bytes=%d", len(image_bytes))
            return {"error": "Camera image is too large to inspect"}
        image_url = f"data:image/webp;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        logger.info("Captured camera frame: image_bytes=%d", len(image_bytes))
        return ToolOutputImage(image_url=image_url, detail="high")
    except Exception as error:
        logger.exception("Camera inspection failed")
        return {"error": f"Camera inspection failed: {type(error).__name__}: {error}"}


camera: FunctionTool = camera_tool
