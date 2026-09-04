import logging
from io import BytesIO

from PIL import Image
from agents import FunctionTool, RunContextWrapper, function_tool

from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies


logger = logging.getLogger(__name__)


@function_tool(
    name_override="camera",
    description_override=(
        "Capture the current camera view when the user asks what you see, asks about their appearance, "
        "or wants you to inspect something in front of the robot."
    ),
)
async def camera_tool(context: RunContextWrapper[ToolDependencies], question: str) -> ToolResult:
    """Capture and submit the current camera frame."""
    if not question.strip():
        return {"error": "question must be a non-empty string"}
    dependencies = context.context
    if not dependencies.camera_enabled:
        return {"error": "Camera is disabled"}
    if dependencies.send_image is None:
        return {"error": "Camera input is unavailable before the realtime session starts"}
    try:
        frame = dependencies.reachy_mini.media.get_frame()
        if frame is None:
            logger.warning("No camera frame available")
            return {"error": "No frame available"}
        with Image.fromarray(frame[:, :, ::-1]) as image, BytesIO() as jpeg_buffer:
            image.save(jpeg_buffer, format="JPEG", quality=85)
            jpeg_bytes = jpeg_buffer.getvalue()
        await dependencies.send_image(question.strip(), jpeg_bytes)
        logger.info("Submitted camera frame: jpeg_bytes=%d", len(jpeg_bytes))
        return {"status": "image submitted", "question": question.strip()}
    except Exception as error:
        logger.exception("Camera capture failed")
        return {"error": f"Camera capture failed: {type(error).__name__}: {error}"}


camera: FunctionTool = camera_tool
