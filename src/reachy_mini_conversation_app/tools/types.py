from typing import Protocol
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.memory import MemoryState


ToolResult = dict[str, object]
ImageSender = Callable[[str, bytes], Awaitable[None]]


class SleepCallback(Protocol):
    """Put the robot to sleep and request application shutdown."""

    def __call__(self) -> ToolResult:
        """Execute the sleep lifecycle action."""
        ...


@dataclass
class ToolDependencies:
    """Runtime dependencies available to local agent tools."""

    reachy_mini: ReachyMini
    movement_manager: MovementManager
    memory: MemoryState
    instance_path: str | Path | None = None
    camera_enabled: bool = True
    motion_duration_s: float = 1.0
    go_to_sleep: SleepCallback | None = None
    send_image: ImageSender | None = None
