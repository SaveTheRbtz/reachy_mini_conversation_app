import os
import logging
from pathlib import Path

from pydantic import Field, BaseModel, ConfigDict


logger = logging.getLogger(__name__)

MAX_MEMORY_BYTES = 32 * 1024
MEMORY_FILENAME = "memory.json"


class MemorySnapshot(BaseModel):
    """Complete replacement snapshot of durable, non-sensitive shared household memories."""

    model_config = ConfigDict(extra="forbid")

    memories: list[str] = Field(
        description=(
            "Every retained household memory as a concise standalone statement; include the full replacement list, "
            "not only memories changed by the latest user statement."
        )
    )


def memory_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the shared memory path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / MEMORY_FILENAME
    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "reachy_mini_conversation_app" / MEMORY_FILENAME


def load_memory(instance_path: str | Path | None = None) -> MemorySnapshot:
    """Load the shared household memory snapshot."""
    path = memory_path_for_instance(instance_path)
    try:
        serialized = path.read_bytes()
        if len(serialized) > MAX_MEMORY_BYTES:
            raise ValueError(f"memory snapshot exceeds {MAX_MEMORY_BYTES} bytes")
        return MemorySnapshot.model_validate_json(serialized)
    except FileNotFoundError:
        return MemorySnapshot(memories=[])
    except (OSError, ValueError) as error:
        logger.warning("Failed to load memory snapshot from %s: %s", path, error)
        return MemorySnapshot(memories=[])


def save_memory(snapshot: MemorySnapshot, instance_path: str | Path | None = None) -> None:
    """Atomically replace the shared household memory snapshot."""
    serialized = f"{snapshot.model_dump_json(indent=2)}\n".encode()
    if len(serialized) > MAX_MEMORY_BYTES:
        raise ValueError(f"memory snapshot exceeds {MAX_MEMORY_BYTES} bytes")

    path = memory_path_for_instance(instance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(serialized)
    temporary_path.replace(path)
