import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.memory import MAX_MEMORY_BYTES, MemorySnapshot, load_memory, save_memory
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.manage_memory import manage_memory


@pytest.fixture
def dependencies(tmp_path: Path) -> ToolDependencies:
    """Keep tool state and durable storage local to each scenario."""
    return ToolDependencies(
        reachy_mini=MagicMock(spec=ReachyMini),
        movement_manager=MagicMock(spec=MovementManager),
        memory=MemorySnapshot(memories=[]),
        instance_path=tmp_path,
    )


def _tool_context(dependencies: ToolDependencies, arguments: str) -> ToolContext[ToolDependencies]:
    return ToolContext(
        dependencies,
        tool_name="manage_memory",
        tool_call_id="manage-memory-call",
        tool_arguments=arguments,
    )


@pytest.mark.asyncio
async def test_manage_memory_replaces_loaded_snapshot_and_persists(
    tmp_path: Path, dependencies: ToolDependencies
) -> None:
    """Persist a complete typed replacement supplied by the conversation backend."""
    save_memory(MemorySnapshot(memories=["Любит книги о Земле."]), tmp_path)
    current = load_memory(tmp_path)
    replacement = MemorySnapshot(memories=["Любит книги о космосе."])
    dependencies.memory = current
    arguments = json.dumps({"replacement": replacement.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result == {"status": "updated"}
    assert dependencies.memory == replacement
    assert load_memory(tmp_path) == replacement
    assert not (tmp_path / "memory.json.tmp").exists()


@pytest.mark.asyncio
async def test_manage_memory_reports_unchanged_snapshot_as_not_saved(
    tmp_path: Path, dependencies: ToolDependencies
) -> None:
    """Never present an unchanged replacement as a successful update."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    original_bytes = (tmp_path / "memory.json").read_bytes()
    dependencies.memory = original
    arguments = json.dumps({"replacement": original.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result["status"] == "unchanged"
    assert result["message"]
    assert dependencies.memory is original
    assert (tmp_path / "memory.json").read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["oversized", "save"])
async def test_manage_memory_failure_preserves_snapshot(
    tmp_path: Path,
    dependencies: ToolDependencies,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Keep both memory copies unchanged when validation or persistence fails."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    original_bytes = (tmp_path / "memory.json").read_bytes()
    dependencies.memory = original
    if failure == "oversized":
        replacement = MemorySnapshot(memories=["x" * MAX_MEMORY_BYTES])
    else:
        replacement = MemorySnapshot(memories=["Любит го."])

        def fail_replace(_temporary_path: Path, _target: Path) -> Path:
            raise OSError("disk unavailable")

        monkeypatch.setattr(Path, "replace", fail_replace)
    arguments = json.dumps({"replacement": replacement.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert set(result) == {"error"}
    assert result["error"]
    assert dependencies.memory is original
    assert dependencies.memory.memories == ["Любит шахматы."]
    assert (tmp_path / "memory.json").read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [{"memories": "not a list"}, {"memories": [], "unexpected": True}])
async def test_manage_memory_rejects_invalid_replacement(
    tmp_path: Path, replacement: dict[str, object], dependencies: ToolDependencies
) -> None:
    """Invalid tool arguments must leave household memory untouched."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    dependencies.memory = original
    arguments = json.dumps({"replacement": replacement})

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert isinstance(result, str) and "error" in result.casefold()
    assert dependencies.memory is original
    assert load_memory(tmp_path) == original
