import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from reachy_mini_conversation_app.memory import MAX_MEMORY_BYTES, MemorySnapshot, load_memory, save_memory
from reachy_mini_conversation_app.prompts import get_backend_instructions, get_session_instructions
from reachy_mini_conversation_app.tools.manage_memory import manage_memory


def _tool_context(dependencies: object, arguments: str) -> ToolContext:
    return ToolContext(
        dependencies,
        tool_name="manage_memory",
        tool_call_id="manage-memory-call",
        tool_arguments=arguments,
    )


@pytest.mark.asyncio
async def test_manage_memory_replaces_loaded_snapshot_and_persists(tmp_path: Path) -> None:
    """Persist a complete typed replacement supplied by the conversation backend."""
    save_memory(MemorySnapshot(memories=["Любит книги о Земле."]), tmp_path)
    current = load_memory(tmp_path)
    replacement = MemorySnapshot(memories=["Любит книги о космосе."])
    dependencies = SimpleNamespace(memory=current, instance_path=tmp_path)
    arguments = json.dumps({"replacement": replacement.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result == {"status": "updated"}
    assert dependencies.memory == replacement
    assert load_memory(tmp_path) == replacement
    assert not (tmp_path / "memory.json.tmp").exists()
    schema = MemorySnapshot.model_json_schema()
    assert "shared household" in schema["description"]
    assert "full replacement list" in schema["properties"]["memories"]["description"]


@pytest.mark.asyncio
async def test_manage_memory_reports_unchanged_snapshot_as_not_saved(tmp_path: Path) -> None:
    """Never present an unchanged replacement as a successful update."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    original_bytes = (tmp_path / "memory.json").read_bytes()
    dependencies = SimpleNamespace(memory=original, instance_path=tmp_path)
    arguments = json.dumps({"replacement": original.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result == {
        "status": "unchanged",
        "message": "No memory was changed. Do not say that anything was remembered or forgotten.",
    }
    assert dependencies.memory is original
    assert (tmp_path / "memory.json").read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["oversized", "save"])
async def test_manage_memory_failure_preserves_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Keep both memory copies unchanged when validation or persistence fails."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    original_bytes = (tmp_path / "memory.json").read_bytes()
    dependencies = SimpleNamespace(memory=original, instance_path=tmp_path)
    if failure == "oversized":
        replacement = MemorySnapshot(memories=["x" * MAX_MEMORY_BYTES])
    else:
        replacement = MemorySnapshot(memories=["Любит го."])

        def fail_replace(_temporary_path: Path, _target: Path) -> Path:
            raise OSError("disk unavailable")

        monkeypatch.setattr(Path, "replace", fail_replace)
    arguments = json.dumps({"replacement": replacement.model_dump()}, ensure_ascii=False)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result == {"error": "Memory could not be changed. Do not say that anything was remembered or forgotten."}
    assert dependencies.memory is original
    assert dependencies.memory.memories == ["Любит шахматы."]
    assert (tmp_path / "memory.json").read_bytes() == original_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [{"memories": "not a list"}, {"memories": [], "unexpected": True}])
async def test_manage_memory_rejects_invalid_replacement(tmp_path: Path, replacement: dict[str, object]) -> None:
    """Invalid tool arguments must leave household memory untouched."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    dependencies = SimpleNamespace(memory=original, instance_path=tmp_path)
    arguments = json.dumps({"replacement": replacement})

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result != {"status": "updated"}
    assert dependencies.memory is original
    assert load_memory(tmp_path) == original


def test_backend_instructions_inject_current_memory_as_untrusted_context() -> None:
    """Keep mutable household memory in the backend rather than immutable voice instructions."""
    dependencies = SimpleNamespace(
        memory=MemorySnapshot(memories=["Кто-то в семье любит книги о космосе."]),
    )

    instructions = get_backend_instructions(dependencies)

    assert "<shared_household_memory>" in instructions
    assert '"memories"' in instructions
    assert "Кто-то в семье любит книги о космосе." in instructions
    assert "untrusted background context" in instructions
    assert "current request and current conversation always take precedence" in instructions
    assert "Do not infer who a memory describes" in instructions
    assert 'only when its result has status "updated"' in instructions
    assert "never imply success" in instructions
    voice_instructions = get_session_instructions(())
    assert "<shared_household_memory>" not in voice_instructions
    assert "Кто-то в семье любит книги о космосе." not in voice_instructions
    assert "asks what you remember" in voice_instructions
    dependencies.memory = MemorySnapshot(memories=[])
    assert "Кто-то в семье любит книги о космосе." not in get_backend_instructions(dependencies)
    assert "memory_id" not in instructions


def test_live_prompt_exposes_only_enabled_backend_capabilities() -> None:
    """A restricted profile should not promise disabled tools to the voice model."""
    instructions = get_session_instructions(("camera", "move_head"))

    assert "- camera:" in instructions
    assert "- move_head:" in instructions
    assert "- web_search:" not in instructions
    assert "- manage_memory:" not in instructions
    assert "wait_for_user" not in instructions
    assert "Delegate before giving an answer" in instructions
    assert "without giving the final answer" in instructions
