import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import RunContextWrapper
from openai import OpenAIError
from agents.tool_context import ToolContext

import reachy_mini_conversation_app.tools.manage_memory as manage_memory_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import MAX_MEMORY_BYTES, MemorySnapshot, load_memory, save_memory
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.manage_memory import MEMORY_MODEL, manage_memory


def _tool_context(dependencies: object, arguments: str) -> ToolContext:
    return ToolContext(
        dependencies,
        tool_name="manage_memory",
        tool_call_id="manage-memory-call",
        tool_arguments=arguments,
    )


def _mock_memory_response(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot: MemorySnapshot | None = None,
    error: Exception | None = None,
) -> tuple[MagicMock, AsyncMock]:
    parse = AsyncMock(side_effect=error)
    if error is None:
        parse.return_value = SimpleNamespace(output_parsed=snapshot)
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=client)
    client_context.__aexit__ = AsyncMock(return_value=None)
    constructor = MagicMock(return_value=client_context)
    monkeypatch.setattr(manage_memory_module, "AsyncOpenAI", constructor)
    return constructor, parse


@pytest.mark.asyncio
async def test_manage_memory_replaces_loaded_snapshot_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the complete snapshot through one typed Responses call."""
    save_memory(MemorySnapshot(memories=["Любит книги о Земле."]), tmp_path)
    current = load_memory(tmp_path)
    replacement = MemorySnapshot(memories=["Любит книги о космосе."])
    dependencies = SimpleNamespace(memory=current, instance_path=tmp_path)
    statement = "Теперь мне больше нравятся книги о космосе, а не о Земле."
    arguments = json.dumps({"user_statement": statement}, ensure_ascii=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    constructor, parse = _mock_memory_response(monkeypatch, snapshot=replacement)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert result == {"status": "updated"}
    assert dependencies.memory is replacement
    assert load_memory(tmp_path) == replacement
    assert not (tmp_path / "memory.json.tmp").exists()
    constructor.assert_called_once_with(api_key="test-key", max_retries=0)
    parse.assert_awaited_once()
    request = parse.await_args.kwargs
    assert request["model"] == MEMORY_MODEL
    assert request["reasoning"] == {"effort": "high"}
    assert request["store"] is False
    assert request["text_format"] is MemorySnapshot
    assert "tools" not in request
    assert "previous_response_id" not in request
    assert json.loads(request["input"]) == {
        "current_snapshot": {"memories": ["Любит книги о Земле."]},
        "user_statement": statement,
    }
    assert manage_memory.params_json_schema["required"] == ["user_statement"]
    assert "exact relevant wording" in manage_memory.params_json_schema["properties"]["user_statement"]["description"]
    assert "exact relevant user wording" in manage_memory.description
    schema = MemorySnapshot.model_json_schema()
    assert "shared household" in schema["description"]
    assert "full replacement list" in schema["properties"]["memories"]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["api", "oversized", "save"])
async def test_manage_memory_failure_preserves_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Keep both memory copies unchanged when reduction or persistence fails."""
    original = MemorySnapshot(memories=["Любит шахматы."])
    save_memory(original, tmp_path)
    original_bytes = (tmp_path / "memory.json").read_bytes()
    dependencies = SimpleNamespace(memory=original, instance_path=tmp_path)
    arguments = json.dumps({"user_statement": "Теперь любит го."}, ensure_ascii=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")

    if failure == "api":
        _mock_memory_response(monkeypatch, error=OpenAIError("offline"))
    elif failure == "oversized":
        _mock_memory_response(
            monkeypatch,
            snapshot=MemorySnapshot(memories=["x" * MAX_MEMORY_BYTES]),
        )
    else:
        _mock_memory_response(monkeypatch, snapshot=MemorySnapshot(memories=["Любит го."]))

        def fail_replace(_temporary_path: Path, _target: Path) -> Path:
            raise OSError("disk unavailable")

        monkeypatch.setattr(Path, "replace", fail_replace)

    result = await manage_memory.on_invoke_tool(_tool_context(dependencies, arguments), arguments)

    assert "error" in result
    assert dependencies.memory is original
    assert dependencies.memory.memories == ["Любит шахматы."]
    assert (tmp_path / "memory.json").read_bytes() == original_bytes


def test_session_instructions_inject_shared_memory_as_untrusted_context() -> None:
    """Inject the shared snapshot once as lower-priority untrusted context."""
    dependencies = SimpleNamespace(
        memory=MemorySnapshot(memories=["Кто-то в семье любит книги о космосе."]),
    )

    instructions = get_session_instructions(
        RunContextWrapper(dependencies),
        MagicMock(),
    )

    assert "<shared_household_memory>" in instructions
    assert '"memories"' in instructions
    assert "Кто-то в семье любит книги о космосе." in instructions
    assert "untrusted background context" in instructions
    assert "current request and current conversation always take precedence" in instructions
    assert "Do not infer who a memory describes" in instructions
    assert "memory_id" not in instructions
