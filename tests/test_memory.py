from pathlib import Path

import pytest

from reachy_mini_conversation_app.memory import MAX_MEMORY_BYTES, MemorySnapshot, load_memory, save_memory


@pytest.mark.parametrize(
    "memories",
    [[], ["Любит книги о Земле.", "家庭喜欢音乐。", "Enjoys 🪴 gardening."]],
    ids=["empty", "unicode-household"],
)
def test_snapshot_round_trip_creates_storage_and_replaces_previous_contents(
    tmp_path: Path, memories: list[str]
) -> None:
    """Reload the complete replacement from disk, including empty and Unicode snapshots."""
    instance = tmp_path / "new-instance"
    save_memory(MemorySnapshot(memories=["Old memory."]), instance)
    snapshot = MemorySnapshot(memories=memories)

    save_memory(snapshot, instance)

    assert load_memory(instance) == snapshot
    assert not (instance / "memory.json.tmp").exists()


def test_missing_snapshot_starts_empty_without_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A new installation has no household memories yet."""
    assert load_memory(tmp_path) == MemorySnapshot(memories=[])
    assert not caplog.records


@pytest.mark.parametrize(
    "contents",
    [b"{", b'{"memories": "not a list"}', b'{"memories": [], "unknown": true}', b"x" * (MAX_MEMORY_BYTES + 1)],
    ids=["invalid-json", "invalid-schema", "unknown-field", "oversized"],
)
def test_unusable_snapshot_starts_empty_and_logs_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, contents: bytes
) -> None:
    """Damaged stored state cannot prevent the robot from starting a conversation."""
    (tmp_path / "memory.json").write_bytes(contents)

    assert load_memory(tmp_path) == MemorySnapshot(memories=[])
    assert "Failed to load memory snapshot" in caplog.text
    assert (tmp_path / "memory.json").read_bytes() == contents


def test_unreadable_snapshot_starts_empty_and_logs_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Filesystem read failures degrade to empty memory without silent data loss."""
    (tmp_path / "memory.json").mkdir()

    assert load_memory(tmp_path) == MemorySnapshot(memories=[])
    assert "Failed to load memory snapshot" in caplog.text
    assert (tmp_path / "memory.json").is_dir()
