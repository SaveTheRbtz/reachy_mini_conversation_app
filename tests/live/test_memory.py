from pathlib import Path

import pytest

from tests.live.session import live_session, wait_for_content
from reachy_mini_conversation_app.memory import MemorySnapshot, load_memory, save_memory


pytestmark = [pytest.mark.live, pytest.mark.asyncio, pytest.mark.enable_socket]


@pytest.mark.parametrize(
    ("statement", "retained", "removed"),
    [
        (
            "Remember that Nora enjoys gardening.",
            ("maya", "astronomy", "leo", "chess", "nora", "gardening"),
            (),
        ),
        (
            "Correct Leo's preference: he now prefers Go instead of chess. Remember that.",
            ("maya", "astronomy", "leo", "go"),
            ("leo prefers chess",),
        ),
        ("Please forget everything saved about Leo.", ("maya", "astronomy"), ("leo", "chess")),
        ("Please forget all saved household memories.", (), ("maya", "astronomy", "leo", "chess")),
    ],
    ids=["preserve-and-add", "correct", "forget-one", "forget-all"],
)
async def test_live_backend_replaces_memory_without_losing_unrelated_facts(
    tmp_path: Path, statement: str, retained: tuple[str, ...], removed: tuple[str, ...]
) -> None:
    """Apply real backend memory updates through the production Live tool and persistence path."""
    original = MemorySnapshot(memories=["Maya enjoys astronomy.", "Leo prefers chess."])
    save_memory(original, tmp_path)

    async with live_session(tmp_path) as conversation:
        await conversation.say(f"{statement} Use manage_memory and confirm only after saving.")
        await wait_for_content(
            conversation,
            lambda text: conversation.dependencies.memory != original and conversation.backend_completions >= 2,
        )
        replacement = conversation.dependencies.memory

    memories = " ".join(replacement.memories).casefold()
    assert all(fragment in memories for fragment in retained)
    assert not any(fragment in memories for fragment in removed)
    assert load_memory(tmp_path) == replacement
    if not retained:
        assert replacement.memories == []
