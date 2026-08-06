from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.console import LocalStream


@pytest.mark.asyncio
async def test_voice_change_persists_and_requests_one_session_restart(tmp_path) -> None:
    """Apply an immutable session voice through the stream's restart owner."""
    conversation = SimpleNamespace(
        voice="marin",
        last_activity_time=0.0,
        connected=True,
        shutdown=AsyncMock(),
        set_clear_player=MagicMock(),
        set_activity_observer=MagicMock(),
    )
    robot = SimpleNamespace()
    stream = LocalStream(
        robot,
        conversation_factory=MagicMock(return_value=conversation),
        instance_path=tmp_path,
    )

    status = await stream.change_voice("coral")

    assert status == "Voice changed to coral; restarting the conversation."
    conversation.shutdown.assert_awaited_once_with()
    assert '"voice": "coral"' in (tmp_path / "startup_settings.json").read_text(encoding="utf-8")
