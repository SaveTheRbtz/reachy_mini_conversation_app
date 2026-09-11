"""Regression coverage for deleting custom personalities."""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.personality as personality_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.personality import delete_personality
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
    write_profile_tool_override,
)


def test_delete_removes_user_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a user profile also removes its tool override."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    personality_mod.save_user_personality("doomed", "Be brief.")
    profile_dir = tmp_path / "user_personalities" / "doomed"
    write_profile_tool_override("user_personalities/doomed", ["dance"], tmp_path)
    assert profile_dir.is_dir()

    assert delete_personality("user_personalities/doomed") is True
    assert not profile_dir.exists()
    assert read_profile_tool_override("user_personalities/doomed", tmp_path) is None


def test_delete_refuses_builtin_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in profiles cannot be deleted."""
    builtin_dir = config.resolve_profile_dir("mad_scientist_assistant")
    assert builtin_dir.is_dir()

    assert delete_personality("mad_scientist_assistant") is False
    assert builtin_dir.is_dir()


def test_delete_refuses_path_outside_user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Profile deletion cannot escape the user profile root."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    victim = tmp_path / "user_personalities" / "outside_target"
    victim.mkdir(parents=True)

    assert delete_personality("user_personalities/../outside_target") is False
    assert victim.is_dir()
