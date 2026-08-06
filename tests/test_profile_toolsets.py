"""Tests for instance-local personality tool selections."""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.profile_store as profile_store_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_toolsets,
    read_profile_tool_names,
    get_profile_toolsets_path,
    read_profile_tool_override,
    clear_profile_tool_override,
    write_profile_tool_override,
)


TOOL_NAME = "pollen_robotics_reachy_mini_search_tool__search_web"


@pytest.fixture
def configured_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configure two strict profile documents and return their instance path."""
    instance_path = tmp_path / "instance"
    profiles_root = tmp_path / "profiles"
    write_profile("default", profiles_root / "default", "Default profile.", ["dance", TOOL_NAME])
    write_profile(
        "guide",
        profiles_root / "guide",
        "Guide profile.",
        ["camera", TOOL_NAME, "wait_for_user"],
    )
    monkeypatch.setattr(config, "INSTANCE_PATH", instance_path)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles_root)
    monkeypatch.setattr(profile_store_mod, "DEFAULT_PROFILES_DIRECTORY", profiles_root)
    return instance_path


def test_profile_tool_override_round_trip_and_reset(configured_profiles: Path) -> None:
    """An explicit override should replace authored defaults until it is reset."""
    instance_path = configured_profiles
    assert read_profile_tool_names("guide", instance_path) == ["camera", TOOL_NAME, "wait_for_user"]
    assert read_profile_tool_override("guide", instance_path) is None

    settings_path = write_profile_tool_override(
        "guide",
        [" camera ", "# disabled", "", "camera", "wait_for_user"],
        instance_path,
    )

    assert settings_path == get_profile_toolsets_path(instance_path)
    assert read_profile_tool_override("guide", instance_path) == ["camera", "wait_for_user"]
    assert read_profile_tool_names("guide", instance_path) == ["camera", "wait_for_user"]

    write_profile_tool_override("guide", [], instance_path)

    assert read_profile_tool_override("guide", instance_path) == []
    assert read_profile_tool_names("guide", instance_path) == []
    assert clear_profile_tool_override("guide", instance_path) is True
    assert read_profile_tool_override("guide", instance_path) is None
    assert read_profile_tool_names("guide", instance_path) == ["camera", TOOL_NAME, "wait_for_user"]
    assert not settings_path.exists()
    assert clear_profile_tool_override("guide", instance_path) is False


def test_default_profile_uses_canonical_storage_key(configured_profiles: Path) -> None:
    """An empty runtime selection should use the canonical default override."""
    instance_path = configured_profiles

    write_profile_tool_override(None, ["dance"], instance_path)

    assert read_profile_tool_override("default", instance_path) == ["dance"]
    assert read_profile_toolsets(instance_path).profiles == {"default": ["dance"]}
