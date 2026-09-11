from pathlib import Path

import pytest

import reachy_mini_conversation_app.profile_store as profile_store_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_profile_instructions
from reachy_mini_conversation_app.personality import delete_personality, list_personalities, save_user_personality
from reachy_mini_conversation_app.profile_store import read_profile, write_profile
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_override, write_profile_tool_override


@pytest.fixture(autouse=True)
def profile_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep even rejected deletion attempts away from the bundled project files."""
    profiles = tmp_path / "profiles"
    write_profile("default", profiles / "default", "Default instructions.", ["dance"])
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(profile_store_module, "DEFAULT_PROFILES_DIRECTORY", profiles)


@pytest.mark.parametrize("standalone", [False, True], ids=["instance", "standalone"])
@pytest.mark.parametrize("greeting, voice", [("Hello there.", "coral"), ("", None)], ids=["authored", "inherited"])
def test_user_profile_round_trips_through_writable_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    standalone: bool,
    greeting: str,
    voice: str | None,
) -> None:
    """Created personalities preserve metadata and are usable by the conversation."""
    monkeypatch.chdir(tmp_path)
    if standalone:
        monkeypatch.setattr(config, "INSTANCE_PATH", None)
    selection = save_user_personality("guide", "Be calm.", greeting=greeting, voice=voice)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", selection)

    expected_root = tmp_path / "external_content" if standalone else tmp_path
    assert selection == "user_personalities/guide"
    assert config.resolve_profile_dir(selection).resolve() == expected_root / selection
    profile = read_profile(selection)
    assert profile.instructions == "Be calm."
    assert profile.greeting == (greeting or None)
    assert profile.voice == voice
    assert profile.default_tools == ("dance",)
    assert selection in list_personalities()
    assert get_profile_instructions().endswith("Be calm.")


def test_overwrite_preserves_authored_tools_and_user_selection(tmp_path: Path) -> None:
    """Editing prose must preserve authored defaults and the independent tool override."""
    selection = save_user_personality("guide", "Original.", default_tools=["camera"])
    write_profile_tool_override(selection, [], tmp_path)

    with pytest.raises(FileExistsError):
        save_user_personality("guide", "Rejected.")
    assert read_profile(selection).instructions == "Original."
    save_user_personality("guide", "Updated.", overwrite=True)

    assert read_profile(selection).instructions == "Updated."
    assert read_profile(selection).default_tools == ("camera",)
    assert read_profile_tool_override(selection, tmp_path) == []


def test_creation_rejects_unsafe_name(tmp_path: Path) -> None:
    """Headless callers must not escape the user-personality storage root."""
    with pytest.raises(ValueError):
        save_user_personality("../outside", "Unsafe.")
    assert not (tmp_path / "outside").exists()


def test_delete_removes_user_profile_and_override(tmp_path: Path) -> None:
    """Deleting a personality removes its stored content and independent tool selection."""
    selection = save_user_personality("doomed", "Be brief.")
    write_profile_tool_override(selection, ["dance"], tmp_path)

    assert delete_personality(selection) is True
    assert not config.resolve_profile_dir(selection).exists()
    assert read_profile_tool_override(selection, tmp_path) is None


def test_delete_refuses_builtin_profile() -> None:
    """The bundled default remains readable after a rejected deletion."""
    assert delete_personality("default") is False
    assert read_profile("default").instructions == "Default instructions."


def test_delete_refuses_path_outside_user_root(tmp_path: Path) -> None:
    """Traversal must preserve the actual directory that the supplied path resolves to."""
    victim = tmp_path / "outside_target"
    victim.mkdir()
    (victim / "important.txt").write_text("Keep me.", encoding="utf-8")

    assert delete_personality("user_personalities/../outside_target") is False
    assert (victim / "important.txt").read_text(encoding="utf-8") == "Keep me."


def test_listing_skips_hidden_and_invalid_profiles() -> None:
    """A damaged or hidden personality must not displace usable choices."""
    write_profile("visible", config.PROFILES_DIRECTORY / "visible", "Visible.", [])
    write_profile("hidden", config.PROFILES_DIRECTORY / "hidden", "Hidden.", [], hidden=True)
    invalid = config.PROFILES_DIRECTORY / "invalid"
    invalid.mkdir()
    (invalid / "profile.md").write_text("Invalid document.", encoding="utf-8")

    assert list_personalities() == ["default", "visible"]
