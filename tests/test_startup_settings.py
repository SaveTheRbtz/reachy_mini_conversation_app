from pathlib import Path

import pytest

import reachy_mini_conversation_app.startup_settings as settings_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    read_startup_settings,
    write_startup_settings,
    load_startup_settings_into_runtime,
)


@pytest.mark.parametrize("standalone", [False, True], ids=["instance", "standalone"])
def test_settings_persist_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, standalone: bool) -> None:
    """Both launch modes persist preferences and restore defaults when cleared."""
    monkeypatch.chdir(tmp_path)
    instance = None if standalone else tmp_path
    write_startup_settings(instance, profile="guide", voice="coral")
    assert read_startup_settings(instance) == StartupSettings(profile="guide", voice="coral")

    write_startup_settings(instance, profile=None, voice=None)
    assert read_startup_settings(instance) == StartupSettings()


@pytest.mark.parametrize(
    ("saved", "environment_profile", "expected"),
    [
        (True, None, "saved_guide"),
        (True, "environment_guide", "saved_guide"),
        (False, "environment_guide", "environment_guide"),
        (False, None, None),
    ],
    ids=["saved", "saved-overrides-environment", "environment", "default"],
)
def test_startup_applies_profile_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved: bool,
    environment_profile: str | None,
    expected: str | None,
) -> None:
    """Persisted preferences win, followed by the environment and bundled default."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", environment_profile)
    if environment_profile is None:
        monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    else:
        monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", environment_profile)
    if saved:
        write_startup_settings(tmp_path, profile="saved_guide", voice="coral")

    settings = load_startup_settings_into_runtime(tmp_path)

    assert config.REACHY_MINI_CUSTOM_PROFILE == expected
    assert settings.voice == ("coral" if saved else None)


def test_locked_profile_ignores_saved_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed launch profile cannot be replaced by instance preferences."""
    monkeypatch.setattr(settings_module, "LOCKED_PROFILE", "locked_guide")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "locked_guide")
    write_startup_settings(tmp_path, profile="saved_guide", voice="coral")

    assert load_startup_settings_into_runtime(tmp_path) == StartupSettings()
    assert config.REACHY_MINI_CUSTOM_PROFILE == "locked_guide"


@pytest.mark.parametrize(
    "document", ['{"profile":', '["guide"]', b"\xff"], ids=["invalid-json", "wrong-shape", "invalid-utf8"]
)
def test_unreadable_settings_fall_back_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, document: str | bytes
) -> None:
    """Damaged local preferences must not prevent the application from starting."""
    (tmp_path / "startup_settings.json").write_bytes(document.encode() if isinstance(document, str) else document)

    assert read_startup_settings(tmp_path) == StartupSettings()
    assert any(record.levelname == "WARNING" for record in caplog.records)
