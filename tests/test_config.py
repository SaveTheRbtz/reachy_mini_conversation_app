"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("", "meridian"),
        ("quartz", "quartz"),
        ("gleam", "gleam"),
        ("unsupported", "meridian"),
    ],
)
def test_get_default_voice(monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: str) -> None:
    """The default voice should accept only supported environment overrides."""
    monkeypatch.setenv(config.OPENAI_VOICE_ENV, raw_value)

    assert config.get_default_voice() == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", 15.0),
        ("soon", 15.0),
        ("0", None),
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: float | None) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected
