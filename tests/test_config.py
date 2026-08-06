"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", 15.0),  # unset/blank falls back to the default
        ("soon", 15.0),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected
