from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.audio.startup_config import (
    AUDIO_STARTUP_CONFIG,
    WRITE_SETTLE_SECONDS,
    apply_audio_startup_config,
)


@pytest.mark.parametrize(
    "verify, write_settle_seconds",
    [(True, WRITE_SETTLE_SECONDS), (False, 0)],
    ids=["verified-defaults", "explicit-options"],
)
def test_startup_delegates_audio_configuration_to_sdk(verify: bool, write_settle_seconds: float) -> None:
    """Keep the SDK responsible for applying and verifying the device configuration."""
    apply_config = MagicMock(return_value=True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(apply_audio_config=apply_config)))

    assert apply_audio_startup_config(robot, verify=verify, write_settle_seconds=write_settle_seconds)
    apply_config.assert_called_once_with(
        AUDIO_STARTUP_CONFIG, verify=verify, write_settle_seconds=write_settle_seconds
    )


@pytest.mark.parametrize("audio", [None, object()], ids=["audio-unavailable", "sdk-api-unavailable"])
def test_startup_continues_when_audio_configuration_is_unavailable(
    audio: object, caplog: pytest.LogCaptureFixture
) -> None:
    """Unavailable device configuration must be observable without preventing startup."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    assert apply_audio_startup_config(robot) is False
    assert "Skipping Reachy audio startup config" in caplog.text


@pytest.mark.parametrize(
    "failure", [False, RuntimeError("audio board unavailable")], ids=["sdk-rejected", "sdk-error"]
)
def test_startup_continues_and_logs_audio_configuration_failure(
    failure: bool | RuntimeError, caplog: pytest.LogCaptureFixture
) -> None:
    """Device write failures are reported while the rest of the application can start."""
    apply_config = MagicMock(return_value=failure if isinstance(failure, bool) else None)
    if isinstance(failure, RuntimeError):
        apply_config.side_effect = failure
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(apply_audio_config=apply_config)))

    assert apply_audio_startup_config(robot) is False
    apply_config.assert_called_once()
    assert caplog.records and caplog.records[-1].levelname == "WARNING"
