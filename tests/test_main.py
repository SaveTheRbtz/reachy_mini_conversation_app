import logging
import argparse
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.main as main_module
from reachy_mini_conversation_app.memory import MemorySnapshot


@pytest.mark.parametrize(("no_camera", "expected_events"), [(False, ["camera", "launch"]), (True, ["launch"])])
def test_run_prewarms_camera_once_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    no_camera: bool,
    expected_events: list[str],
) -> None:
    """Warm the JPEG pipeline once before launch unless camera use is disabled."""
    events: list[str] = []
    robot = MagicMock()
    robot.media.get_frame_jpeg.side_effect = lambda: events.append("camera")
    robot.media.get_output_audio_samplerate.return_value = 48_000
    stream = MagicMock()
    stream.launch.side_effect = lambda: events.append("launch")

    monkeypatch.setattr(main_module, "setup_logger", MagicMock(return_value=logging.getLogger(main_module.__name__)))
    monkeypatch.setattr(main_module.config, "REACHY_MINI_CUSTOM_PROFILE", "curious_kids_ru")
    monkeypatch.setattr(
        main_module, "load_memory", MagicMock(return_value=MemorySnapshot(memories=["Private household memory"]))
    )
    monkeypatch.setattr(main_module.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(main_module, "MovementManager", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_module, "get_session_voice", MagicMock(return_value="coral"))
    monkeypatch.setattr(main_module, "LocalStream", MagicMock(return_value=stream))
    monkeypatch.setattr(main_module, "resolve_app_timeout_minutes", MagicMock(return_value=None))

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        main_module.run(argparse.Namespace(debug=False, no_camera=no_camera, ui=False), robot=robot)

    assert events == expected_events
    assert (
        "Conversation configuration: profile=curious_kids_ru voice=coral "
        f"camera_enabled={not no_camera} memories=1 output_sample_rate=48000 Hz"
    ) in caplog.messages
    assert "Private household memory" not in caplog.text
