import argparse
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.main as main_module


@pytest.mark.parametrize(("no_camera", "expected_events"), [(False, ["camera", "launch"]), (True, ["launch"])])
def test_run_prewarms_camera_once_before_launch(
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(main_module, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_module.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(main_module, "MovementManager", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_module, "get_session_voice", MagicMock(return_value="coral"))
    monkeypatch.setattr(main_module, "LocalStream", MagicMock(return_value=stream))
    monkeypatch.setattr(main_module, "resolve_app_timeout_minutes", MagicMock(return_value=None))

    main_module.run(argparse.Namespace(debug=False, no_camera=no_camera, ui=False), robot=robot)

    assert events == expected_events
