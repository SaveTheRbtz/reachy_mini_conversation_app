import os

import pytest


@pytest.fixture(autouse=True)
def require_live_credentials() -> None:
    """Fail explicitly selected paid evaluations before opening a session without credentials."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        pytest.fail("Set OPENAI_API_KEY to run paid evaluations selected with -m live")
