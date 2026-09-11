import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep developer configuration out of test collection and restore it afterwards."""
    environment = pytest.MonkeyPatch()
    environment.setenv("REACHY_MINI_SKIP_DOTENV", "1")
    for name in ("REACHY_MINI_CUSTOM_PROFILE", "OPENAI_VOICE", "REACHY_MINI_APP_TIMEOUT_MINUTES"):
        environment.delenv(name, raising=False)
    config.add_cleanup(environment.undo)
