import os
import logging
from typing import Final
from pathlib import Path
from importlib.resources import files

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

REALTIME_MODEL: Final = "gpt-realtime-2.1"
OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
APP_TIMEOUT_MINUTES_ENV: Final = "REACHY_MINI_APP_TIMEOUT_MINUTES"
DEFAULT_APP_TIMEOUT_MINUTES: Final = 1440.0
DEFAULT_VOICE: Final = "marin"
OPENAI_VOICES: Final = (
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
)
LOCKED_PROFILE: Final[str | None] = None
USER_PERSONALITIES_DIRNAME: Final = "user_personalities"
TERMINAL_USER_PERSONALITIES_DIRECTORY: Final = Path("external_content") / USER_PERSONALITIES_DIRNAME
PROJECT_ROOT: Final = Path(__file__).parents[2].resolve()


def _default_profiles_directory() -> Path:
    source_profiles = PROJECT_ROOT / "profiles"
    if (PROJECT_ROOT / "pyproject.toml").is_file() and source_profiles.is_dir():
        return source_profiles
    try:
        return Path(str(files("reachy_talk_data").joinpath("profiles")))
    except (ModuleNotFoundError, TypeError):
        return source_profiles


DEFAULT_PROFILES_DIRECTORY: Final = _default_profiles_directory()


def _load_project_env() -> None:
    if os.getenv("REACHY_MINI_SKIP_DOTENV", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)


_load_project_env()


class Config:
    """Mutable runtime configuration for profiles and the OpenAI key."""

    def __init__(self) -> None:
        """Load environment-backed runtime values."""
        self.OPENAI_API_KEY = os.getenv(OPENAI_API_KEY_ENV)
        self.PROFILES_DIRECTORY = DEFAULT_PROFILES_DIRECTORY
        self.INSTANCE_PATH: Path | None = None
        self.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    def user_personalities_root(self) -> Path:
        """Return the writable root for user-created profiles."""
        if self.INSTANCE_PATH is None:
            return TERMINAL_USER_PERSONALITIES_DIRECTORY
        return self.INSTANCE_PATH / USER_PERSONALITIES_DIRNAME

    def resolve_profile_dir(self, profile: str) -> Path:
        """Return the on-disk directory for a profile selection."""
        prefix, _, profile_name = profile.partition("/")
        if prefix == USER_PERSONALITIES_DIRNAME and profile_name:
            return self.user_personalities_root() / profile_name
        return self.PROFILES_DIRECTORY / profile


config = Config()


def refresh_runtime_config_from_env() -> None:
    """Refresh values that may be changed through the settings UI."""
    config.OPENAI_API_KEY = os.getenv(OPENAI_API_KEY_ENV)
    config.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")


def has_openai_api_key() -> bool:
    """Return whether an OpenAI API key is configured."""
    return bool((config.OPENAI_API_KEY or "").strip())


def get_available_voices() -> list[str]:
    """Return supported OpenAI Realtime voices."""
    return list(OPENAI_VOICES)


def get_default_voice() -> str:
    """Return the default OpenAI Realtime voice."""
    return DEFAULT_VOICE


def resolve_app_timeout_minutes() -> float | None:
    """Read the inactivity timeout in minutes; non-positive values disable it."""
    raw_value = os.getenv(APP_TIMEOUT_MINUTES_ENV, "").strip()
    if not raw_value:
        return DEFAULT_APP_TIMEOUT_MINUTES
    try:
        timeout_minutes = float(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default.", APP_TIMEOUT_MINUTES_ENV, raw_value)
        return DEFAULT_APP_TIMEOUT_MINUTES
    return timeout_minutes if timeout_minutes > 0 else None


def set_instance_path(instance_path: str | Path | None) -> None:
    """Record the app instance directory for writable state."""
    config.INSTANCE_PATH = Path(instance_path) if instance_path else None


def set_custom_profile(profile: str | None) -> None:
    """Update the selected profile in runtime configuration."""
    if LOCKED_PROFILE is not None:
        return
    if profile:
        os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
    else:
        os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    config.REACHY_MINI_CUSTOM_PROFILE = profile
