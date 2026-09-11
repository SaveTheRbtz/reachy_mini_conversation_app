import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini import ReachyMini
import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.prompts as prompts_mod
import reachy_mini_conversation_app.profile_store as profile_store_mod
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.profile_store import write_profile, read_profile_from_directory


def test_prompts_load_from_compact_builtin_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt loading should read compact built-in profile instructions directly."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "mad_scientist_assistant")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)

    expected = read_profile_from_directory(
        "mad_scientist_assistant",
        DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant",
    ).instructions

    instructions = prompts_mod.get_profile_instructions()
    assert instructions.startswith("# Personality")
    assert instructions.endswith(expected)


def test_default_session_instructions_load_from_default_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-profile session prompt should come from the built-in default profile."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    expected = read_profile_from_directory("default", DEFAULT_PROFILES_DIRECTORY / "default").instructions

    instructions = prompts_mod.get_profile_instructions()
    assert instructions.startswith("# Personality")
    assert instructions.endswith(expected)


def test_bracketed_prompt_line_stays_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bracketed prompt text should not be treated as an include."""
    profile_dir = tmp_path / "literal_prompt"
    write_profile("literal_prompt", profile_dir, "[custom_prompt]\n\nStay extra brief.", [])

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "literal_prompt")

    assert prompts_mod.get_profile_instructions().endswith("[custom_prompt]\n\nStay extra brief.")


def test_session_instructions_fall_back_to_default_for_incomplete_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete selected profiles should not stop session startup."""
    profile_dir = tmp_path / "incomplete_prompt"
    profile_dir.mkdir()

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "incomplete_prompt")

    expected = read_profile_from_directory("default", DEFAULT_PROFILES_DIRECTORY / "default").instructions

    assert prompts_mod.get_profile_instructions().endswith(expected)


def test_explicit_default_profile_does_not_fall_back_to_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken explicit default profile should fail without a self-fallback log."""
    default_dir = tmp_path / "default"
    default_dir.mkdir()

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(profile_store_mod, "DEFAULT_PROFILES_DIRECTORY", tmp_path)

    with caplog.at_level(logging.WARNING, logger="reachy_mini_conversation_app.prompts"):
        with pytest.raises(RuntimeError, match="Default profile has no usable instructions"):
            prompts_mod.get_profile_instructions()

    assert "Using default profile instructions" not in caplog.text


def test_session_voice_defaults_to_openai_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session voice should fall back to the OpenAI default voice."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.delenv(config_mod.OPENAI_VOICE_ENV, raising=False)

    assert prompts_mod.get_session_voice() == "gleam"


def test_session_greeting_prompt_loads_from_selected_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile metadata should steer the startup greeting prompt."""
    profile_dir = tmp_path / "friendly"
    write_profile(
        "friendly",
        profile_dir,
        "test instructions",
        [],
        greeting="Greet me like a tiny stage host.",
    )

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "friendly")

    assert "Greet me like a tiny stage host." in prompts_mod.get_session_greeting_prompt()


def test_session_greeting_prompt_uses_builtin_default_without_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-profile greeting should come from the built-in constant only."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert prompts_mod.get_session_greeting_prompt() == prompts_mod.DEFAULT_GREETING_PROMPT


def test_backend_memory_updates_without_changing_voice_instructions() -> None:
    """Only backend instructions receive the current serialized household memory."""
    dependencies = ToolDependencies(
        reachy_mini=MagicMock(spec=ReachyMini),
        movement_manager=MagicMock(spec=MovementManager),
        memory=MemorySnapshot(memories=["Кто-то в семье любит книги о космосе."]),
    )
    instructions = prompts_mod.get_backend_instructions(dependencies)
    assert "<shared_household_memory>" in instructions
    serialized = instructions.split("<shared_household_memory>\n", 1)[1].split("\n</shared_household_memory>", 1)[0]
    assert MemorySnapshot.model_validate_json(serialized) == dependencies.memory
    voice_instructions = prompts_mod.get_session_instructions(())
    assert "<shared_household_memory>" not in voice_instructions
    assert "Кто-то в семье любит книги о космосе." not in voice_instructions

    dependencies.memory = MemorySnapshot(memories=[])

    assert "Кто-то в семье любит книги о космосе." not in prompts_mod.get_backend_instructions(dependencies)
    assert prompts_mod.get_session_instructions(()) == voice_instructions


@pytest.mark.parametrize("enabled", [(), ("camera", "move_head"), ("web_search", "manage_memory")])
def test_voice_prompt_exposes_only_selected_backend_tools(enabled: tuple[str, ...]) -> None:
    """A profile's enabled tool catalog determines which capabilities the voice can promise."""
    instructions = prompts_mod.get_session_instructions(enabled)

    for name in ("camera", "move_head", "web_search", "manage_memory"):
        assert (f"- {name}:" in instructions) is (name in enabled)
