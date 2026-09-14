import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from tests.live.session import FIXTURES, REACHY_SAMPLE_RATE, live_session, wait_for_reply, wait_for_content
from reachy_mini_conversation_app.memory import load_memory


pytestmark = [pytest.mark.live, pytest.mark.asyncio, pytest.mark.enable_socket]


async def test_synthesized_speech_drives_production_audio_path(tmp_path: Path) -> None:
    """Receive audible speech while the microphone continues streaming silence."""
    async with live_session(tmp_path, FIXTURES / "hear_test.ogg") as conversation:
        await wait_for_reply(conversation)
        assert conversation.played_samples > REACHY_SAMPLE_RATE // 10


async def test_synthesized_speech_uses_camera_image(tmp_path: Path) -> None:
    """Answer repeated visual questions through native images without filling backend history."""
    with Image.open(FIXTURES / "blue_chair.jpg") as image:
        camera_frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
    async with live_session(tmp_path, FIXTURES / "camera_request.ogg", camera_frame) as conversation:
        robot = conversation.dependencies.reachy_mini
        assert isinstance(robot, MagicMock)
        await wait_for_content(conversation, lambda text: "blue" in text and "chair" in text)
        robot.media.get_frame.assert_called_once_with()
        for captures, (question, expected) in enumerate(
            [
                ("What color is the seat?", ("blue",)),
                ("What material is the floor?", ("wood", "laminate")),
                ("Does the chair have ordinary legs or wheels?", ("legs",)),
                ("What main object do you see?", ("chair",)),
            ],
            start=2,
        ):
            conversation.transcript = ""
            completed_before = conversation.backend_completions
            await conversation.say(f"Take a fresh camera picture. {question} Answer briefly after inspecting it.")
            await wait_for_content(
                conversation,
                lambda text: (
                    robot.media.get_frame.call_count >= captures
                    and conversation.backend_completions >= completed_before + 2
                    and any(word in text for word in expected)
                ),
            )


async def test_typed_request_executes_production_tool(tmp_path: Path) -> None:
    """Route typed input through the backend and preserve the robot tool result."""
    async with live_session(tmp_path) as conversation:
        movement_manager = conversation.dependencies.movement_manager
        assert isinstance(movement_manager, MagicMock)
        await conversation.say(
            "Enable head tracking by calling head_tracking with enabled=true. Confirm when enabled."
        )
        await wait_for_content(conversation, lambda text: "track" in text or "follow" in text)
        movement_manager.set_head_tracking.assert_called_once_with(True)


async def test_camera_preserves_small_label_text(tmp_path: Path) -> None:
    """Keep fine printed identifiers readable through the production image and speech path."""
    with Image.open(FIXTURES / "camera_label.png") as image:
        camera_frame = np.asarray(image.convert("RGB"))[:, :, ::-1]
    async with live_session(tmp_path, camera_frame=camera_frame) as conversation:
        robot = conversation.dependencies.reachy_mini
        assert isinstance(robot, MagicMock)
        await conversation.say(
            "Use your camera to read the printed MODEL, SERIAL, and LOT identifiers exactly. "
            "Read all three briefly; do not guess characters that you cannot see."
        )
        await wait_for_content(
            conversation,
            lambda text: all(
                identifier in re.sub(r"[^a-z0-9]", "", text) for identifier in ("rx204", "h6p94721", "b7k2")
            ),
        )
        robot.media.get_frame.assert_called_once_with()


async def test_hosted_search_returns_a_spoken_answer(tmp_path: Path) -> None:
    """Run native backend web search and return its answer through Live audio."""
    async with live_session(tmp_path) as conversation:
        await conversation.say(
            "Use web search to check the official Python documentation for the name of its standard library "
            "module for async/await concurrency. Briefly tell me the module name after checking."
        )
        await wait_for_content(
            conversation, lambda text: conversation.hosted_searches > 0 and "asyncio" in text.replace(" ", "")
        )


async def test_memory_changes_persist_across_live_sessions(tmp_path: Path) -> None:
    """Execute memory updates through Live tools and recall them in a fresh session."""
    async with live_session(tmp_path) as conversation:
        await conversation.say(
            "Remember this household preference using manage_memory: We love books about space. "
            "Confirm only after saving it."
        )
        await wait_for_content(
            conversation,
            lambda text: bool(conversation.dependencies.memory.memories) and "space" in text,
        )
        saved = conversation.dependencies.memory.model_copy(deep=True)
    assert load_memory(tmp_path) == saved
    async with live_session(tmp_path) as conversation:
        await conversation.say("What kind of books does this household like? Answer from shared household memory.")
        await wait_for_content(conversation, lambda text: "space" in text)
        conversation.transcript = ""
        await conversation.say(
            "Please forget our preference for books about space using manage_memory. Confirm after removing it."
        )
        await wait_for_content(
            conversation,
            lambda text: not conversation.dependencies.memory.memories and len(text.strip()) > 10,
        )
    assert not load_memory(tmp_path).memories


async def test_default_prompt_withholds_homework_answer_and_answers_facts_directly(tmp_path: Path) -> None:
    """Preserve the learning policy and ordinary factual answers in production Live."""
    async with live_session(tmp_path) as conversation:
        await conversation.say(
            "This is a homework exercise. Help me solve 9x + 8 = 71; I have not tried anything yet."
        )
        homework = await wait_for_content(
            conversation,
            lambda text: (
                conversation.backend_completions > 0
                and "?" in text
                and any(word in text for word in ("equation", "subtract", "isolate", "first", "both sides"))
            ),
        )
        assert re.search(r"\b(?:7|seven)\b", homework.casefold()) is None
        conversation.transcript = ""
        await conversation.say("Separate factual question, not an exercise: which planet is known as the Red Planet?")
        await wait_for_content(conversation, lambda text: "mars" in text)
