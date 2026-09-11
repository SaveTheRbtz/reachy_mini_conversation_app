import asyncio
from unittest.mock import MagicMock, create_autospec

from reachy_mini import ReachyMini
from reachy_mini.io.ws_client import WSClient
from reachy_mini_conversation_app.realtime import LiveConversation


def make_robot() -> MagicMock:
    """Provide controllable audio I/O through the SDK's public robot surface."""
    robot = create_autospec(ReachyMini, instance=True)
    robot.client = create_autospec(WSClient, instance=True)
    robot.media.get_input_audio_samplerate.return_value = 16_000
    robot.media.get_output_audio_samplerate.return_value = 48_000
    return robot


def make_conversation() -> MagicMock:
    """Isolate the stream owner while checking its public conversation calls."""
    conversation = create_autospec(LiveConversation, instance=True)
    conversation.voice = "gleam"
    conversation.history = []
    conversation.connected = True
    conversation.output_queue = asyncio.Queue()
    return conversation
