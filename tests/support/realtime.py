import asyncio
from contextlib import suppress, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, create_autospec
from collections.abc import Callable, Coroutine, AsyncIterator

from pydantic import TypeAdapter
from openai.resources.live.live import (
    AsyncLiveSessionResource,
    AsyncLiveResponseResource,
    AsyncLiveResponseItemResource,
    AsyncLiveSessionInputAudioResource,
    AsyncLiveSessionInstructionsResource,
)
from openai.types.live.server_event import ServerEvent

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.realtime import LiveConversation
from reachy_mini_conversation_app.tools.types import ToolDependencies


SESSION = {"id": "session_test", "expires_at": 1000, "model": "gpt-live-1", "status": "active"}
EVENT_ADAPTER = TypeAdapter(ServerEvent)


class LiveTransport:
    """Queue typed server events and spy on the official SDK's command signatures."""

    def __init__(self) -> None:
        """Create fresh SDK command spies and an event queue."""
        self.events: asyncio.Queue[ServerEvent] = asyncio.Queue()
        self.session = create_autospec(AsyncLiveSessionResource, instance=True)
        self.session.input_audio = create_autospec(AsyncLiveSessionInputAudioResource, instance=True)
        self.session.instructions = create_autospec(AsyncLiveSessionInstructionsResource, instance=True)
        self.response = create_autospec(AsyncLiveResponseResource, instance=True)
        self.response.item = create_autospec(AsyncLiveResponseItemResource, instance=True)
        self.close = AsyncMock()
        self._received = False

    async def __aenter__(self) -> "LiveTransport":
        """Expose the connection as the SDK context manager would."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Leave transport closure to the production session owner."""
        return None

    def __aiter__(self) -> "LiveTransport":
        """Iterate queued server events."""
        return self

    async def recv(self) -> ServerEvent:
        """Acknowledge the prior event when its consumer is ready for the next one."""
        if self._received:
            self.events.task_done()
            self._received = False
        received = await self.events.get()
        self._received = True
        return received

    async def __anext__(self) -> ServerEvent:
        """Continue the production receiver."""
        return await self.recv()

    async def send_event(self, received: ServerEvent) -> None:
        """Wait for the session receiver to process a nonterminal server event."""
        self.events.put_nowait(received)
        await asyncio.wait_for(self.events.join(), timeout=2)


def event(event_type: str, **fields: object) -> ServerEvent:
    """Validate a simulated server event against the installed SDK contract."""
    return EVENT_ADAPTER.validate_python({"type": event_type, "event_id": "event_test", **fields})


def backend_event(event_type: str, **fields: object) -> ServerEvent:
    """Wrap one Responses event in the Live server envelope."""
    return event("response.event", delegation_id="delegation_test", event={"type": event_type, **fields})


def response(status: str) -> dict[str, object]:
    """Supply the required Responses envelope fields for protocol scenarios."""
    return {
        "id": "response_test",
        "created_at": 1,
        "model": "test-backend",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }


def make_conversation(output_rate: int = 24_000, voice: str = "gleam") -> LiveConversation:
    """Use real application state and constrain hardware spies to their public APIs."""
    dependencies = ToolDependencies(
        reachy_mini=MagicMock(spec=ReachyMini),
        movement_manager=MagicMock(spec=MovementManager),
        memory=MemorySnapshot(memories=[]),
    )
    return LiveConversation(dependencies, voice=voice, output_sample_rate=output_rate)


async def eventually(predicate: Callable[[], bool]) -> None:
    """Wait for an observable condition with a failure deadline, not a timing assertion."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@asynccontextmanager
async def running_task(coroutine: Coroutine[object, object, None]) -> AsyncIterator[asyncio.Task[None]]:
    """Join a background task and propagate any failure other than requested cancellation."""
    task = asyncio.create_task(coroutine)
    try:
        yield task
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def tool_call_event(name: str, call_id: str, arguments: str = "{}") -> ServerEvent:
    """Deliver a completed function call using the shared response identity."""
    return backend_event(
        "response.output_item.done",
        sequence_number=1,
        output_index=0,
        item={
            "id": f"item_{call_id}",
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        },
    )
