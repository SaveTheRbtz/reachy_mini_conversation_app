import asyncio
from unittest.mock import AsyncMock, MagicMock
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from agents import FunctionTool

from tests.support.realtime import SESSION, LiveTransport, event, eventually, running_task, make_conversation
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.realtime import LiveConversation


@pytest.fixture
def live_transport(monkeypatch: pytest.MonkeyPatch) -> LiveTransport:
    """Replace the external SDK connection while retaining the production session lifecycle."""
    transport = LiveTransport()
    client = MagicMock()
    client.live.connect.return_value = transport
    client.__aenter__.return_value = client
    monkeypatch.setattr(realtime_module, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    return transport


@pytest_asyncio.fixture
async def conversation(
    live_transport: LiveTransport, registered_tools: dict[str, FunctionTool]
) -> AsyncIterator[LiveConversation]:
    """Own the receiver and workers for each backend protocol scenario."""
    conversation = make_conversation()
    async with running_task(conversation.start_up()) as session:
        live_transport.events.put_nowait(event("session.started", session=SESSION))
        await eventually(lambda: conversation.connected)
        live_transport.session.instructions.append.reset_mock()
        yield conversation
        live_transport.events.put_nowait(
            event("session.closed", session=SESSION, reason="close_requested", usage={"seconds": 1.0})
        )
        await asyncio.wait_for(session, timeout=2)


@pytest.fixture
def registered_tools(monkeypatch: pytest.MonkeyPatch) -> dict[str, FunctionTool]:
    """Supply controllable tool boundaries through the production registration function."""
    tools = {
        name: FunctionTool(
            name=name,
            description=f"Test {name} boundary",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=AsyncMock(return_value={"status": "following"}),
        )
        for name in ("head_tracking", "camera", "manage_memory", "network_lookup", "stop_dance")
    }
    monkeypatch.setattr(realtime_module, "get_function_tools", lambda _names: list(tools.values()))
    return tools
