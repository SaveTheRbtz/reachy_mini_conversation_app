import json
import asyncio
from unittest.mock import AsyncMock, MagicMock
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from websockets.uri import parse_uri
from websockets.client import ClientProtocol
from websockets.frames import Close, Frame, Opcode
from websockets.http11 import Response
from websockets.protocol import OPEN, CLOSED
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.client import ClientConnection
from websockets.datastructures import Headers
from openai.resources.live.live import AsyncLiveConnection

from tests.support.realtime import make_conversation
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.realtime import LiveConversation


pytestmark = pytest.mark.integration


@pytest.fixture
def conversation() -> LiveConversation:
    """Build a conversation with observable movement state."""
    return make_conversation()


@pytest_asyncio.fixture
async def sdk_transport(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[ClientConnection, MagicMock]]:
    """Use the actual SDK and WebSocket lifecycle with a controllable socket transport."""
    # Keep test deadlines above Windows' roughly 16 ms clock resolution.
    websocket = ClientConnection(
        ClientProtocol(parse_uri("wss://example.invalid/live"), state=OPEN),
        ping_interval=None,
        close_timeout=0.1,
    )
    websocket.process_event(Response(101, "Switching Protocols", Headers()))
    transport = MagicMock(spec=asyncio.Transport)
    transport.is_closing.return_value = False
    transport.can_write_eof.return_value = False
    closing = False

    def close() -> None:
        nonlocal closing
        if not closing:
            closing = True
            asyncio.get_running_loop().call_soon(websocket.connection_lost, None)

    transport.abort.side_effect = close
    transport.close.side_effect = close
    websocket.connection_made(transport)
    monkeypatch.setattr("openai.lib._websocket._WebSocketConnect", AsyncMock(return_value=websocket))
    monkeypatch.setattr(realtime_module.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(realtime_module, "SESSION_TIMEOUT_SECONDS", 0.1)
    try:
        yield websocket, transport
    finally:
        transport.abort()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_shutdown_closes_connecting_socket_gracefully(
    conversation: LiveConversation, sdk_transport: tuple[ClientConnection, MagicMock]
) -> None:
    """Keep graceful WebSocket closure when the peer acknowledges promptly."""
    websocket, transport = sdk_transport
    close_reply = Frame(Opcode.CLOSE, Close(1000, "").serialize()).serialize(mask=False)

    def acknowledge_close(_packet: bytes) -> None:
        loop = asyncio.get_running_loop()
        loop.call_soon(websocket.data_received, close_reply)
        loop.call_soon(transport.close)

    transport.write.side_effect = acknowledge_close
    conversation._transport = AsyncLiveConnection(websocket)

    await asyncio.wait_for(conversation.shutdown(), timeout=1)

    assert websocket.state is CLOSED
    transport.close.assert_called_once()
    transport.abort.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_releases_backpressured_socket(
    conversation: LiveConversation, sdk_transport: tuple[ClientConnection, MagicMock]
) -> None:
    """Stop a blocked close command and release the receiver without socket progress."""
    websocket, transport = sdk_transport
    websocket.pause_writing()
    connection = AsyncLiveConnection(websocket)
    conversation._connection = connection
    conversation._transport = connection
    receiver = asyncio.create_task(connection.recv())
    shutdown = asyncio.create_task(conversation.shutdown())
    try:
        await asyncio.wait_for(shutdown, timeout=1)
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(receiver, timeout=1)
        assert websocket.state is CLOSED
        transport.abort.assert_called()
        assert not conversation.connected
    finally:
        shutdown.cancel()
        receiver.cancel()
        await asyncio.gather(shutdown, receiver, return_exceptions=True)


@pytest.mark.asyncio
async def test_startup_error_releases_backpressured_sdk_context(
    conversation: LiveConversation, sdk_transport: tuple[ClientConnection, MagicMock]
) -> None:
    """Preserve a startup failure while the SDK context closes a stalled socket."""
    websocket, transport = sdk_transport
    startup = asyncio.create_task(conversation.start_up())
    try:
        async with asyncio.timeout(1):
            while transport.write.call_count == 0:
                await asyncio.sleep(0)
        websocket.pause_writing()
        event = {
            "type": "error",
            "event_id": "startup_error",
            "error": {"type": "server_error", "code": "test_failure", "message": "Simulated startup failure"},
        }
        websocket.data_received(Frame(Opcode.TEXT, json.dumps(event).encode()).serialize(mask=False))

        with pytest.raises(RuntimeError, match="Live startup failed"):
            await asyncio.wait_for(startup, timeout=1)
        assert websocket.state is CLOSED
        transport.abort.assert_called()
        assert not conversation.connected
    finally:
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)
