import json
import asyncio
from unittest.mock import MagicMock

import pytest
from agents import FunctionTool
from agents.tool_context import ToolContext

from tests.support.realtime import LiveTransport, response, backend_event, tool_call_event
import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.realtime import LiveConversation
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.stop_dance import stop_dance


@pytest.mark.asyncio
async def test_timed_out_tool_returns_error_and_unblocks_robot_tools(
    conversation: LiveConversation,
    live_transport: LiveTransport,
    registered_tools: dict[str, FunctionTool],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report a stalled tool and still execute the following robot request."""
    movement_manager = conversation.dependencies.movement_manager
    assert isinstance(movement_manager, MagicMock)
    pending_output: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def stalled_tool(_context: ToolContext[ToolDependencies], _arguments: str) -> str:
        return await pending_output

    registered_tools["network_lookup"].on_invoke_tool = stalled_tool
    registered_tools["stop_dance"].on_invoke_tool = stop_dance.on_invoke_tool
    monkeypatch.setattr(realtime_module, "TOOL_TIMEOUT_SECONDS", 0.1)

    for tool_name in ("network_lookup", "stop_dance"):
        await live_transport.send_event(
            backend_event("response.created", sequence_number=0, response=response("in_progress"))
        )
        await live_transport.send_event(tool_call_event(tool_name, f"call_{tool_name}"))
        await live_transport.send_event(
            backend_event("response.completed", sequence_number=2, response=response("completed"))
        )
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=2)

    outputs = [
        json.loads(call.kwargs["item"]["output"]) for call in live_transport.response.item.create.await_args_list
    ]
    assert len(outputs) == 2
    assert "TimeoutError" in outputs[0]["error"]
    assert outputs[1] == {"status": "stopped dance and cleared queue"}
    assert pending_output.cancelled()
    movement_manager.clear_move_queue.assert_called_once_with()
    assert live_transport.response.create.await_count == 2
    assert "Tool failed: network_lookup" in caplog.text
    assert conversation.connected
