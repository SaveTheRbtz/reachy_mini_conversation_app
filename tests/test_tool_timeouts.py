import json
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import FunctionTool
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall

import reachy_mini_conversation_app.realtime as realtime_module
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.realtime import BackendResponse, LiveConversation
from reachy_mini_conversation_app.tools.types import ToolDependencies
from reachy_mini_conversation_app.tools.stop_dance import stop_dance


@pytest.mark.asyncio
async def test_timed_out_tool_returns_error_and_unblocks_robot_tools(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Report a stalled tool and still execute the following robot request."""
    pending_output: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def stalled_tool(_context: ToolContext[ToolDependencies], _arguments: str) -> str:
        return await pending_output

    network_lookup = FunctionTool(
        name="network_lookup",
        description="Look up information over the network.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=stalled_tool,
    )
    dependencies = ToolDependencies(MagicMock(), MagicMock(), MemorySnapshot(memories=[]))
    conversation = LiveConversation(dependencies, voice="gleam", output_sample_rate=24_000)
    connection = MagicMock()
    connection.response.item.create = AsyncMock()
    connection.response.create = AsyncMock()
    connection.session.update = AsyncMock()
    conversation._connection = connection
    conversation._tools = {"network_lookup": network_lookup, "stop_dance": stop_dance}
    monkeypatch.setattr(realtime_module, "TOOL_TIMEOUT_SECONDS", 0.01)

    for tool_name in ("network_lookup", "stop_dance"):
        call = ResponseFunctionToolCall(
            arguments="{}", call_id=f"call_{tool_name}", name=tool_name, type="function_call"
        )
        conversation._tool_batches.put_nowait(
            BackendResponse(f"response_{tool_name}", f"delegation_{tool_name}", {call.call_id: call})
        )
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        outputs = [
            json.loads(call.kwargs["item"]["output"]) for call in connection.response.item.create.await_args_list
        ]
        assert len(outputs) == 2
        assert "TimeoutError" in outputs[0]["error"]
        assert outputs[1] == {"status": "stopped dance and cleared queue"}
        assert pending_output.cancelled()
        dependencies.movement_manager.clear_move_queue.assert_called_once_with()
        assert connection.response.create.await_count == 2
        assert "Tool failed: network_lookup" in caplog.text
        assert not worker.done()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
