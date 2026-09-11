import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.live.response_event import ResponseEvent

from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.realtime import LiveConversation


FAILURE_EVENTS = ("response.failed", "response.incomplete", "response.cancelled", "error")
pytestmark = pytest.mark.asyncio


@pytest.fixture
def conversation() -> LiveConversation:
    """Provide a production conversation with observable transport commands."""
    dependencies = SimpleNamespace(
        instance_path=None,
        memory=MemorySnapshot(memories=[]),
        movement_manager=SimpleNamespace(set_listening=MagicMock(), set_speaking=MagicMock()),
    )
    conversation = LiveConversation(dependencies, voice="gleam", output_sample_rate=24_000)
    conversation._connection = SimpleNamespace(
        session=SimpleNamespace(update=AsyncMock(), instructions=SimpleNamespace(append=AsyncMock())),
        response=SimpleNamespace(create=AsyncMock(), item=SimpleNamespace(create=AsyncMock())),
    )
    return conversation


def _response_event(
    event_type: str, response_id: str = "response_first", delegation_id: str | None = "delegation_first"
) -> ResponseEvent:
    nested: dict[str, object] = {"type": event_type, "sequence_number": 1}
    if event_type == "error":
        nested.update(code="server_error", message="Backend failed")
    else:
        nested["response"] = {
            "id": response_id,
            "created_at": 1,
            "model": "gpt-6-astra",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "in_progress" if event_type == "response.created" else event_type.removeprefix("response."),
        }
    return ResponseEvent(type="response.event", event_id="event_test", delegation_id=delegation_id, event=nested)


def _tool_call_event() -> ResponseEvent:
    return ResponseEvent(
        type="response.event",
        event_id="event_tool",
        delegation_id="delegation_first",
        event={
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "item_tool",
                "type": "function_call",
                "call_id": "call_tool",
                "name": "head_tracking",
                "arguments": '{"enabled":true}',
                "status": "completed",
            },
        },
    )


@pytest.mark.parametrize("failure", FAILURE_EVENTS)
async def test_failed_backend_continues_queued_text_without_running_aborted_tools(
    conversation: LiveConversation, failure: str
) -> None:
    """Answer accepted typed input once while discarding calls from the failed response."""
    execute = AsyncMock(return_value={"status": "following"})
    conversation._tools = {"head_tracking": SimpleNamespace(on_invoke_tool=execute)}
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(_response_event("response.created"))
        await conversation.say("Please answer this request instead.")
        await conversation._handle_event(_tool_call_event())
        conversation._connection.response.create.assert_not_awaited()
        await conversation._handle_event(_response_event(failure))
        if failure != "error":
            await conversation._handle_event(_response_event(failure))
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)

        execute.assert_not_awaited()
        conversation._connection.response.create.assert_awaited_once()
        submitted = conversation._connection.response.item.create.await_args.kwargs["item"]
        assert submitted["content"] == [{"type": "input_text", "text": "Please answer this request instead."}]
        conversation._connection.response.item.create.assert_awaited_once()
        assert conversation.connected
        await conversation.say("And keep it brief.")
        conversation._connection.response.create.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("failure", FAILURE_EVENTS)
async def test_failed_backend_without_queued_input_accepts_next_request(
    conversation: LiveConversation, failure: str
) -> None:
    """Leave the backend idle after failure without retrying the failed request."""
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(_response_event("response.created"))
        await conversation._handle_event(_response_event(failure))
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        conversation._connection.response.create.assert_not_awaited()
        await conversation.say("A new request.")
        conversation._connection.response.create.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("failure", FAILURE_EVENTS[:-1])
async def test_late_failure_during_tool_execution_does_not_start_competing_response(
    conversation: LiveConversation, failure: str
) -> None:
    """Finish a dequeued tool batch before responding to queued typed input."""
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(context: object, arguments: str) -> dict[str, str]:
        tool_started.set()
        await release_tool.wait()
        return {"status": "following"}

    conversation._tools = {"head_tracking": SimpleNamespace(on_invoke_tool=execute)}
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(_response_event("response.created"))
        await conversation._handle_event(_tool_call_event())
        await conversation._handle_event(_response_event("response.completed"))
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        await conversation._handle_event(_response_event(failure))
        await conversation._handle_event(_response_event(failure, delegation_id="unknown"))
        await conversation.say("Keep the movement gentle.")
        conversation._connection.response.create.assert_not_awaited()
        release_tool.set()
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)

        submitted = [entry.kwargs["item"] for entry in conversation._connection.response.item.create.await_args_list]
        assert [item["type"] for item in submitted] == ["message", "function_call_output"]
        conversation._connection.response.create.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("tool_in_flight", [False, True])
async def test_uncorrelated_backend_error_ends_session(conversation: LiveConversation, tool_in_flight: bool) -> None:
    """Surface uncorrelated errors so session supervision can recover from a stuck backend."""
    tool_started = asyncio.Event()

    async def execute(context: object, arguments: str) -> None:
        tool_started.set()
        await asyncio.Event().wait()

    conversation._tools = {"head_tracking": SimpleNamespace(on_invoke_tool=execute)}
    worker = asyncio.create_task(conversation._run_tools())
    try:
        if tool_in_flight:
            await conversation._handle_event(_response_event("response.created"))
            await conversation._handle_event(_tool_call_event())
            await conversation._handle_event(_response_event("response.completed"))
            await asyncio.wait_for(tool_started.wait(), timeout=1)
        await conversation.say("A request waiting for a backend response.")
        with pytest.raises(RuntimeError, match="Live backend error without an active response"):
            await conversation._handle_event(
                _response_event("error", delegation_id="delegation_first" if tool_in_flight else None)
            )
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("failure", FAILURE_EVENTS)
async def test_failed_delegation_waits_for_another_active_response(
    conversation: LiveConversation, failure: str
) -> None:
    """Keep queued text pending while another backend response remains active."""
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(_response_event("response.created"))
        await conversation._handle_event(_response_event("response.created", "response_second", "delegation_second"))
        await conversation.say("Another request.")
        await conversation._handle_event(_response_event(failure))
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        await conversation.say("A clarification.")
        conversation._connection.response.create.assert_not_awaited()
        await conversation._handle_event(_response_event("response.completed", "response_second", "delegation_second"))
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        conversation._connection.response.create.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("failure", FAILURE_EVENTS[:-1])
async def test_stale_failure_does_not_discard_new_response_in_same_delegation(
    conversation: LiveConversation, failure: str
) -> None:
    """Match terminal response IDs before changing the current delegation state."""
    worker = asyncio.create_task(conversation._run_tools())
    try:
        await conversation._handle_event(_response_event("response.created", "response_second"))
        await conversation._handle_event(_response_event(failure))
        await conversation.say("Please finish the current answer.")
        conversation._connection.response.create.assert_not_awaited()
        await conversation._handle_event(_response_event("response.completed", "response_second"))
        await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
        conversation._connection.response.create.assert_awaited_once()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
