import asyncio
from unittest.mock import AsyncMock

import pytest
from agents import FunctionTool
from openai.types.live.response_event import ResponseEvent

from tests.support.realtime import SESSION, LiveTransport, event, eventually, running_task, make_conversation
from reachy_mini_conversation_app.realtime import LiveConversation


FAILURE_EVENTS = ("response.failed", "response.incomplete", "response.cancelled", "error")
pytestmark = pytest.mark.asyncio


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
    conversation: LiveConversation,
    failure: str,
    live_transport: LiveTransport,
    registered_tools: dict[str, FunctionTool],
) -> None:
    """Answer accepted typed input once while discarding calls from the failed response."""
    execute = AsyncMock(return_value={"status": "following"})
    registered_tools["head_tracking"].on_invoke_tool = execute
    await live_transport.send_event(_response_event("response.created"))
    await conversation.say("Please answer this request instead.")
    await live_transport.send_event(_tool_call_event())
    live_transport.response.create.assert_not_awaited()
    await live_transport.send_event(_response_event(failure))
    if failure != "error":
        await live_transport.send_event(_response_event(failure))
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    execute.assert_not_awaited()
    live_transport.response.create.assert_awaited_once()
    submitted = live_transport.response.item.create.await_args.kwargs["item"]
    assert submitted["content"] == [{"type": "input_text", "text": "Please answer this request instead."}]
    live_transport.response.item.create.assert_awaited_once()
    assert conversation.connected
    await conversation.say("And keep it brief.")
    live_transport.response.create.assert_awaited_once()


@pytest.mark.parametrize("failure", FAILURE_EVENTS)
async def test_failed_backend_without_queued_input_accepts_next_request(
    conversation: LiveConversation,
    failure: str,
    live_transport: LiveTransport,
) -> None:
    """Leave the backend idle after failure without retrying the failed request."""
    await live_transport.send_event(_response_event("response.created"))
    await live_transport.send_event(_response_event(failure))
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    live_transport.response.create.assert_not_awaited()
    await conversation.say("A new request.")
    live_transport.response.create.assert_awaited_once()


@pytest.mark.parametrize("failure", FAILURE_EVENTS[:-1])
async def test_late_failure_during_tool_execution_does_not_start_competing_response(
    conversation: LiveConversation,
    failure: str,
    live_transport: LiveTransport,
    registered_tools: dict[str, FunctionTool],
) -> None:
    """Finish a dequeued tool batch before responding to queued typed input."""
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(context: object, arguments: str) -> dict[str, str]:
        tool_started.set()
        await release_tool.wait()
        return {"status": "following"}

    registered_tools["head_tracking"].on_invoke_tool = execute
    await live_transport.send_event(_response_event("response.created"))
    await live_transport.send_event(_tool_call_event())
    await live_transport.send_event(_response_event("response.completed"))
    await asyncio.wait_for(tool_started.wait(), timeout=1)
    await live_transport.send_event(_response_event(failure))
    await live_transport.send_event(_response_event(failure, delegation_id="unknown"))
    await conversation.say("Keep the movement gentle.")
    live_transport.response.create.assert_not_awaited()
    release_tool.set()
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    submitted = [entry.kwargs["item"] for entry in live_transport.response.item.create.await_args_list]
    assert [item["type"] for item in submitted] == ["message", "function_call_output"]
    live_transport.response.create.assert_awaited_once()


@pytest.mark.parametrize("tool_in_flight", [False, True], ids=["waiting-for-response", "tool-running"])
async def test_uncorrelated_backend_error_ends_session(
    live_transport: LiveTransport, registered_tools: dict[str, FunctionTool], tool_in_flight: bool
) -> None:
    """Surface fatal protocol errors through the session owner and cancel unfinished work."""
    conversation = make_conversation()
    tool_started = asyncio.Event()
    tool_cancelled = asyncio.Event()

    async def execute(context: object, arguments: str) -> None:
        tool_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            tool_cancelled.set()

    registered_tools["head_tracking"].on_invoke_tool = execute
    with pytest.raises(ExceptionGroup) as failure:
        async with running_task(conversation.start_up()) as session:
            live_transport.events.put_nowait(event("session.started", session=SESSION))
            await eventually(lambda: conversation.connected)
            if tool_in_flight:
                await live_transport.send_event(_response_event("response.created"))
                await live_transport.send_event(_tool_call_event())
                await live_transport.send_event(_response_event("response.completed"))
                await asyncio.wait_for(tool_started.wait(), timeout=2)
            await conversation.say("A request waiting for a backend response.")
            live_transport.events.put_nowait(
                _response_event("error", delegation_id="delegation_first" if tool_in_flight else None)
            )
            await asyncio.wait_for(session, timeout=2)
    assert any("Live backend error without an active response" in str(error) for error in failure.value.exceptions)
    assert not conversation.connected
    assert tool_cancelled.is_set() is tool_in_flight
    live_transport.close.assert_awaited_once()


@pytest.mark.parametrize("failure", FAILURE_EVENTS)
async def test_failed_delegation_waits_for_another_active_response(
    conversation: LiveConversation,
    failure: str,
    live_transport: LiveTransport,
) -> None:
    """Keep queued text pending while another backend response remains active."""
    await live_transport.send_event(_response_event("response.created"))
    await live_transport.send_event(_response_event("response.created", "response_second", "delegation_second"))
    await conversation.say("Another request.")
    await live_transport.send_event(_response_event(failure))
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    await conversation.say("A clarification.")
    live_transport.response.create.assert_not_awaited()
    await live_transport.send_event(_response_event("response.completed", "response_second", "delegation_second"))
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    live_transport.response.create.assert_awaited_once()


@pytest.mark.parametrize("failure", FAILURE_EVENTS[:-1])
async def test_stale_failure_does_not_discard_new_response_in_same_delegation(
    conversation: LiveConversation,
    failure: str,
    live_transport: LiveTransport,
) -> None:
    """Match terminal response IDs before changing the current delegation state."""
    await live_transport.send_event(_response_event("response.created", "response_second"))
    await live_transport.send_event(_response_event(failure))
    await conversation.say("Please finish the current answer.")
    live_transport.response.create.assert_not_awaited()
    await live_transport.send_event(_response_event("response.completed", "response_second"))
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    live_transport.response.create.assert_awaited_once()
