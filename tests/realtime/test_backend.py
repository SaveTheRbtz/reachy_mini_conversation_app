import json
import asyncio
from unittest.mock import AsyncMock

import pytest
from agents import FunctionTool, ToolOutputImage

from tests.support.realtime import (
    LiveTransport,
    response,
    eventually,
    backend_event,
    tool_call_event,
)
from reachy_mini_conversation_app.memory import MemorySnapshot
from reachy_mini_conversation_app.realtime import LiveConversation


@pytest.mark.asyncio
async def test_typed_text_is_backend_input(conversation: LiveConversation, live_transport: LiveTransport) -> None:
    """Route typed user text to Responses instead of voice instructions."""
    await conversation.say("My order number is A0042.")
    items = [entry.kwargs["item"] for entry in live_transport.response.item.create.await_args_list]
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "My order number is A0042."}]
    instructions = live_transport.session.instructions.append.await_args.kwargs
    assert instructions["delegation_id"] is None
    assert instructions["content"]
    assert "A0042" not in instructions["content"]


@pytest.mark.asyncio
async def test_function_batch_survives_empty_terminal_output_and_duplicate_events(
    conversation: LiveConversation, live_transport: LiveTransport, registered_tools: dict[str, FunctionTool]
) -> None:
    """Execute each collected call once and continue only after all function results."""

    def assert_complete_batch(**kwargs: object) -> None:
        assert live_transport.response.item.create.await_count == 2

    live_transport.response.create.side_effect = assert_complete_batch
    execute = AsyncMock(return_value={"status": "following"})
    registered_tools["head_tracking"].on_invoke_tool = execute
    await live_transport.send_event(
        backend_event("response.created", sequence_number=0, response=response("in_progress"))
    )
    for call_id in ("call_1", "call_2", "call_1"):
        await live_transport.send_event(
            backend_event(
                "response.output_item.done",
                sequence_number=1,
                output_index=0,
                item={
                    "id": f"item_{call_id}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "head_tracking",
                    "arguments": '{"enabled":true}',
                    "status": "completed",
                },
            )
        )
    live_transport.response.create.assert_not_awaited()
    completed = backend_event("response.completed", sequence_number=2, response=response("completed"))
    await live_transport.send_event(completed)
    await eventually(lambda: live_transport.response.create.await_count == 1)
    await live_transport.send_event(completed)
    assert execute.await_count == 2
    assert live_transport.response.item.create.await_count == 2
    assert live_transport.response.create.await_count == 1
    live_transport.session.update.assert_not_awaited()
    results = [entry.kwargs["item"] for entry in live_transport.response.item.create.await_args_list]
    assert {item["call_id"] for item in results} == {"call_1", "call_2"}
    assert all((item["type"] == "function_call_output" for item in results))


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_available", [True, False])
async def test_failed_tool_returns_error_and_continues_backend(
    tool_available: bool,
    conversation: LiveConversation,
    live_transport: LiveTransport,
    registered_tools: dict[str, FunctionTool],
) -> None:
    """Return failures for unavailable or failing tools without ending the voice stream."""
    if tool_available:
        registered_tools["head_tracking"].on_invoke_tool = AsyncMock(side_effect=ValueError("invalid arguments"))
    await live_transport.send_event(
        backend_event("response.created", sequence_number=0, response=response("in_progress"))
    )
    await live_transport.send_event(
        backend_event(
            "response.output_item.done",
            sequence_number=1,
            output_index=0,
            item={
                "id": "item_failed",
                "type": "function_call",
                "call_id": "call_failed",
                "name": "head_tracking" if tool_available else "unavailable_tool",
                "arguments": "{}",
                "status": "completed",
            },
        )
    )
    await live_transport.send_event(
        backend_event("response.completed", sequence_number=2, response=response("completed"))
    )
    await eventually(lambda: live_transport.response.create.await_count == 1)
    result = live_transport.response.item.create.await_args.kwargs["item"]
    assert result["call_id"] == "call_failed"
    assert "error" in json.loads(result["output"])
    assert conversation.connected


@pytest.mark.asyncio
async def test_typed_input_waits_for_dequeued_tool_result_before_continuation(
    conversation: LiveConversation, live_transport: LiveTransport, registered_tools: dict[str, FunctionTool]
) -> None:
    """Queue typed corrections while a tool executes without starting a competing response."""
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def execute(context: object, arguments: str) -> dict[str, str]:
        tool_started.set()
        await release_tool.wait()
        return {"status": "following"}

    registered_tools["head_tracking"].on_invoke_tool = execute
    await live_transport.send_event(
        backend_event("response.created", sequence_number=0, response=response("in_progress"))
    )
    await conversation.say("Please keep the movement gentle.")
    live_transport.response.create.assert_not_awaited()
    await live_transport.send_event(
        backend_event(
            "response.output_item.done",
            sequence_number=1,
            output_index=0,
            item={
                "id": "item_slow",
                "type": "function_call",
                "call_id": "call_slow",
                "name": "head_tracking",
                "arguments": '{"enabled":true}',
                "status": "completed",
            },
        )
    )
    await live_transport.send_event(
        backend_event("response.completed", sequence_number=2, response=response("completed"))
    )
    await asyncio.wait_for(tool_started.wait(), timeout=1)
    await conversation.say("Actually, stop tracking after this.")
    live_transport.response.create.assert_not_awaited()
    release_tool.set()
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    live_transport.response.create.assert_awaited_once()
    submitted = [entry.kwargs["item"] for entry in live_transport.response.item.create.await_args_list]
    assert [item["type"] for item in submitted] == ["message", "message", "function_call_output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_memory_changes_refresh_backend_before_continuation(
    changed: bool,
    conversation: LiveConversation,
    live_transport: LiveTransport,
    registered_tools: dict[str, FunctionTool],
) -> None:
    """Send updated memory before continuing, with no update for unchanged snapshots."""

    async def remember(context: object, arguments: str) -> dict[str, str]:
        if changed:
            conversation.dependencies.memory = MemorySnapshot(memories=["We enjoy astronomy."])
        return {"status": "updated" if changed else "unchanged"}

    def continue_backend(**kwargs: object) -> None:
        live_transport.response.item.create.assert_awaited_once()
        assert live_transport.session.update.await_count == int(changed)
        if changed:
            instructions = live_transport.session.update.await_args.kwargs["session"]["delegation"]["responses"][
                "instructions"
            ]
            assert "We enjoy astronomy." in instructions

    registered_tools["manage_memory"].on_invoke_tool = remember
    live_transport.response.create.side_effect = continue_backend
    await live_transport.send_event(
        backend_event("response.created", sequence_number=0, response=response("in_progress"))
    )
    await live_transport.send_event(tool_call_event("manage_memory", "remember"))
    await live_transport.send_event(
        backend_event("response.completed", sequence_number=2, response=response("completed"))
    )
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    live_transport.response.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_camera_image_is_a_native_tool_result(
    conversation: LiveConversation, live_transport: LiveTransport, registered_tools: dict[str, FunctionTool]
) -> None:
    """Return image content to the backend before continuing the visual response."""
    image = ToolOutputImage(image_url="data:image/jpeg;base64,aW1hZ2U=", detail="high")
    registered_tools["camera"].on_invoke_tool = AsyncMock(return_value=image)
    await live_transport.send_event(
        backend_event("response.created", sequence_number=0, response=response("in_progress"))
    )
    await live_transport.send_event(tool_call_event("camera", "camera"))
    await live_transport.send_event(
        backend_event("response.completed", sequence_number=2, response=response("completed"))
    )
    await asyncio.wait_for(conversation._tool_batches.join(), timeout=1)
    output = live_transport.response.item.create.await_args.kwargs["item"]
    assert output["call_id"] == "camera"
    assert output["output"] == [{"type": "input_image", "image_url": image.image_url, "detail": "high"}]
    live_transport.response.create.assert_awaited_once()
