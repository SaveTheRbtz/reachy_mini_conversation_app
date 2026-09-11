"""Serve the production Connect API over loopback for browser-client integration tests."""

import json
import socket
import asyncio
from typing import cast
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI
from starlette.types import ASGIApp
from connectrpc.request import RequestContext

from reachy_mini_conversation_app.api import RequestErrors, RequestLimits, ConversationService, profile_resource_name
from reachy_mini_conversation_app.config import config, set_instance_path, set_custom_profile
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_pb import (
    Conversation,
    WatchConversationRequest,
)
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_connect import (
    ConversationServiceASGIApplication,
)


class ConversationControl:
    """Replace robot and network side effects while keeping production API behavior."""

    def __init__(self) -> None:
        """Initialize an idle connected conversation."""
        self.muted = False
        self.profile = "profiles/builtin-default"
        self.active_says = 0
        self.completed_says = 0
        self.restarts = 0

    def snapshot(self) -> Conversation:
        """Return current robot state as the generated resource."""
        return Conversation(
            name="conversation",
            profile=self.profile,
            voice="gleam",
            model="gpt-live-1",
            connection_state=Conversation.ConnectionState.CONNECTED,
            muted=self.muted,
        )

    async def restart(self, profile: str) -> None:
        """Record the restart without opening a model connection."""
        self.restarts += 1
        if profile:
            set_custom_profile(profile)
            self.profile = profile_resource_name(profile)

    async def say(self, text: str) -> None:
        """Expose a cancellable stalled remote send for deadline coverage."""
        self.active_says += 1
        try:
            if text == "stall":
                await asyncio.sleep(60)
            elif text == "unavailable":
                raise ConnectionError("Live connection unavailable")
            self.completed_says += 1
        finally:
            self.active_says -= 1

    async def interrupt(self) -> None:
        """Model an immediate playback interruption."""

    async def set_muted(self, muted: bool) -> None:
        """Apply microphone state for snapshot-stream coverage."""
        self.muted = muted

    async def set_api_key(self, api_key: str) -> None:
        """Keep synthetic credentials inside the fixture process."""
        config.OPENAI_API_KEY = api_key


class ObservedService(ConversationService):
    """Count active watchers around the production streaming implementation."""

    def __init__(self, control: ConversationControl, instance_path: Path) -> None:
        """Initialize the production service and watcher counter."""
        super().__init__(control, instance_path)
        self.active_watches = 0

    async def watch_conversation(
        self,
        request: WatchConversationRequest,
        ctx: RequestContext[WatchConversationRequest, Conversation],
    ) -> AsyncIterator[Conversation]:
        """Record cleanup when a browser cancels its snapshot stream."""
        self.active_watches += 1
        try:
            async for snapshot in super().watch_conversation(request, ctx):
                yield snapshot
        finally:
            self.active_watches -= 1


async def main() -> None:
    """Print the bound port, then serve until the client test stops this process."""
    with TemporaryDirectory(prefix="reachy-connect-test-") as directory, socket.socket() as listener:
        instance_path = Path(directory)
        set_instance_path(instance_path)
        set_custom_profile(None)
        config.OPENAI_API_KEY = None
        control = ConversationControl()
        service = ObservedService(control, instance_path)
        app = FastAPI()
        application = ConversationServiceASGIApplication(service, interceptors=[RequestErrors(), RequestLimits()])
        # Starlette and asgiref annotate the same ASGI protocol with incompatible container types.
        app.mount("/rpc", cast(ASGIApp, application))

        @app.get("/stats")
        async def stats() -> dict[str, int]:
            return {
                "active_watches": service.active_watches,
                "active_says": control.active_says,
                "completed_says": control.completed_says,
                "restarts": control.restarts,
            }

        listener.bind(("127.0.0.1", 0))
        print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
        await uvicorn.Server(uvicorn.Config(app, log_level="warning")).serve(sockets=[listener])


if __name__ == "__main__":
    asyncio.run(main())
