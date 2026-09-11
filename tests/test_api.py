"""Contract behavior through the generated Connect ASGI application."""

import asyncio
from typing import cast
from pathlib import Path
from contextlib import aclosing
from dataclasses import dataclass
from unittest.mock import MagicMock
from collections.abc import Mapping, AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from connectrpc.code import Code
from starlette.types import ASGIApp
from connectrpc.errors import ConnectError

import reachy_mini_conversation_app.api as api_module
import reachy_mini_conversation_app.profile_store as profile_store_module
from reachy_mini_conversation_app.api import RequestErrors, RequestLimits, ConversationService, profile_resource_name
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.profile_store import read_profile, write_profile
from reachy_mini_conversation_app.profile_toolsets import read_profile_tool_override, write_profile_tool_override
from reachy_mini_conversation_app.startup_settings import read_startup_settings
from reachy_mini_conversation_app.gen.reachy.conversation.v1 import api_pb as messages
from reachy_mini_conversation_app.gen.reachy.conversation.v1.api_connect import (
    ConversationServiceASGIApplication,
)


class Control:
    """Record local actions without robot hardware or an OpenAI connection."""

    def __init__(self) -> None:
        """Initialize a disconnected conversation with an idle microphone."""
        self.profile = "default"
        self.muted = False
        self.playing = False
        self.restarts = 0
        self.texts: list[str] = []
        self.interrupts = 0

    def snapshot(self) -> messages.Conversation:
        """Return a fresh complete state snapshot."""
        return messages.Conversation(
            name="conversation",
            profile=profile_resource_name(self.profile),
            voice="gleam",
            muted=self.muted,
            playing=self.playing,
            connection_state=messages.Conversation.ConnectionState.DISCONNECTED,
        )

    async def restart(self, profile: str) -> None:
        """Record a profile selection without contacting OpenAI."""
        if profile:
            self.profile = profile
        self.restarts += 1

    async def say(self, text: str) -> None:
        """Record one user message."""
        self.texts.append(text)

    async def interrupt(self) -> None:
        """Stop local playback while disconnected."""
        self.playing = False
        self.interrupts += 1

    async def set_muted(self, muted: bool) -> None:
        """Record microphone state."""
        self.muted = muted

    async def set_api_key(self, api_key: str) -> None:
        """Keep a synthetic credential in server configuration."""
        config.OPENAI_API_KEY = api_key


@dataclass
class ApiFixture:
    """Expose the production transport and observable local state."""

    client: httpx.AsyncClient
    control: Control
    service: ConversationService
    instance_path: Path

    async def call(self, method: str, payload: Mapping[str, object]) -> httpx.Response:
        """Send a ProtoJSON request through the real generated endpoint."""
        return await self.client.post(f"/rpc/reachy.conversation.v1.ConversationService/{method}", json=payload)


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ApiFixture]:
    """Mount the generated API against temporary profile and preference storage."""
    profiles = tmp_path / "profiles"
    instance = tmp_path / "instance"
    instance.mkdir()
    write_profile("default", profiles / "default", "Default instructions.", ["dance", "web_search"])
    write_profile("Legacy_Guide", profiles / "Legacy_Guide", "Legacy instructions.", ["camera"])
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setattr(config, "INSTANCE_PATH", instance)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(profile_store_module, "DEFAULT_PROFILES_DIRECTORY", profiles)
    control = Control()
    service = ConversationService(control, instance)
    app = FastAPI()
    connect_app = ConversationServiceASGIApplication(service, interceptors=[RequestErrors(), RequestLimits()])
    # Connect uses typed ASGI events; Starlette still annotates them as mutable dictionaries.
    app.mount("/rpc", cast(ASGIApp, connect_app))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        yield ApiFixture(client, control, service, instance)


async def _create(api: ApiFixture, profile_id: str = "user-guide") -> httpx.Response:
    response = await api.call(
        "CreateProfile",
        {
            "profileId": profile_id,
            "profile": {
                "instructions": "Guide instructions.",
                "greeting": "Hello.",
                "defaultToolIds": ["camera", "dance"],
            },
        },
    )
    assert response.status_code == 200, response.text
    return response


@pytest.mark.asyncio
async def test_profile_crud_preserves_inheritance_and_resource_identity(api: ApiFixture) -> None:
    """Create, read, partially update and delete the same generated resource."""
    created = (await _create(api)).json()
    assert created["name"] == "profiles/user-guide"
    assert created["displayName"] == "Guide"
    assert created.get("voice", "") == ""
    assert created["effectiveToolIds"] == ["camera", "dance"]
    assert read_profile("user_personalities/guide").voice is None

    updated = await api.call(
        "UpdateProfile",
        {
            "profile": {"name": created["name"], "greeting": "", "instructions": "Ignored without its mask."},
            "updateMask": "greeting",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["instructions"] == "Guide instructions."
    assert updated.json().get("greeting", "") == ""
    assert updated.json()["defaultToolIds"] == ["camera", "dance"]
    assert api.control.restarts == 0
    loaded = await api.call("GetProfile", {"name": created["name"]})
    assert loaded.json() == updated.json()
    deleted = await api.call("DeleteProfile", {"name": created["name"]})
    assert deleted.status_code == 200
    assert deleted.json() == {}
    missing = await api.call("GetProfile", {"name": created["name"]})
    assert missing.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_duplicate_profile_creation_preserves_existing_content(api: ApiFixture) -> None:
    """Creating an existing ID must not overwrite its stored personality."""
    await _create(api)
    duplicate = await api.call("CreateProfile", {"profileId": "user-guide", "profile": {"instructions": "New."}})
    assert duplicate.json()["code"] == "already_exists"
    assert read_profile("user_personalities/guide").instructions == "Guide instructions."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile_id", ["builtin-guide", "user-../escape", "user-Upper", "user-has_underscore", "user-1guide"]
)
async def test_new_profile_ids_require_safe_user_names(api: ApiFixture, profile_id: str) -> None:
    """New resource IDs cannot escape storage or claim a bundled identity."""
    response = await api.call("CreateProfile", {"profileId": profile_id, "profile": {"instructions": "New."}})
    assert response.json()["code"] == "invalid_argument", response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["user_personalities/guide", "profiles/user-../guide", "profiles/builtin-default/../guide"]
)
async def test_profile_lookups_reject_nonresource_paths(api: ApiFixture, name: str) -> None:
    """Storage paths and traversal expressions are not public resource names."""
    response = await api.call("GetProfile", {"name": name})
    assert response.json()["code"] == "invalid_argument", response.text


@pytest.mark.asyncio
async def test_bundled_profiles_accept_empty_tool_override_and_reset(api: ApiFixture) -> None:
    """Presence distinguishes disabling all tools from inheriting authored defaults."""
    disabled = await api.call(
        "UpdateProfile",
        {"profile": {"name": "profiles/builtin-default", "toolOverride": {}}, "updateMask": "toolOverride"},
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["toolOverride"] == {}
    assert disabled.json().get("effectiveToolIds", []) == []
    assert read_profile_tool_override("default", api.instance_path) == []
    reset = await api.call(
        "UpdateProfile",
        {"profile": {"name": "profiles/builtin-default"}, "updateMask": "toolOverride"},
    )
    assert reset.status_code == 200
    assert "toolOverride" not in reset.json()
    assert reset.json()["effectiveToolIds"] == ["dance", "web_search"]
    assert read_profile_tool_override("default", api.instance_path) is None
    assert api.control.restarts == 0


@pytest.mark.asyncio
async def test_authored_defaults_and_tool_override_update_independently(api: ApiFixture) -> None:
    """Editing a profile never silently clears the user's tool selection."""
    await _create(api)
    write_profile_tool_override("user_personalities/guide", ["camera"], api.instance_path)
    updated = await api.call(
        "UpdateProfile",
        {
            "profile": {"name": "profiles/user-guide", "instructions": "Changed.", "defaultToolIds": ["dance"]},
            "updateMask": "instructions,defaultToolIds",
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["defaultToolIds"] == ["dance"]
    assert updated.json()["toolOverride"] == {"toolIds": ["camera"]}
    assert updated.json()["effectiveToolIds"] == ["camera"]
    assert read_profile_tool_override("user_personalities/guide", api.instance_path) == ["camera"]


@pytest.mark.asyncio
async def test_nested_override_mask_clears_list_without_removing_override(api: ApiFixture) -> None:
    """Masking a subfield preserves the override message's presence."""
    updated = await api.call(
        "UpdateProfile",
        {"profile": {"name": "profiles/builtin-default"}, "updateMask": "toolOverride.toolIds"},
    )
    assert updated.status_code == 200
    assert updated.json()["toolOverride"] == {}
    assert updated.json().get("effectiveToolIds", []) == []


@pytest.mark.asyncio
async def test_masks_support_false_reject_unknown_and_ignore_output_fields(api: ApiFixture) -> None:
    """Masks allow clearing bools and cannot mutate observed state."""
    muted = await api.call("UpdateConversation", {"conversation": {"name": "conversation", "muted": True}})
    assert muted.status_code == 200
    assert api.control.muted is True
    unmuted = await api.call(
        "UpdateConversation",
        {"conversation": {"name": "conversation", "playing": True}, "updateMask": "muted,playing"},
    )
    assert unmuted.status_code == 200
    assert api.control.muted is False
    assert api.control.playing is False
    invalid = await api.call("UpdateConversation", {"conversation": {"name": "conversation"}, "updateMask": "mutedd"})
    assert invalid.json()["code"] == "invalid_argument"
    malformed = await api.call("UpdateConversation", {"conversation": {"name": "conversation", "muted": "false"}})
    assert malformed.json()["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_settings_and_active_profile_remain_independent(api: ApiFixture) -> None:
    """Saving startup settings does not unexpectedly restart the active dialogue."""
    await _create(api)
    saved = await api.call(
        "UpdateSettings",
        {
            "settings": {"name": "settings", "startupProfile": "profiles/user-guide", "voiceOverride": "coral"},
            "updateMask": "startupProfile,voiceOverride",
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["startupProfile"] == "profiles/user-guide"
    assert api.control.profile == "default"
    assert api.control.restarts == 0
    assert read_startup_settings(api.instance_path).profile == "user_personalities/guide"
    assert read_startup_settings(api.instance_path).voice == "coral"
    applied = await api.call("RestartConversation", {"name": "conversation", "profile": "profiles/user-guide"})
    assert applied.status_code == 200
    assert api.control.profile == "user_personalities/guide"
    assert api.control.restarts == 1
    cleared = await api.call("UpdateSettings", {"settings": {"name": "settings"}, "updateMask": "voiceOverride"})
    assert cleared.status_code == 200
    assert cleared.json().get("voiceOverride", "") == ""
    assert read_startup_settings(api.instance_path).voice is None
    assert read_startup_settings(api.instance_path).profile == "user_personalities/guide"


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", ["active", "startup"])
async def test_delete_rejects_profiles_in_use(api: ApiFixture, usage: str) -> None:
    """Both runtime and persisted references prevent deleting a profile."""
    await _create(api)
    if usage == "active":
        api.control.profile = "user_personalities/guide"
    else:
        await api.call("UpdateSettings", {"settings": {"name": "settings", "startupProfile": "profiles/user-guide"}})
    response = await api.call("DeleteProfile", {"name": "profiles/user-guide"})
    assert response.json()["code"] == "failed_precondition"
    assert read_profile("user_personalities/guide").instructions == "Guide instructions."


@pytest.mark.asyncio
async def test_bundled_content_and_unknown_tools_are_rejected(api: ApiFixture) -> None:
    """Read-only authored content and invalid tool IDs fail without changing storage."""
    for method, payload in (
        ("DeleteProfile", {"name": "profiles/builtin-Legacy_Guide"}),
        (
            "UpdateProfile",
            {
                "profile": {"name": "profiles/builtin-default", "instructions": "Changed."},
                "updateMask": "instructions",
            },
        ),
    ):
        response = await api.call(method, payload)
        assert response.json()["code"] == "failed_precondition"
    invalid = await api.call(
        "UpdateProfile",
        {
            "profile": {"name": "profiles/builtin-default", "toolOverride": {"toolIds": ["invented_tool"]}},
            "updateMask": "toolOverride",
        },
    )
    assert invalid.json()["code"] == "invalid_argument"
    assert read_profile_tool_override("default", api.instance_path) is None


@pytest.mark.asyncio
async def test_locked_profiles_reject_edits_but_allow_recovery(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locking a profile leaves local interruption and restarting the same profile available."""
    monkeypatch.setattr(api_module, "LOCKED_PROFILE", "default")
    for method, payload in (
        ("CreateProfile", {"profileId": "user-new", "profile": {"instructions": "Hello."}}),
        ("DeleteProfile", {"name": "profiles/builtin-Legacy_Guide"}),
        (
            "UpdateProfile",
            {"profile": {"name": "profiles/builtin-default", "toolOverride": {}}, "updateMask": "toolOverride"},
        ),
        ("RestartConversation", {"name": "conversation", "profile": "profiles/builtin-Legacy_Guide"}),
    ):
        response = await api.call(method, payload)
        assert response.json()["code"] == "failed_precondition", response.text
    assert (await api.call("RestartConversation", {"name": "conversation"})).status_code == 200
    assert (await api.call("InterruptConversation", {"name": "conversation"})).status_code == 200
    assert api.control.interrupts == 1


@pytest.mark.asyncio
async def test_credentials_never_appear_in_resources(api: ApiFixture) -> None:
    """Credential writes expose status only and never trigger an implicit restart."""
    secret = "sk-synthetic-contract-test"
    saved = await api.call("SetSettingsApiKey", {"name": "settings", "apiKey": secret})
    assert saved.status_code == 200
    assert saved.json()["apiKeyConfigured"] is True
    assert secret not in saved.text
    for method, name in (("GetSettings", "settings"), ("GetConversation", "conversation")):
        response = await api.call(method, {"name": name})
        assert secret not in response.text
    assert api.control.restarts == 0


@pytest.mark.asyncio
async def test_legacy_names_and_pagination_round_trip(api: ApiFixture) -> None:
    """Names remain distinct and pages contain complete resources, including legacy IDs."""
    await _create(api)
    seen: list[str] = []
    token = ""
    while True:
        response = await api.call("ListProfiles", {"pageSize": 1, "pageToken": token})
        assert response.status_code == 200, response.text
        page = response.json()
        assert len(page["profiles"]) == 1
        profile = page["profiles"][0]
        assert profile["instructions"]
        seen.append(profile["name"])
        token = page.get("nextPageToken", "")
        if not token:
            break
    assert seen == sorted({"profiles/builtin-default", "profiles/builtin-Legacy_Guide", "profiles/user-guide"})
    for payload in ({"pageSize": -1}, {"pageToken": "not-a-token"}):
        assert (await api.call("ListProfiles", payload)).json()["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_watch_starts_with_complete_state_and_tracks_local_changes(api: ApiFixture) -> None:
    """The stream needs no separate snapshot fetch or event reconciliation."""
    async with aclosing(
        api.service.watch_conversation(messages.WatchConversationRequest(name="conversation"), MagicMock())
    ) as snapshots:
        first = await anext(snapshots)
        assert first.connection_state == messages.Conversation.ConnectionState.DISCONNECTED
        assert first.muted is False
        api.control.muted = True
        api.control.playing = True
        second = await asyncio.wait_for(anext(snapshots), timeout=1)
        assert second.muted is True
        assert second.playing is True


@pytest.mark.asyncio
async def test_profile_write_failure_restores_previous_override(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed authored write must not leave half of a combined resource update applied."""
    await _create(api)
    write_profile_tool_override("user_personalities/guide", ["dance"], api.instance_path)

    def fail_save(*args: object, **kwargs: object) -> str:
        raise OSError("disk unavailable")

    monkeypatch.setattr(api_module, "save_user_personality", fail_save)
    response = await api.call(
        "UpdateProfile",
        {
            "profile": {
                "name": "profiles/user-guide",
                "instructions": "Changed.",
                "toolOverride": {"toolIds": ["camera"]},
            },
            "updateMask": "instructions,toolOverride",
        },
    )
    assert response.json()["code"] == "internal"
    assert read_profile("user_personalities/guide").instructions == "Guide instructions."
    assert read_profile_tool_override("user_personalities/guide", api.instance_path) == ["dance"]


@pytest.mark.asyncio
async def test_network_errors_use_canonical_unavailable_status(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disconnected model is reported distinctly from a storage error."""

    async def unavailable(text: str) -> None:
        raise ConnectionError("remote disconnected")

    monkeypatch.setattr(api.control, "say", unavailable)
    response = await api.call("SendConversationText", {"name": "conversation", "text": "Hello"})
    assert response.json()["code"] == "unavailable"
    assert "remote disconnected" not in response.text


@pytest.mark.asyncio
async def test_server_deadline_cancels_stalled_command(api: ApiFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server enforces a caller deadline even when the client keeps waiting."""
    cancelled = asyncio.Event()

    async def blocked(text: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(api.control, "say", blocked)
    response = await asyncio.wait_for(
        api.client.post(
            "/rpc/reachy.conversation.v1.ConversationService/SendConversationText",
            json={"name": "conversation", "text": "Hello"},
            headers={"Connect-Timeout-Ms": "100"},
        ),
        timeout=2,
    )
    assert response.json()["code"] == "deadline_exceeded", response.text
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_watch_rejects_invalid_resource(api: ApiFixture) -> None:
    """A stream request validates its resource before sending any snapshots."""
    async with aclosing(
        api.service.watch_conversation(messages.WatchConversationRequest(name="wrong"), MagicMock())
    ) as snapshots:
        with pytest.raises(ConnectError) as error:
            await anext(snapshots)
    assert error.value.code == Code.INVALID_ARGUMENT
