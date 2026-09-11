"""Typed Connect operations for the conversation and its local configuration."""

import re
import time
import base64
import asyncio
import logging
from typing import TypeVar, Protocol
from pathlib import Path
from collections.abc import Callable, Awaitable, AsyncIterator

from protobuf.wkt import Empty, FieldMask
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from connectrpc.interceptor import UnaryInterceptor, MetadataInterceptor

from reachy_mini_conversation_app.config import (
    LOCKED_PROFILE,
    USER_PERSONALITIES_DIRNAME,
    config,
    get_default_voice,
    has_openai_api_key,
    get_available_voices,
)
from reachy_mini_conversation_app.personality import (
    delete_personality,
    list_personalities,
    save_user_personality,
)
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileFormatError,
    read_profile,
    normalize_tool_names,
    canonical_profile_name,
    profile_directory_has_definition,
)
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
    clear_profile_tool_override,
    write_profile_tool_override,
    profile_toolsets_transaction,
)
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings
from reachy_mini_conversation_app.tools.core_tools import available_tool_catalog
from reachy_mini_conversation_app.gen.reachy.conversation.v1 import api_pb as messages
from reachy_mini_conversation_app.gen.reachy.conversation.v1 import api_connect


logger = logging.getLogger(__name__)
WATCH_POLL_SECONDS = 0.25
WATCH_HEARTBEAT_SECONDS = 5.0
Request = TypeVar("Request")
Response = TypeVar("Response")
_PROFILE_CONTENT_FIELDS = {"instructions", "greeting", "voice", "default_tool_ids"}
_PROFILE_MUTABLE_FIELDS = _PROFILE_CONTENT_FIELDS | {"tool_override"}


class ConversationControl(Protocol):
    """Local conversation operations owned by the audio loop."""

    def snapshot(self) -> messages.Conversation:
        """Return the current local state without waiting for the network."""
        ...

    async def restart(self, profile: str) -> None:
        """Select a storage profile and accept a restart; empty retains the selection."""
        ...

    async def say(self, text: str) -> None:
        """Send text to the conversation."""
        ...

    async def interrupt(self) -> None:
        """Interrupt playback immediately."""
        ...

    async def set_muted(self, muted: bool) -> None:
        """Change local microphone capture."""
        ...

    async def set_api_key(self, api_key: str) -> None:
        """Persist a credential without exposing it in snapshots."""
        ...


class RequestErrors(MetadataInterceptor[None]):
    """Translate malformed protobuf input before it reaches a method handler."""

    async def on_start(self, ctx: RequestContext[Request, Response]) -> None:
        """Install error handling before request decoding."""
        return None

    async def on_end(self, token: None, ctx: RequestContext[Request, Response], error: Exception | None) -> None:
        """Return a canonical input error for malformed wire values."""
        if isinstance(error, (TypeError, ValueError)):
            logger.warning("RPC %s received an invalid message: %s", ctx.method.name, error)
            raise ConnectError(Code.INVALID_ARGUMENT, "The request does not match the API contract.") from error


class RequestLimits(UnaryInterceptor):
    """Bound unary work and translate storage failures at the transport boundary."""

    async def intercept_unary(
        self,
        call_next: Callable[[Request, RequestContext[Request, Response]], Awaitable[Response]],
        request: Request,
        ctx: RequestContext[Request, Response],
    ) -> Response:
        """Apply the earlier of the caller deadline and the local request limit."""
        remaining = ctx.timeout_ms
        timeout = min(remaining / 1000, 8.0) if remaining is not None else 8.0
        try:
            async with asyncio.timeout(timeout):
                return await call_next(request, ctx)
        except TimeoutError as error:
            logger.warning("RPC %s exceeded its deadline", ctx.method.name)
            raise ConnectError(
                Code.DEADLINE_EXCEEDED, "The request timed out. Refresh before retrying changes."
            ) from error
        except ValueError as error:
            logger.warning("RPC %s rejected: %s", ctx.method.name, error)
            raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
        except (ConnectionError, RuntimeError) as error:
            logger.warning("RPC %s unavailable: %s", ctx.method.name, error)
            raise ConnectError(
                Code.UNAVAILABLE, "The conversation is unavailable. Wait for it to reconnect."
            ) from error
        except TypeError as error:
            logger.exception("RPC %s failed", ctx.method.name)
            raise ConnectError(Code.INTERNAL, "Unable to process the request. Check the app logs.") from error
        except OSError as error:
            logger.exception("RPC %s failed", ctx.method.name)
            raise ConnectError(
                Code.INTERNAL, "Unable to read or save app configuration. Check the app logs."
            ) from error


def profile_resource_name(profile: str | None) -> str:
    """Map an existing storage selector to a stable API resource name."""
    selector = canonical_profile_name(profile)
    if selector.startswith(f"{USER_PERSONALITIES_DIRNAME}/"):
        return f"profiles/user-{selector.split('/', 1)[1]}"
    return f"profiles/builtin-{selector}"


def _profile_selector(name: str) -> str:
    match = re.fullmatch(r"profiles/(builtin|user)-([a-zA-Z0-9_-]+)", name)
    if match is None:
        raise ConnectError(Code.INVALID_ARGUMENT, "Choose an available profile resource.")
    kind, folder = match.groups()
    return f"{USER_PERSONALITIES_DIRNAME}/{folder}" if kind == "user" else folder


def _require_name(name: str, expected: str) -> None:
    if name != expected:
        raise ConnectError(Code.INVALID_ARGUMENT, f"The resource name must be '{expected}'.")


def _require_unlocked() -> None:
    if LOCKED_PROFILE is not None:
        raise ConnectError(Code.FAILED_PRECONDITION, "Profile selection and editing are locked.")


def _update_paths(
    resource: messages.Conversation | messages.Settings | messages.Profile,
    mask: FieldMask | None,
    mutable: set[str],
) -> set[str]:
    fields = resource.desc().fields
    paths = (
        set(mask.paths) if mask is not None and mask.paths else {field.name for field in fields if field in resource}
    )
    if paths == {"*"}:
        return mutable.copy()
    known = {field.name for field in fields}
    if isinstance(resource, messages.Profile):
        known.add("tool_override.tool_ids")
    if invalid := paths - known:
        raise ConnectError(Code.INVALID_ARGUMENT, f"Unknown update fields: {', '.join(sorted(invalid))}.")
    return paths & (mutable | {"tool_override.tool_ids"})


def _validate_voice(voice: str) -> None:
    if voice and voice not in get_available_voices():
        raise ConnectError(Code.INVALID_ARGUMENT, "Choose an available voice or leave it empty to inherit.")


def _validate_tools(tool_ids: list[str], previous: list[str]) -> list[str]:
    selected = normalize_tool_names(tool_ids)
    allowed = {entry["id"] for entry in available_tool_catalog()} | set(previous)
    if unknown := set(selected) - allowed:
        raise ConnectError(Code.INVALID_ARGUMENT, f"Unavailable tools: {', '.join(sorted(unknown))}.")
    return selected


class ConversationService(api_connect.ConversationService):
    """Implement the resource contract using the existing profile and settings stores."""

    def __init__(self, control: ConversationControl, instance_path: str | Path | None) -> None:
        """Bind the API to one local conversation and its instance directory."""
        self._control = control
        self._instance_path = Path(instance_path) if instance_path is not None else None
        self._startup_profile = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)

    async def get_conversation(
        self,
        request: messages.GetConversationRequest,
        ctx: RequestContext[messages.GetConversationRequest, messages.Conversation],
    ) -> messages.Conversation:
        """Return a complete conversation snapshot."""
        _require_name(request.name, "conversation")
        return self._control.snapshot()

    async def update_conversation(
        self,
        request: messages.UpdateConversationRequest,
        ctx: RequestContext[messages.UpdateConversationRequest, messages.Conversation],
    ) -> messages.Conversation:
        """Update local microphone state without depending on the Live connection."""
        if request.conversation is None:
            raise ConnectError(Code.INVALID_ARGUMENT, "Conversation is required.")
        _require_name(request.conversation.name, "conversation")
        if "muted" in _update_paths(request.conversation, request.update_mask, {"muted"}):
            await self._control.set_muted(request.conversation.muted)
        return self._control.snapshot()

    async def watch_conversation(
        self,
        request: messages.WatchConversationRequest,
        ctx: RequestContext[messages.WatchConversationRequest, messages.Conversation],
    ) -> AsyncIterator[messages.Conversation]:
        """Stream current state with periodic snapshots to detect stalled connections."""
        _require_name(request.name, "conversation")
        previous: messages.Conversation | None = None
        last_sent = 0.0
        while True:
            current = self._control.snapshot()
            if current != previous or time.monotonic() - last_sent >= WATCH_HEARTBEAT_SECONDS:
                yield current
                previous = current
                last_sent = time.monotonic()
            await asyncio.sleep(WATCH_POLL_SECONDS)

    async def restart_conversation(
        self,
        request: messages.RestartConversationRequest,
        ctx: RequestContext[messages.RestartConversationRequest, messages.Conversation],
    ) -> messages.Conversation:
        """Accept an explicit profile reload or selection."""
        _require_name(request.name, "conversation")
        selector = ""
        if request.profile:
            if request.profile != self._control.snapshot().profile:
                _require_unlocked()
            await asyncio.to_thread(self._load_profile, request.profile)
            selector = _profile_selector(request.profile)
        await self._control.restart(selector)
        return self._control.snapshot()

    async def send_conversation_text(
        self,
        request: messages.SendConversationTextRequest,
        ctx: RequestContext[messages.SendConversationTextRequest, messages.SendConversationTextResponse],
    ) -> messages.SendConversationTextResponse:
        """Send one typed user message without retrying it automatically."""
        _require_name(request.name, "conversation")
        text = request.text.strip()
        if not text:
            raise ConnectError(Code.INVALID_ARGUMENT, "Enter a message to send.")
        await self._control.say(text)
        return messages.SendConversationTextResponse()

    async def interrupt_conversation(
        self,
        request: messages.InterruptConversationRequest,
        ctx: RequestContext[messages.InterruptConversationRequest, messages.InterruptConversationResponse],
    ) -> messages.InterruptConversationResponse:
        """Stop current playback through the local conversation owner."""
        _require_name(request.name, "conversation")
        await self._control.interrupt()
        return messages.InterruptConversationResponse()

    def _settings(self) -> messages.Settings:
        persisted = read_startup_settings(self._instance_path)
        return messages.Settings(
            name="settings",
            startup_profile=profile_resource_name(persisted.profile or self._startup_profile),
            voice_override=persisted.voice or "",
            api_key_configured=has_openai_api_key(),
            locked_profile=profile_resource_name(LOCKED_PROFILE) if LOCKED_PROFILE is not None else "",
        )

    async def get_settings(
        self, request: messages.GetSettingsRequest, ctx: RequestContext[messages.GetSettingsRequest, messages.Settings]
    ) -> messages.Settings:
        """Read startup preferences without returning credentials."""
        _require_name(request.name, "settings")
        return await asyncio.to_thread(self._settings)

    def _save_settings(self, request: messages.UpdateSettingsRequest) -> messages.Settings:
        updated = request.settings
        if updated is None:
            raise ConnectError(Code.INVALID_ARGUMENT, "Settings are required.")
        _require_name(updated.name, "settings")
        paths = _update_paths(updated, request.update_mask, {"startup_profile", "voice_override"})
        with profile_toolsets_transaction():
            current = self._settings()
            if "startup_profile" in paths:
                _require_unlocked()
                selected = updated.startup_profile or profile_resource_name(DEFAULT_PROFILE_NAME)
                self._load_profile(selected)
                current.startup_profile = selected
            if "voice_override" in paths:
                _validate_voice(updated.voice_override)
                current.voice_override = updated.voice_override
            if paths:
                selected_profile = _profile_selector(current.startup_profile)
                write_startup_settings(
                    self._instance_path,
                    profile=selected_profile,
                    voice=current.voice_override or None,
                )
                self._startup_profile = selected_profile
            return current

    async def update_settings(
        self,
        request: messages.UpdateSettingsRequest,
        ctx: RequestContext[messages.UpdateSettingsRequest, messages.Settings],
    ) -> messages.Settings:
        """Save startup preferences; applying them requires an explicit restart."""
        return await asyncio.to_thread(self._save_settings, request)

    async def set_settings_api_key(
        self,
        request: messages.SetSettingsApiKeyRequest,
        ctx: RequestContext[messages.SetSettingsApiKeyRequest, messages.Settings],
    ) -> messages.Settings:
        """Store an API credential and return only its configured status."""
        _require_name(request.name, "settings")
        if not request.api_key.strip():
            raise ConnectError(Code.INVALID_ARGUMENT, "Enter an OpenAI API key.")
        await self._control.set_api_key(request.api_key)
        return await asyncio.to_thread(self._settings)

    async def get_capabilities(
        self,
        request: messages.GetCapabilitiesRequest,
        ctx: RequestContext[messages.GetCapabilitiesRequest, messages.Capabilities],
    ) -> messages.Capabilities:
        """Read the available tools and voices once for settings editors."""
        _require_name(request.name, "capabilities")
        return messages.Capabilities(
            name="capabilities",
            available_tools=[
                messages.ToolInfo(id=tool["id"], description=tool["description"]) for tool in available_tool_catalog()
            ],
            available_voices=get_available_voices(),
            default_voice=get_default_voice(),
        )

    def _load_profile(self, name: str) -> messages.Profile:
        selector = _profile_selector(name)
        with profile_toolsets_transaction():
            try:
                definition = read_profile(selector)
                override = read_profile_tool_override(selector, self._instance_path)
            except FileNotFoundError as error:
                raise ConnectError(Code.NOT_FOUND, "The profile no longer exists.") from error
            except (ProfileFormatError, RuntimeError) as error:
                logger.warning("Unable to read profile %s: %s", name, error)
                raise ConnectError(
                    Code.FAILED_PRECONDITION, "The profile configuration is invalid. Check the app logs."
                ) from error
            return messages.Profile(
                name=name,
                display_name=selector.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ").title(),
                instructions=definition.instructions,
                greeting=definition.greeting or "",
                voice=definition.voice or "",
                default_tool_ids=list(definition.default_tools),
                tool_override=messages.ToolSelection(tool_ids=override) if override is not None else None,
                effective_tool_ids=override if override is not None else list(definition.default_tools),
                editable=selector.startswith(f"{USER_PERSONALITIES_DIRNAME}/") and LOCKED_PROFILE is None,
            )

    def _list_profiles(self, request: messages.ListProfilesRequest) -> messages.ListProfilesResponse:
        if request.page_size < 0:
            raise ConnectError(Code.INVALID_ARGUMENT, "Page size must not be negative.")
        after = ""
        if request.page_token:
            try:
                after = base64.b64decode(request.page_token, altchars=b"-_", validate=True).decode("utf-8")
                _profile_selector(after)
            except (ValueError, UnicodeError) as error:
                raise ConnectError(Code.INVALID_ARGUMENT, "The profile page token is invalid.") from error
        size = min(request.page_size or 50, 100)
        with profile_toolsets_transaction():
            names = {profile_resource_name(selector) for selector in list_personalities()}
            active = self._control.snapshot().profile
            if active and profile_directory_has_definition(config.resolve_profile_dir(_profile_selector(active))):
                names.add(active)
            remaining = sorted(name for name in names if name > after)
            selected = remaining[:size]
            return messages.ListProfilesResponse(
                profiles=[self._load_profile(name) for name in selected],
                next_page_token=base64.urlsafe_b64encode(selected[-1].encode()).decode()
                if len(remaining) > size
                else "",
            )

    async def list_profiles(
        self,
        request: messages.ListProfilesRequest,
        ctx: RequestContext[messages.ListProfilesRequest, messages.ListProfilesResponse],
    ) -> messages.ListProfilesResponse:
        """List full profile resources using stable name ordering."""
        return await asyncio.to_thread(self._list_profiles, request)

    async def get_profile(
        self, request: messages.GetProfileRequest, ctx: RequestContext[messages.GetProfileRequest, messages.Profile]
    ) -> messages.Profile:
        """Load authored settings and their effective tool selection."""
        return await asyncio.to_thread(self._load_profile, request.name)

    def _write_profile(self, profile: messages.Profile, paths: set[str], *, create: bool) -> messages.Profile:
        _require_unlocked()
        selector = _profile_selector(profile.name)
        folder = selector.rsplit("/", 1)[-1]
        with profile_toolsets_transaction():
            current = messages.Profile(name=profile.name) if create else self._load_profile(profile.name)
            content_changed = bool(paths & _PROFILE_CONTENT_FIELDS)
            if content_changed and not selector.startswith(f"{USER_PERSONALITIES_DIRNAME}/"):
                raise ConnectError(
                    Code.FAILED_PRECONDITION, "Bundled profile content is read-only. Create a custom profile."
                )
            if create and config.resolve_profile_dir(selector).exists():
                raise ConnectError(Code.ALREADY_EXISTS, "A custom profile with this ID already exists.")
            if "instructions" in paths:
                if not profile.instructions.strip():
                    raise ConnectError(Code.INVALID_ARGUMENT, "Profile instructions must not be empty.")
                current.instructions = profile.instructions.strip()
            if "greeting" in paths:
                current.greeting = profile.greeting.strip()
            if "voice" in paths:
                _validate_voice(profile.voice)
                current.voice = profile.voice
            if "default_tool_ids" in paths:
                current.default_tool_ids = _validate_tools(profile.default_tool_ids, current.default_tool_ids)
            previous_override = current.tool_override
            override_changed = bool(paths & {"tool_override", "tool_override.tool_ids"})
            if override_changed:
                override = profile.tool_override
                if override is None and "tool_override.tool_ids" in paths:
                    override = messages.ToolSelection()
                current.tool_override = (
                    messages.ToolSelection(tool_ids=_validate_tools(override.tool_ids, current.effective_tool_ids))
                    if override is not None
                    else None
                )
                if current.tool_override is None:
                    clear_profile_tool_override(selector, self._instance_path)
                else:
                    write_profile_tool_override(selector, current.tool_override.tool_ids, self._instance_path)
            try:
                if content_changed:
                    save_user_personality(
                        folder,
                        current.instructions,
                        voice=current.voice or None,
                        greeting=current.greeting or None,
                        default_tools=current.default_tool_ids,
                        overwrite=not create,
                    )
            except (OSError, ValueError):
                if override_changed:
                    try:
                        if previous_override is None:
                            clear_profile_tool_override(selector, self._instance_path)
                        else:
                            write_profile_tool_override(selector, previous_override.tool_ids, self._instance_path)
                    except (OSError, RuntimeError):
                        logger.exception("Failed to restore tool settings for %s", profile.name)
                raise
            return self._load_profile(profile.name)

    async def create_profile(
        self,
        request: messages.CreateProfileRequest,
        ctx: RequestContext[messages.CreateProfileRequest, messages.Profile],
    ) -> messages.Profile:
        """Create a custom profile with explicit defaults and an optional override."""
        if re.fullmatch(r"user-[a-z](?:[a-z0-9-]{0,56}[a-z0-9])?", request.profile_id) is None:
            raise ConnectError(
                Code.INVALID_ARGUMENT,
                "Profile IDs must be user- followed by a lowercase name using letters, numbers and dashes.",
            )
        if request.profile is None:
            raise ConnectError(Code.INVALID_ARGUMENT, "Profile is required.")
        profile = request.profile
        profile.name = f"profiles/{request.profile_id}"
        return await asyncio.to_thread(self._write_profile, profile, _PROFILE_MUTABLE_FIELDS, create=True)

    async def update_profile(
        self,
        request: messages.UpdateProfileRequest,
        ctx: RequestContext[messages.UpdateProfileRequest, messages.Profile],
    ) -> messages.Profile:
        """Update selected profile fields while retaining unrelated authored settings and overrides."""
        if request.profile is None:
            raise ConnectError(Code.INVALID_ARGUMENT, "Profile is required.")
        paths = _update_paths(request.profile, request.update_mask, _PROFILE_MUTABLE_FIELDS)
        return await asyncio.to_thread(self._write_profile, request.profile, paths, create=False)

    def _delete_profile(self, name: str) -> Empty:
        _require_unlocked()
        with profile_toolsets_transaction():
            profile = self._load_profile(name)
            if not profile.editable:
                raise ConnectError(Code.FAILED_PRECONDITION, "Bundled profiles cannot be deleted.")
            if name in {self._control.snapshot().profile, self._settings().startup_profile}:
                raise ConnectError(
                    Code.FAILED_PRECONDITION, "Select a different active and startup profile before deleting this one."
                )
            if not delete_personality(_profile_selector(name)):
                raise ConnectError(Code.FAILED_PRECONDITION, "The profile cannot be deleted.")
        return Empty()

    async def delete_profile(
        self, request: messages.DeleteProfileRequest, ctx: RequestContext[messages.DeleteProfileRequest, Empty]
    ) -> Empty:
        """Delete an inactive custom profile and its tool override."""
        return await asyncio.to_thread(self._delete_profile, request.name)
