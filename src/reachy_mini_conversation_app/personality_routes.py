"""Personality and voice management exposed over JSON-RPC."""

import asyncio
import logging
from typing import TypeVar, Protocol
from collections.abc import Callable, Coroutine

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import (
    LOCKED_PROFILE,
    config,
    get_default_voice,
    get_available_voices,
)
from reachy_mini_conversation_app.personality import (
    delete_personality,
    list_personalities,
    save_user_personality,
    available_tool_catalog,
)
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileFormatError,
    read_profile,
    normalize_tool_names,
    canonical_profile_name,
)
from reachy_mini_conversation_app.profile_toolsets import (
    read_profile_tool_override,
)


logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


class ConversationControl(Protocol):
    """Conversation operations used as callback fallbacks by personality routes."""

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a profile."""
        ...

    async def get_available_voices(self) -> list[str]:
        """List voices."""
        ...

    def get_current_voice(self) -> str:
        """Return the active voice."""
        ...

    async def change_voice(self, voice: str) -> str:
        """Change the active voice."""
        ...


class RouteError(Exception):
    """Represent a JSON-RPC operation error with a stable reason."""

    def __init__(
        self,
        reason: str,
        *,
        extra: dict[str, object] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize an operation error."""
        super().__init__(message or reason)
        self.reason = reason
        self.extra = extra or {}
        self.message = message or reason


class PersonalityOps:
    """Provide transport-independent personality and voice operations."""

    def __init__(
        self,
        handler: ConversationControl,
        get_loop: Callable[[], asyncio.AbstractEventLoop | None],
        *,
        persist_personality: Callable[[str | None, str | None], None] | None = None,
        get_persisted_personality: Callable[[], str | None] | None = None,
    ) -> None:
        """Initialize operations with runtime callbacks."""
        self._handler = handler
        self._get_loop = get_loop
        self._persist_personality = persist_personality
        self._get_persisted_personality = get_persisted_personality
        self._startup_choice = self._configured_startup_choice()

    def _configured_startup_choice(self) -> str:
        try:
            if self._get_persisted_personality is not None:
                persisted = self._get_persisted_personality()
                if persisted:
                    return canonical_profile_name(persisted)
            return canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
        except Exception as exc:
            logger.warning("Failed to read configured startup personality: %s", exc)
            return DEFAULT_PROFILE_NAME

    def _startup_choice_value(self) -> str:
        try:
            if self._get_persisted_personality is not None:
                persisted = self._get_persisted_personality()
                if persisted:
                    return canonical_profile_name(persisted)
        except Exception as exc:
            logger.warning("Failed to read persisted startup personality: %s", exc)
        return self._startup_choice

    def _set_startup_choice(self, selected_name: str) -> None:
        self._startup_choice = canonical_profile_name(selected_name)

    def _current_choice(self) -> str:
        return canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)

    def _voice_override(self) -> str | None:
        try:
            return self._handler.get_current_voice()
        except Exception as exc:
            logger.warning("Failed to read current voice override: %s", exc)
            return None

    async def _run_on_loop(
        self,
        coroutine: Coroutine[object, object, ResultT],
        timeout: float = 10.0,
    ) -> ResultT:
        loop = self._get_loop()
        if loop is None:
            coroutine.close()
            raise RouteError("loop_unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)

    def get_choices(self) -> dict[str, object]:
        """List personalities and the current startup state."""
        return {
            "choices": list_personalities(),
            "current": self._current_choice(),
            "startup": self._startup_choice_value(),
            "locked": LOCKED_PROFILE is not None,
            "locked_to": LOCKED_PROFILE,
        }

    def _load_profile(self, name: str, available_tools: list[str]) -> dict[str, object]:
        try:
            profile = read_profile(name)
        except (FileNotFoundError, ProfileFormatError) as exc:
            logger.warning("Failed to load profile %r: %s", name, exc)
            raise RouteError("profile_unavailable", message=str(exc)) from exc
        try:
            override = read_profile_tool_override(name, config.INSTANCE_PATH)
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to load tools for profile %r: %s", name, exc)
            raise RouteError("profile_tools_unavailable", message=str(exc)) from exc
        enabled_tools = list(override) if override is not None else list(profile.default_tools)
        return {
            "instructions": profile.instructions,
            "greeting": profile.greeting or "",
            "tools_text": "".join(f"{tool_name}\n" for tool_name in enabled_tools),
            "voice": profile.voice or get_default_voice(),
            "uses_default_voice": profile.voice is None,
            "available_tools": available_tools,
            "enabled_tools": enabled_tools,
        }

    def load(self, name: str) -> dict[str, object]:
        """Load the full configuration for one personality."""
        available_tools = [tool["id"] for tool in available_tool_catalog()]
        return self._load_profile(name, available_tools)

    def save(self, payload: dict[str, object]) -> dict[str, object]:
        """Create or update a user personality."""
        if LOCKED_PROFILE is not None:
            raise RouteError("profile_locked", extra={"locked_to": LOCKED_PROFILE})

        name = str(payload.get("name", ""))
        instructions = str(payload.get("instructions", ""))
        if not instructions.strip():
            raise RouteError("invalid_instructions")
        voice = str(payload["voice"]) if payload.get("voice") is not None else None
        greeting = str(payload["greeting"]) if payload.get("greeting") is not None else None
        has_tools_text = "tools_text" in payload
        default_tools = (
            normalize_tool_names(str(payload.get("tools_text") or "").splitlines()) if has_tools_text else None
        )
        overwrite = bool(payload.get("overwrite", has_tools_text))
        try:
            value = save_user_personality(
                name,
                instructions,
                voice,
                greeting,
                overwrite=overwrite,
                default_tools=default_tools,
            )
        except FileExistsError as exc:
            raise RouteError("profile_exists") from exc
        except ProfileFormatError as exc:
            logger.warning("Failed to edit profile %r: %s", name, exc)
            raise RouteError("profile_unavailable", message=str(exc)) from exc
        except ValueError as exc:
            raise RouteError("invalid_name", message=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            logger.exception("Failed to save personality %r", name)
            raise RouteError("profile_save_failed", message=str(exc)) from exc
        return {"ok": True, "value": value, "choices": list_personalities()}

    def delete(self, name: str) -> dict[str, object]:
        """Delete an inactive user personality."""
        choices = list_personalities()
        if LOCKED_PROFILE is not None:
            raise RouteError("profile_locked", extra={"locked_to": LOCKED_PROFILE})
        profile_name = canonical_profile_name(name)
        if profile_name in (self._current_choice(), self._startup_choice_value()):
            raise RouteError("profile_in_use", extra={"choices": choices})
        try:
            deleted = delete_personality(profile_name)
        except OSError as exc:
            logger.exception("Failed to delete personality %r", profile_name)
            raise RouteError("profile_delete_failed", message=str(exc)) from exc
        if not deleted:
            raise RouteError("not_deletable", extra={"choices": choices})
        return {"ok": True, "choices": list_personalities()}

    async def apply(self, name: str, persist: bool = False, *, force: bool = False) -> dict[str, object]:
        """Apply or reload a personality and optionally persist it for startup."""
        if LOCKED_PROFILE is not None:
            raise RouteError("profile_locked", extra={"locked_to": LOCKED_PROFILE})
        selected_name = canonical_profile_name(name)
        persisted_choice = self._startup_choice_value()

        def _persist_if_requested() -> None:
            nonlocal persisted_choice
            if not persist or self._persist_personality is None:
                return
            try:
                self._persist_personality(
                    None if selected_name == DEFAULT_PROFILE_NAME else selected_name,
                    self._voice_override(),
                )
                self._set_startup_choice(selected_name)
                persisted_choice = self._startup_choice_value()
            except Exception as exc:
                logger.warning("Failed to persist startup personality: %s", exc)

        if selected_name == self._current_choice() and not force:
            _persist_if_requested()
            return {"ok": True, "status": "Personality unchanged.", "startup": persisted_choice}

        async def _apply() -> str:
            profile = None if selected_name == DEFAULT_PROFILE_NAME else selected_name
            return await self._handler.apply_personality(profile)

        try:
            status = await self._run_on_loop(_apply())
        except RouteError:
            raise
        except Exception as exc:
            raise RouteError("profile_apply_failed", message=str(exc)) from exc
        _persist_if_requested()
        return {"ok": True, "status": status, "startup": persisted_choice}

    async def voices(self) -> list[str]:
        """List voices available for OpenAI Realtime."""
        if self._get_loop() is None:
            return get_available_voices()

        async def _get_available() -> list[str]:
            return await self._handler.get_available_voices()

        try:
            return await self._run_on_loop(_get_available())
        except Exception as exc:
            logger.warning("Failed to read available voices: %s", exc)
            return get_available_voices()

    def current_voice(self) -> dict[str, str]:
        """Return the current voice."""
        try:
            return {"voice": self._handler.get_current_voice()}
        except Exception as exc:
            logger.warning("Failed to read current voice: %s", exc)
            return {"voice": get_default_voice()}

    async def apply_voice(self, voice: str) -> dict[str, object]:
        """Change the current voice by restarting the Realtime session."""
        selected_voice = voice.strip()
        if not selected_voice:
            raise RouteError("missing_voice")

        async def _apply() -> str:
            return await self._handler.change_voice(selected_voice)

        try:
            status = await self._run_on_loop(_apply())
        except RouteError:
            raise
        except Exception as exc:
            raise RouteError("voice_apply_failed", message=str(exc)) from exc
        return {"ok": True, "status": status}


def build_personality_ops(
    handler: ConversationControl,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    persist_personality: Callable[[str | None, str | None], None] | None = None,
    get_persisted_personality: Callable[[], str | None] | None = None,
) -> PersonalityOps:
    """Build personality operations for a control transport."""
    return PersonalityOps(
        handler,
        get_loop,
        persist_personality=persist_personality,
        get_persisted_personality=get_persisted_personality,
    )


def register_personality_methods(rpc: JsonRpcServer, ops: PersonalityOps) -> None:
    """Register personality and voice operations as JSON-RPC methods."""

    def _wrap(operation: Callable[[dict[str, object]], object]) -> Callable[[dict[str, object]], object]:
        async def _method(params: dict[str, object]) -> object:
            try:
                result = operation(params)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except RouteError as exc:
                raise JsonRpcError(
                    exc.message,
                    reason=exc.reason,
                    data=exc.extra,
                    code=-32000,
                ) from exc

        return _method

    rpc.register("personalities.list", _wrap(lambda params: ops.get_choices()))
    rpc.register("personalities.load", _wrap(lambda params: ops.load(str(params["name"]))))
    rpc.register("personalities.save", _wrap(ops.save))
    rpc.register("personalities.delete", _wrap(lambda params: ops.delete(str(params["name"]))))
    rpc.register(
        "personalities.apply",
        _wrap(
            lambda params: ops.apply(
                str(params.get("name", "")),
                bool(params.get("persist", False)),
                force=bool(params.get("force", False)),
            )
        ),
    )
    rpc.register("voices.list", _wrap(lambda params: ops.voices()))
    rpc.register("voices.current", _wrap(lambda params: ops.current_voice()))
    rpc.register("voices.apply", _wrap(lambda params: ops.apply_voice(str(params.get("voice", "")))))
