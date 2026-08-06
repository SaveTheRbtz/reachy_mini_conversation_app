import os
import re
import html
import json
import time
import logging
import secrets
import threading
from typing import Literal
from pathlib import Path
from dataclasses import field, dataclass


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_MEMORY_NOTES = 20
MAX_MEMORY_CHARS = 280
MEMORY_FILENAME = "memory.v1.json"

_SENSITIVE_PATTERN = re.compile(
    r"\b(?:password(?!\s+manager\b)|passcode|api[ -]?key|access token|secret key|credit card|debit card|cvv|"
    r"bank account|routing number|social security|ssn|passport number|driver'?s license|"
    r"medical record|diagnos(?:is|ed)|health condition|medication|prescription|diabet(?:es|ic)|cancer|"
    r"hypertension|asthma|allerg(?:y|ies)|pregnan(?:t|cy))\b",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERN = re.compile(
    r"\bsk-[a-zA-Z0-9_-]{12,}\b|\b\d{3}-\d{2}-\d{4}\b|(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)
_MEMORY_ID_PATTERN = re.compile(r"m_[a-zA-Z0-9_-]{1,64}\Z")
_INSTRUCTION_PATTERN = re.compile(
    r"\b(?:ignore|disregard|override)\b.{0,40}\b(?:instruction|prompt|policy|rule)s?\b|"
    r"\b(?:system|developer)\s+(?:message|prompt|instruction)s?\b|"
    r"\b(?:obey|follow)\b.{0,64}\b(?:note|memory|instruction|prompt|policy|rule|request|message)s?\b|"
    r"\bfrom now on\b.{0,80}\b(?:answer|respond|reply|say|act|behave|do|call|use|obey|follow|ignore)\b|"
    r"\b(?:always|never)\s+(?:answer|respond|reply|say|call|use|obey|follow|ignore)\b|"
    r"\b(?:answer|respond|reply)\b.{0,50}\b(?:every|all)\b.{0,30}\b(?:question|request|message)s?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryNote:
    """One durable, user-authored memory note."""

    id: str
    text: str

    def to_json(self) -> dict[str, object]:
        """Return the persisted memory-store shape."""
        return {
            "id": self.id,
            "text": self.text,
        }


@dataclass(frozen=True)
class MemoryChange:
    """Describe the result of a memory write."""

    note: MemoryNote
    status: Literal["saved", "updated", "unchanged"]


def memory_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the durable memory path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / MEMORY_FILENAME
    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "reachy_mini_conversation_app" / MEMORY_FILENAME


def _validated_memory_text(text: str) -> str:
    if len(text) > MAX_MEMORY_CHARS:
        raise ValueError(f"memory must be at most {MAX_MEMORY_CHARS} characters")
    normalized = " ".join(text.split()).strip()
    if not normalized:
        raise ValueError("memory must be a non-empty fact")
    if _SENSITIVE_PATTERN.search(normalized) or _SECRET_VALUE_PATTERN.search(normalized):
        raise ValueError("sensitive identifiers and secrets cannot be stored")
    if _INSTRUCTION_PATTERN.search(normalized):
        raise ValueError("instructions and policy overrides cannot be stored as memories")
    return normalized


def _validated_memory_id(memory_id: str) -> str:
    if _MEMORY_ID_PATTERN.fullmatch(memory_id) is None:
        raise ValueError("invalid memory id")
    return memory_id


def _new_memory_id() -> str:
    return f"m_{int(time.time() * 1000)}_{secrets.token_hex(3)}"


def _memory_note_from_json(value: object) -> MemoryNote | None:
    if not isinstance(value, dict):
        return None
    memory_id = value.get("id")
    text = value.get("text")
    if not isinstance(memory_id, str) or not isinstance(text, str):
        return None
    try:
        _validated_memory_id(memory_id)
    except ValueError:
        logger.warning("Ignoring memory with an invalid id")
        return None
    try:
        normalized = _validated_memory_text(text)
    except ValueError as error:
        logger.warning("Ignoring unsafe memory %s: %s", memory_id, error)
        return None
    return MemoryNote(id=memory_id, text=normalized)


@dataclass
class MemoryState:
    """Local-first memory state shared through the Agents SDK run context."""

    path: Path | None = None
    notes: list[MemoryNote] = field(default_factory=list)
    revision: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, instance_path: str | Path | None = None) -> "MemoryState":
        """Load a bounded memory state for an app instance."""
        path = memory_path_for_instance(instance_path)
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(path=path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            logger.warning("Failed to load memory state from %s: %s", path, error)
            return cls(path=path)
        if not isinstance(payload, dict) or payload.get("version") != SCHEMA_VERSION:
            logger.warning("Ignoring unsupported memory state at %s", path)
            return cls(path=path)
        raw_notes = payload.get("facts")
        if not isinstance(raw_notes, list):
            logger.warning("Ignoring invalid memory state at %s", path)
            return cls(path=path)
        notes: list[MemoryNote] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        for raw_note in raw_notes:
            note = _memory_note_from_json(raw_note)
            normalized_text = note.text.casefold() if note is not None else ""
            if note is not None and note.id not in seen_ids and normalized_text not in seen_texts:
                notes.append(note)
                seen_ids.add(note.id)
                seen_texts.add(normalized_text)
                if len(notes) == MAX_MEMORY_NOTES:
                    break
        return cls(path=path, notes=notes)

    def remember(self, fact: str, *, replaces_memory_id: str | None = None) -> MemoryChange:
        """Persist one explicit durable fact, optionally replacing an old note."""
        normalized = _validated_memory_text(fact)
        if replaces_memory_id is not None:
            _validated_memory_id(replaces_memory_id)
        with self._lock:
            replaced = next((note for note in self.notes if note.id == replaces_memory_id), None)
            if replaces_memory_id is not None and replaced is None:
                raise ValueError(f"unknown memory id: {replaces_memory_id}")
            existing = next((note for note in self.notes if note.text.casefold() == normalized.casefold()), None)
            if replaces_memory_id is None and existing is not None:
                return MemoryChange(note=existing, status="unchanged")
            if replaced is not None and replaced.text == normalized:
                return MemoryChange(note=replaced, status="unchanged")

            note = MemoryNote(id=_new_memory_id(), text=normalized)
            previous_notes = self.notes
            if replaces_memory_id is None:
                self.notes = [note, *self.notes][:MAX_MEMORY_NOTES]
                status: Literal["saved", "updated"] = "saved"
            else:
                removed_ids = {replaces_memory_id}
                if existing is not None:
                    removed_ids.add(existing.id)
                self.notes = [
                    note,
                    *(saved_note for saved_note in self.notes if saved_note.id not in removed_ids),
                ][:MAX_MEMORY_NOTES]
                status = "updated"
            try:
                self._persist()
            except OSError:
                self.notes = previous_notes
                raise
            self.revision += 1
            return MemoryChange(note=note, status=status)

    def forget(self, memory_id: str) -> MemoryNote | None:
        """Remove one memory by its exact injected identifier."""
        _validated_memory_id(memory_id)
        with self._lock:
            removed = next((note for note in self.notes if note.id == memory_id), None)
            if removed is None:
                return None
            previous_notes = self.notes
            self.notes = [note for note in self.notes if note.id != memory_id]
            try:
                self._persist()
            except OSError:
                self.notes = previous_notes
                raise
            self.revision += 1
            return removed

    def render_for_instructions(self) -> str:
        """Render bounded, escaped memory notes for dynamic instructions."""
        with self._lock:
            if not self.notes:
                return "- (none)"
            return "\n".join(f"- [{note.id}] {html.escape(note.text, quote=False)}" for note in self.notes)

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "facts": [note.to_json() for note in self.notes],
        }
        temporary_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
        try:
            temporary_path.write_text(
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("Failed to remove temporary memory file %s: %s", temporary_path, error)
