"""
Simple JSON-file persistence layer.

Design rationale:
  - For an interview / prototype, file-based storage is fast to reason about
    and has zero infrastructure dependencies.
  - The StorageManager is a thin wrapper so callers never touch the filesystem
    directly — swapping to PostgreSQL/Redis later only requires changing this file.
  - We store sessions and memory under separate subdirectories so they can be
    independently archived, backed-up, or migrated to different stores.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from models.schemas import MemoryEntry, MemoryStore, SessionHistory


def _json_default(obj: Any) -> Any:
    """JSON serialiser that handles datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


class StorageManager:
    """
    Thin file-system abstraction for session history and long-term memory.

    Thread-safety note: For production you'd wrap reads/writes in a lock or
    use an async-capable store (e.g. aiosqlite).  Here we keep it synchronous
    since the interview focus is on agent architecture, not I/O concurrency.
    """

    def __init__(self, base_dir: str = "data") -> None:
        self.sessions_dir = Path(base_dir) / "sessions"
        self.memory_dir = Path(base_dir) / "memory"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ── internal helpers ───────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _memory_path(self, session_id: str) -> Path:
        # Memory is global — shared across all sessions so facts persist
        # between conversations.  Session history remains per-session.
        return self.memory_dir / "global.json"

    def _read(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)

    # ── session history ────────────────────────────────────────────────────

    def load_history(self, session_id: str) -> SessionHistory:
        raw = self._read(self._session_path(session_id))
        if raw is None:
            return SessionHistory(session_id=session_id)
        return SessionHistory.model_validate(raw)

    def save_history(self, history: SessionHistory) -> None:
        history.updated_at = datetime.utcnow()
        self._write(self._session_path(history.session_id), history.model_dump())

    def session_exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()

    # ── long-term memory ───────────────────────────────────────────────────

    def load_memory(self, session_id: str) -> MemoryStore:
        raw = self._read(self._memory_path(session_id))
        if raw is None:
            return MemoryStore(session_id="global")
        return MemoryStore.model_validate(raw)

    def save_memory(self, memory: MemoryStore) -> None:
        self._write(self._memory_path(memory.session_id), memory.model_dump())

    @staticmethod
    def _value_slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower().strip())[:40].strip("_")

    def upsert_memory_entry(
        self, session_id: str, key: str, value: str, context: str | None = None
    ) -> MemoryEntry:
        memory = self.load_memory(session_id)
        now = datetime.utcnow()

        compound_key = f"{key}:{self._value_slug(value)}"

        # Migrate any old-format entry for this category (plain key, no ':')
        if key in memory.entries:
            old_entry = memory.entries.pop(key)
            old_compound = f"{key}:{self._value_slug(old_entry.value)}"
            old_entry.category = key
            old_entry.key = old_compound
            memory.entries[old_compound] = old_entry

        if compound_key in memory.entries:
            entry = memory.entries[compound_key]
            entry.value = value
            entry.context = context or entry.context
            entry.updated_at = now
        else:
            entry = MemoryEntry(
                key=compound_key,
                category=key,
                value=value,
                context=context,
                saved_at=now,
                updated_at=now,
            )
            memory.entries[compound_key] = entry

        self.save_memory(memory)
        return entry

    def search_memory_entries(self, session_id: str, query: str) -> list[MemoryEntry]:
        """
        Keyword search over memory keys + values.

        Splits the query into individual words so natural-language queries like
        "what's my name?" correctly match the entry with key "name".
        Falls back to returning all entries so the LLM always has context.

        Production upgrade path: embed entries with a small model (e.g. text-embedding-3-small)
        and do cosine similarity search for semantic recall.
        """
        memory = self.load_memory(session_id)
        if not memory.entries:
            return []

        query_lower = query.lower()
        # Extract meaningful words (length > 2, skip common stop words)
        _STOP_WORDS = {
            "the",
            "and",
            "for",
            "are",
            "was",
            "what",
            "who",
            "when",
            "where",
            "how",
            "did",
            "does",
            "have",
            "has",
            "its",
            "my",
            "you",
            "your",
            "me",
            "is",
            "it",
            "do",
            "not",
            "any",
        }
        keywords = [
            w
            for w in query_lower.replace("'", "").split()
            if len(w) > 2 and w not in _STOP_WORDS
        ]

        results = []
        for entry in memory.entries.values():
            cat = entry.category or entry.key.split(":")[0]
            entry_text = (
                f"{cat.lower()} {entry.value.lower()} {(entry.context or '').lower()}"
            )
            if query_lower in entry_text or any(kw in entry_text for kw in keywords):
                results.append(entry)

        # If nothing matched, return all entries so the LLM always has context
        if not results:
            results = list(memory.entries.values())

        return results
