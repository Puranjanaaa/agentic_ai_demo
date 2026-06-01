from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TraceStepType(str, Enum):
    LOAD_HISTORY = "load_history"
    LOAD_MEMORY = "load_memory"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    SAVE_HISTORY = "save_history"
    SAVE_MEMORY = "save_memory"
    RESPONSE = "response"


class TraceStep(BaseModel):
    step: TraceStepType
    detail: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryEntry(BaseModel):
    key: str
    category: str | None = None  # canonical category ("preference", "work", etc.)
    value: str
    context: str | None = None
    saved_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryStore(BaseModel):
    session_id: str
    entries: dict[str, MemoryEntry] = Field(default_factory=dict)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class HistoryMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionHistory(BaseModel):
    session_id: str
    messages: list[HistoryMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
