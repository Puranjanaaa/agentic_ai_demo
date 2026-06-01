"""
Pydantic schemas for the AI Agent API.

Design note: Keeping request/response models separate from internal data models
gives us clean API contracts that can evolve independently from storage format.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class TraceStepType(str, Enum):
    LOAD_HISTORY = "load_history"
    LOAD_MEMORY  = "load_memory"
    LLM_CALL     = "llm_call"
    TOOL_CALL    = "tool_call"
    SAVE_HISTORY = "save_history"
    SAVE_MEMORY  = "save_memory"
    RESPONSE     = "response"


# ── Trace ──────────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    """A single observable step in the agent's reasoning loop."""
    step: TraceStepType
    detail: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Memory ─────────────────────────────────────────────────────────────────

class MemoryEntry(BaseModel):
    key: str
    value: str
    context: str | None = None
    saved_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryStore(BaseModel):
    session_id: str
    entries: dict[str, MemoryEntry] = Field(default_factory=dict)


# ── History ────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER      = "user"
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


# ── API Request / Response ─────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Optional. Supply to resume an existing session; omit to create a new one.",
    )


class StartSessionResponse(BaseModel):
    session_id: str
    is_new: bool
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    trace: list[TraceStep]


class GetHistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class GetMemoryResponse(BaseModel):
    session_id: str
    memory: dict[str, MemoryEntry]
