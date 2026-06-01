"""
FastAPI route handlers.

Design note:
  Routes are thin — they validate input, delegate to the AgentLoop, and
  format output.  No business logic lives here.  This makes the core agent
  independently testable without standing up the HTTP layer.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from agent.loop import AgentLoop
from models.schemas import (
    ChatRequest,
    ChatResponse,
    GetHistoryResponse,
    GetMemoryResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from storage.store import StorageManager

router = APIRouter()


# ── Dependency injection ───────────────────────────────────────────────────
# Both objects are lightweight singletons.  In a multi-process deployment
# you'd use a proper DI framework or lifespan context.

_storage: StorageManager | None = None
_agent: AgentLoop | None = None


def get_storage() -> StorageManager:
    global _storage
    if _storage is None:
        _storage = StorageManager(base_dir="data")
    return _storage


def get_agent(storage: StorageManager = Depends(get_storage)) -> AgentLoop:
    global _agent
    if _agent is None:
        _agent = AgentLoop(storage=storage)
    return _agent


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post(
    "/sessions",
    response_model=StartSessionResponse,
    summary="Start or resume a session",
    description=(
        "Creates a new session or resumes an existing one. "
        "If `session_id` is omitted a new UUID is generated."
    ),
)
def start_session(
    body: StartSessionRequest,
    storage: StorageManager = Depends(get_storage),
) -> StartSessionResponse:
    if body.session_id:
        is_new = not storage.session_exists(body.session_id)
        session_id = body.session_id
        msg = (
            "Resumed existing session."
            if not is_new
            else "Created new session with provided ID."
        )
    else:
        session_id = str(uuid.uuid4())
        is_new = True
        msg = "Created new session."

    # Ensure the session file exists
    history = storage.load_history(session_id)
    storage.save_history(history)

    return StartSessionResponse(session_id=session_id, is_new=is_new, message=msg)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the agent",
    description=(
        "Runs the full agent loop: loads context, calls the LLM (with tools as needed), "
        "persists history, and returns the response with an execution trace."
    ),
)
def chat(
    body: ChatRequest,
    storage: StorageManager = Depends(get_storage),
    agent: AgentLoop = Depends(get_agent),
) -> ChatResponse:
    if not storage.session_exists(body.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found. Create it first via POST /sessions.",
        )

    response_text, trace_steps = agent.run(
        session_id=body.session_id,
        user_message=body.message,
    )

    return ChatResponse(
        session_id=body.session_id,
        response=response_text,
        trace=trace_steps,
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=GetHistoryResponse,
    summary="Retrieve session message history",
)
def get_history(
    session_id: str,
    storage: StorageManager = Depends(get_storage),
) -> GetHistoryResponse:
    if not storage.session_exists(session_id):
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    history = storage.load_history(session_id)
    return GetHistoryResponse(session_id=session_id, messages=history.messages)


@router.get(
    "/sessions/{session_id}/memory",
    response_model=GetMemoryResponse,
    summary="Retrieve long-term memory for a session",
)
def get_memory(
    session_id: str,
    storage: StorageManager = Depends(get_storage),
) -> GetMemoryResponse:
    memory = storage.load_memory(session_id)
    return GetMemoryResponse(session_id=session_id, memory=memory.entries)


@router.delete(
    "/sessions/{session_id}/memory/{key}",
    summary="Delete a specific memory entry",
)
def delete_memory_entry(
    session_id: str,
    key: str,
    storage: StorageManager = Depends(get_storage),
) -> dict:
    memory = storage.load_memory(session_id)
    if key not in memory.entries:
        raise HTTPException(status_code=404, detail=f"Memory key '{key}' not found.")
    del memory.entries[key]
    storage.save_memory(memory)
    return {"deleted": key, "session_id": session_id}
