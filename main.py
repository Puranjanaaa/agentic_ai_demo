"""
AI Agent – FastAPI + uvicorn entry point.

Run locally:
    # Option A: export in your shell
    # export ANTHROPIC_API_KEY=sk-...
    #
    # Option B (recommended): set it in .env
    # ANTHROPIC_API_KEY=sk-...
    uvicorn main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router

# Load local environment variables from .env for development
load_dotenv()

# ── Guard: fail fast if the API key is missing ─────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY environment variable is not set. "
        "Set it in your shell or in a .env file before starting the server."
    )

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Agent API",
    description=(
        "A simple but production-minded AI agent with short-term session history, "
        "long-term memory, tool usage, and per-request execution traces."
    ),
    version="1.0.0",
)

# Allow all origins in development; tighten this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["agent"])


@app.get("/health", include_in_schema=False)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ── Dev runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # auto-reload on file changes during development
        log_level="info",
    )
