# AI Agent

A production-minded AI agent backend with session history, long-term memory, tool use, and execution traces.

## Architecture

```
ai_agent/
├── main.py                  # FastAPI app + uvicorn entry point
├── requirements.txt
├── agent/
│   ├── loop.py              # Core ReAct agent loop
│   ├── tools.py             # Tool definitions + safe execution
│   └── tracer.py            # Per-request execution trace collector
├── api/
│   └── routes.py            # HTTP endpoints (thin — no business logic)
├── models/
│   └── schemas.py           # Pydantic request/response/internal schemas
└── storage/
    └── store.py             # JSON-file persistence (swap for DB here)
```

## Setup

```bash
pip install -r requirements.txt
# Option A: set it in your shell
# export ANTHROPIC_API_KEY=sk-...
#
# Option B (recommended): add it to .env
# ANTHROPIC_API_KEY=sk-...
# Optional: point to a hosted LLM gateway/proxy
# ANTHROPIC_BASE_URL=https://your-llm-host.example.com
uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## API Walkthrough

### 1. Start a session

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{ "session_id": "abc123", "is_new": true, "message": "Created new session." }
```

---

### 2. Chat — agent saves memory automatically

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "My name is Stefan and I work on airport systems."}'
```

Response:
```json
{
  "session_id": "abc123",
  "response": "Nice to meet you, Stefan! I've saved that you work on airport systems — I'll remember that for our future conversations.",
  "trace": [
    { "step": "load_history",  "detail": "Loaded 0 messages from session history" },
    { "step": "load_memory",   "detail": "Injected 0 long-term memory entries into context" },
    { "step": "llm_call",      "detail": "LLM call #1 with 1 messages" },
    { "step": "tool_call",     "detail": "Executed tool: save_memory",
      "data": { "tool": "save_memory", "input": {"key": "name", "value": "Stefan"}, "result": "Saved memory: [name] = \"Stefan\"" } },
    { "step": "tool_call",     "detail": "Executed tool: save_memory",
      "data": { "tool": "save_memory", "input": {"key": "work", "value": "airport systems"}, "result": "Saved memory: [work] = \"airport systems\"" } },
    { "step": "llm_call",      "detail": "LLM call #2 with 5 messages" },
    { "step": "save_history",  "detail": "Persisted session history (2 total messages)" },
    { "step": "response",      "detail": "Generated final response for user" }
  ]
}
```

---

### 3. Continue later — agent recalls memory

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "What systems do I work on?"}'
```

The agent will call `search_memory` and return the correct answer.

---

### 4. Use the calculator tool

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "What is sqrt(1764) + 2 ** 8?"}'
```

---

### 5. Inspect memory and history

```bash
# Long-term memory
curl http://localhost:8000/api/v1/sessions/abc123/memory

# Full conversation history
curl http://localhost:8000/api/v1/sessions/abc123/history
```

---

## Agent Loop (ReAct Pattern)

```
User message
    ↓
Load session history     (short-term / working memory)
    ↓
Load long-term memory    (injected into system prompt)
    ↓
┌── LLM call ──────────────────────────────────────────┐
│  stop_reason == "tool_use"?                           │
│    → execute tools → append results → loop back      │
│  stop_reason == "end_turn"?                           │
│    → extract text response, exit loop                │
└──────────────────────────────────────────────────────┘
    ↓
Persist history
    ↓
Return response + trace
```

## Tools

| Tool | Purpose |
|------|---------|
| `save_memory` | Persist user info (name, goals, projects, preferences) |
| `search_memory` | Keyword search over stored memory |
| `calculator` | Safe AST-based math evaluation (no `eval()`) |
| `current_time` | Return UTC datetime |
| `summarize_history` | Summarise conversation so far |

## Design Decisions

**Why file-based storage?**  
Zero infrastructure dependencies. The `StorageManager` class is the only place that touches the filesystem — replacing it with PostgreSQL/Redis requires changing exactly one file.

**Why inject memory into the system prompt AND expose `search_memory`?**  
The system prompt ensures Claude always has context. The tool lets it explicitly verify or surface specific details during reasoning — two complementary access patterns.

**Why a `Tracer` object per request?**  
A per-request instance means parallel requests can't bleed into each other's traces. It's also trivially serialisable for logging pipelines.

**Why cap the loop at `MAX_ITERATIONS`?**  
Defense against tool-calling cycles. A misbehaving tool that always triggers another call would otherwise run indefinitely.

**Why not `eval()` in the calculator?**  
`eval()` on user-supplied strings is a remote code execution vulnerability. The safe evaluator uses Python's `ast` module to whitelist only numeric operations.
