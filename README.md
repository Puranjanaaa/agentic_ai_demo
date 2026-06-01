# Agentic AI Demo

A terminal-based AI chat agent with persistent long-term memory, per-session conversation history, tool use, and a per-turn execution tracer. Built on the Anthropic SDK using a ReAct-style agentic loop.

---

## Architecture

```
agentic_ai_demo/
├── demo.py                  # Terminal chat REPL — entry point
├── requirements.txt
├── .env                     # API credentials (not committed)
├── agent/
│   ├── loop.py              # Core ReAct agentic loop (AgentLoop)
│   ├── tools.py             # Tool definitions + safe executor (ToolExecutor)
│   └── tracer.py            # Per-turn execution trace collector (Tracer)
├── models/
│   └── schemas.py           # Pydantic schemas (TraceStep, MemoryEntry, SessionHistory, …)
└── storage/
    └── store.py             # JSON-file persistence layer (StorageManager)

data/
├── memory/
│   └── global.json          # Long-term memory — shared across all sessions
└── sessions/
    └── <session-id>.json    # Per-session conversation history
```

### How the agent loop works

Each user message triggers a full ReAct cycle:

```
User message
    │
    ├─ Regex pre-scan: extract memory updates, math expressions, time queries, recap requests
    │      └─ Execute matching tools immediately (before the first LLM call)
    │
    ├─ Load session history       (short-term / working memory)
    ├─ Load long-term memory      (injected into system prompt)
    │
    └─ Agentic loop (max 10 iterations)
           │
           ├─ LLM call (Anthropic messages API)
           │
           ├─ stop_reason == "tool_use"
           │      └─ ToolExecutor dispatches each tool → appends results → loops
           │
           └─ stop_reason == "end_turn"
                  └─ Extract text response → persist history → return response + trace
```

**Memory is global, history is per-session.** Facts the user shares (name, preferences, projects) are written to `data/memory/global.json` and reloaded into the system prompt on every turn. Conversation messages are stored per UUID session in `data/sessions/`.

---

## Available Tools

| Tool | Input | What it does |
|------|-------|-------------|
| `save_memory` | `key: str` (required) — canonical category: `name`, `preference`, `work`, `project`, or `goal`<br>`value: str` (required) — the fact to store<br>`context: str` (optional) — why this is relevant | Persists a user fact. Entries are keyed by `category:value-slug` so multiple values per category accumulate without overwriting each other. |
| `search_memory` | `query: str` (required) — keywords or topic to look up | Keyword search over stored memory entries. Used when the pre-loaded system-prompt memory doesn't already answer the question. |
| `calculator` | `expression: str` (required) — e.g. `sqrt(144) + 2 ** 10` | Safe AST-based math evaluator. Supports `+`, `-`, `*`, `/`, `**`, `%`, `//`, and functions: `sqrt`, `abs`, `round`, `floor`, `ceil`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`. Constants: `pi`, `e`. Never calls `eval()`. |
| `current_time` | _(none)_ | Returns the current UTC date, time, and weekday. |
| `summarize_history` | `focus: str` (optional) — aspect to focus on, e.g. `"decisions made"` | Serialises the last 20 turns of the conversation for the LLM to summarise. |

---

## Setup

### 1. Configure environment

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY= <api_key>
ANTHROPIC_BASE_URL= <base_url>
```

### 2. Create data directories

```bash
mkdir -p data/memory data/sessions
```

### 3. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Run

```bash
python demo.py
```

---

## In-Chat Terminal Commands


| Command | Effect |
|---------|--------|
| `trace on` | Show per-turn agent trace after each response |
| `trace off` | Hide the trace |
| `memory` | Print all long-term memory entries grouped by category |
| `history` | Print full conversation history for this session |
| `quit` | Exit and print a session-level summary (total turns, LLM calls, tool calls) |
| Ctrl-C / Ctrl-D | Same as `quit` |

---

## Design Notes

**File-based storage** — zero infrastructure dependencies. `StorageManager` is the only place that touches the filesystem; swapping it for a database means changing one file.

**Memory injected into the system prompt AND exposed as a tool** — the system prompt gives the model instant access to known facts; `search_memory` lets it explicitly verify or surface specific details during reasoning. Two complementary access patterns.

**Regex pre-scan before the first LLM call** — common patterns (name declarations, arithmetic, time queries) are detected client-side and the relevant tools are called first. This means the model's first response already has accurate results in context, cutting one round-trip.

**Per-request `Tracer`** — a fresh instance per turn means parallel calls can't bleed into each other's traces, and the trace is trivially serialisable.

**`MAX_ITERATIONS = 10`** — guards against tool-calling cycles where a misbehaving tool always triggers another call.


