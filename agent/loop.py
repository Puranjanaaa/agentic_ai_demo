"""
Core agent loop.

The loop follows the ReAct pattern (Reason + Act):

    ┌─────────────────────────────────────────────────────────────┐
    │  User message arrives                                        │
    │       ↓                                                      │
    │  Load session history  (short-term memory)                   │
    │       ↓                                                      │
    │  Load long-term memory  → inject into system prompt          │
    │       ↓                                                      │
    │  ┌─── LLM call ────────────────────────────────────────┐    │
    │  │  stop_reason == "tool_use"?                          │    │
    │  │    → execute each tool                               │    │
    │  │    → append tool results as new "user" turn          │    │
    │  │    → loop back to LLM call                          │    │
    │  │  stop_reason == "end_turn"?                          │    │
    │  │    → extract text response, exit loop               │    │
    │  └──────────────────────────────────────────────────────┘    │
    │       ↓                                                      │
    │  Persist updated history                                     │
    │       ↓                                                      │
    │  Return response + trace                                     │
    └─────────────────────────────────────────────────────────────┘

Design decisions:
  - MAX_ITERATIONS guards against infinite tool loops (e.g. a buggy tool
    that always triggers another tool call).
  - The system prompt is rebuilt on every run so it always reflects the
    latest memory state without needing a separate "memory retrieval" pass.
  - Tool results are fed back as the 'user' role per the Anthropic multi-turn
    tool-use protocol.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import anthropic

from agent.tools import TOOL_DEFINITIONS, ToolExecutor
from agent.tracer import Tracer
from models.schemas import HistoryMessage, MessageRole
from storage.store import StorageManager

# Safety cap on tool-call iterations per user turn
MAX_ITERATIONS = 10

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _get_model_name() -> str:
    """Resolve the model name from the environment, with a safe default."""
    return (
        os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("MODEL")
        or DEFAULT_MODEL
    )


def _get_base_url() -> str | None:
    """Resolve the Anthropic-compatible base URL and fix common typos."""
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not base_url:
        return None

    base_url = base_url.strip()
    if base_url.startswith("hhttp://") or base_url.startswith("hhttps://"):
        return base_url[1:]

    return base_url


def _build_system_prompt(memory_entries: dict) -> str:
    """
    Construct the system prompt, embedding long-term memory so Claude
    always has user context without an explicit search_memory call.

    We still expose search_memory as a tool so Claude can verify or
    surface specific details during reasoning.
    """
    memory_section = ""
    if memory_entries:
        by_category: dict[str, list[str]] = defaultdict(list)
        for entry in memory_entries.values():
            cat = entry.category or entry.key.split(':')[0]
            by_category[cat].append(entry.value)
        lines = ["## What I know about this user (long-term memory)\n"]
        for cat, values in by_category.items():
            lines.append(f"- **{cat}**: {', '.join(values)}")
        memory_section = "\n".join(lines)
    else:
        memory_section = "## Long-term memory\nNo entries yet."

    return f"""You are a helpful, friendly AI assistant with persistent long-term memory \
and broad general knowledge. You can answer ANY question — factual, scientific, technical, \
creative — using your own knowledge. Memory is only for personal details the user has shared.

{memory_section}

## How to use pre-loaded memory
The section above is VERIFIED personal information from persistent storage. Treat it as ground truth.

Rules:
1. Personal questions (name, job, preferences, projects, goals): answer DIRECTLY from the list
   above. Do NOT call search_memory when the answer is already there.
2. Only call search_memory when the user asks about something personal they may have shared
   before AND it is NOT in the list above.
3. General knowledge questions (science, technology, history, math concepts, etc.): answer
   directly from your own knowledge. Do NOT check memory; do NOT say "I don't have that in memory."
4. Your response MUST be plain conversational text — no JSON or schemas.

## Tone and naturalness — CRITICAL
Respond like a knowledgeable friend, not a system. NEVER:
- Mention memory, storage, or saving. Do NOT say anything like "I saved that", "I've noted that",
  "I'll remember that", "I have that in memory", "I remember that!", or any variation.
- Announce what tool you are about to use. Do NOT say "let me calculate", "I'll use my calculator",
  "I checked the time", "let me look that up", or anything that reveals tool usage.
- Explain where your information comes from ("Based on what you told me...", "According to my records...")
- Use robotic filler ("I'd be happy to help!", "Certainly!", "Of course!")
- Acknowledge uncertainty about internal systems ("I don't have that saved", "I don't have personal info about X")

Instead:
- When the user tells you their name: just greet them naturally. ("Hey Ben!" / "Nice to meet you!")
- When the user shares a preference or hobby: briefly acknowledge it. ("Nice!" / "That's fun!" / "Cool!")
  Do NOT repeat the preference back in a robotic way. Do NOT reference unrelated past preferences.
- For personal facts you know: answer directly. ("Your name is Alex." / "You love pasta.")
- For dates and times: state the answer directly. ("Today is Monday, June 1st.")
- For calculations: just give the answer. ("That comes to $25.04.")
- For general knowledge: answer directly and naturally.
- For things you genuinely don't know: say so simply. ("I'm not sure about that.")

## Rules for every tool
Tools are called silently in the background. The user never sees tool calls or results — only your
final text response. NEVER mention, announce, or allude to any tool in your response.

### save_memory
Call whenever the user tells you something personal about themselves. Use a canonical key:

| key          | save when the user mentions…                                        |
|--------------|---------------------------------------------------------------------|
| `name`       | their name, what to call them                                       |
| `preference` | likes, dislikes, favourites, things they enjoy or hate              |
| `work`       | job, profession, occupation, career, what they do (for a living)    |
| `project`    | app, side-project, anything they are building / developing / making |
| `goal`       | aim, objective, ambition, plan, what they want to achieve           |

After calling save_memory, do NOT mention it. Just respond naturally to the conversation.

### search_memory
Call ONLY when the user asks about personal details they may have shared before AND the
answer is NOT already in the pre-loaded memory list above.

## Other tools
- `calculator`: use for any arithmetic; never compute mentally. Give the result naturally in your response.
- `current_time`: MUST be called before answering any question about the current date, day,
  or time. Never guess — call the tool, then state the answer naturally (e.g. "Today is Monday, June 1st.").
- `summarize_history`: use when the user asks for a recap or summary. Read the history returned
  by the tool and write a concise 3-5 sentence summary in your own words.

Keep responses short and natural. Answer the question directly.
"""


def _infer_memory_updates(user_message: str) -> list[dict[str, str]]:
    """Extract a small set of obvious self-descriptions worth remembering."""
    normalized = " ".join(user_message.strip().split())
    updates: list[dict[str, str]] = []

    def add_update(key: str, value: str, context: str) -> None:
        value = value.strip().rstrip(".,;:!? ")
        if value:
            updates.append({"key": key, "value": value, "context": context})

    patterns: list[tuple[str, str, str]] = [
        # ── name ──────────────────────────────────────────────────────────────
        (r"\bmy name is\s+([^.,;!?]+)", "name", "The user told us their name."),
        (r"\bcall me\s+([^.,;!?]+)", "name", "The user told us what to call them."),
        (r"\bpeople call me\s+([^.,;!?]+)", "name", "The user told us their name."),
        (r"\byou can call me\s+([^.,;!?]+)", "name", "The user told us what to call them."),
        (r"\bi(?:'m| am) known as\s+([^.,;!?]+)", "name", "The user told us their name."),

        # ── goals (aim / objective / ambition / plan) ─────────────────────────
        (r"\bmy goal is\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bmy aim is\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bmy objective is\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bmy ambition is\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bi(?:'m| am) trying to\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bi want to\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bi hope to\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bi plan to\s+([^.,;!?]+)", "goal", "The user described a goal."),

        # ── work (job / profession / occupation / career) ─────────────────────
        (r"\bi work (?:on|in|at|for)\s+([^.,;!?]+)", "work", "The user described their work or industry."),
        (r"\bi work as\s+([^.,;!?]+)", "work", "The user described their profession."),
        (r"\bmy job is\s+([^.,;!?]+)", "work", "The user described their job."),
        (r"\bmy profession is\s+([^.,;!?]+)", "work", "The user described their profession."),
        (r"\bmy occupation is\s+([^.,;!?]+)", "work", "The user described their occupation."),
        (r"\bmy career is\s+([^.,;!?]+)", "work", "The user described their career."),
        (r"\bi(?:'m| am) employed as\s+([^.,;!?]+)", "work", "The user described their job."),
        (r"\bi(?:'m| am) working as\s+([^.,;!?]+)", "work", "The user described their job."),
        (r"\bwhat i do (?:is|for a living is?)\s+([^.,;!?]+)", "work", "The user described what they do."),
        (r"\bi do\s+([^.,;!?]+?)\s+for a living", "work", "The user described what they do for a living."),
        (r"\bi(?:'m| am) a\s+([^.,;!?]+?)\s+(?:by profession|professionally|by trade)", "work", "The user described their profession."),

        # ── projects (app / side-project / thing being built) ─────────────────
        (r"\bmy project is called\s+([^.,;!?]+)", "project", "The user named a project."),
        (r"\bmy secret project is called\s+([^.,;!?]+)", "project", "The user named a project."),
        (r"\bmy current project is\s+([^.,;!?]+)", "project", "The user described their current project."),
        (r"\bi(?:'m| am) building\s+([^.,;!?]+)", "project", "The user described what they are building."),
        (r"\bi(?:'m| am) working on\s+([^.,;!?]+)", "project", "The user described what they are working on."),
        (r"\bi(?:'m| am) developing\s+([^.,;!?]+)", "project", "The user described what they are developing."),
        (r"\bi(?:'m| am) creating\s+([^.,;!?]+)", "project", "The user described what they are creating."),
        (r"\bi(?:'m| am) making\s+([^.,;!?]+)", "project", "The user described what they are making."),

        # ── preferences (likes / dislikes / favorites) ─────────────────────────
        (r"\bi prefer\s+([^.,;!?]+)", "preference", "The user described a preference."),
        (r"\bi love\s+([^.,;!?]+)", "preference", "The user described something they like."),
        (r"\bi like\s+([^.,;!?]+)", "preference", "The user described something they like."),
        (r"\bi enjoy\s+([^.,;!?]+)", "preference", "The user described something they enjoy."),
        (r"\bi dislike\s+([^.,;!?]+)", "preference", "The user described something they dislike."),
        (r"\bi hate\s+([^.,;!?]+)", "preference", "The user described something they dislike."),
        (r"\bmy preference is\s+([^.,;!?]+)", "preference", "The user described a preference."),
        (r"\bmy favou?rite (?:color|colour|food|music|sport|hobby) is\s+([^.,;!?]+)", "preference", "The user described a preference."),
        (r"\bmy favorite (?:color|colour|food|music|sport|hobby) is\s+([^.,;!?]+)", "preference", "The user described a preference."),
    ]

    for pattern, key, context in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            add_update(key, match.group(1), context)

    return updates


def _infer_calculator_expressions(user_message: str) -> list[str]:
    """Extract simple arithmetic expressions that are obvious from the user's wording."""
    normalized = " ".join(user_message.strip().split())
    expressions: list[str] = []

    match = re.search(
        r"budget is\s+(\d+).+?divide(?: it)? among\s+(\d+)\s+teams?",
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        expressions.append(f"{match.group(1)} / {match.group(2)}")

    return expressions



class AgentLoop:
    """
    Stateless callable that runs one user turn through the full agent loop.

    'Stateless' here means the object holds no session state itself —
    all state lives in the StorageManager.  This makes it safe to share
    a single AgentLoop instance across requests.
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        base_url = _get_base_url()
        if base_url:
            self.client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=base_url,
            )
        else:
            self.client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

    def run(self, session_id: str, user_message: str) -> tuple[str, list]:
        """
        Execute one full agent turn.

        Returns:
            (assistant_response_text, trace_steps)
        """
        tracer = Tracer()

        # ── 1. Load short-term history ─────────────────────────────────────
        history = self.storage.load_history(session_id)
        tracer.load_history(len(history.messages))

        # ── 2. Load long-term memory ───────────────────────────────────────
        memory_store = self.storage.load_memory(session_id)
        tracer.load_memory(
            len(memory_store.entries),
            list(memory_store.entries.keys()),
        )

        # ── 3. Build the message list for the API ─────────────────────────
        # Convert stored history to Anthropic message format
        api_messages: list[dict[str, Any]] = [
            {"role": msg.role.value, "content": msg.content}
            for msg in history.messages
        ]
        # Append the new user turn
        api_messages.append({"role": "user", "content": user_message})

        # ── 4. Agentic loop ────────────────────────────────────────────────
        system_prompt = _build_system_prompt(memory_store.entries)
        executor = ToolExecutor(
            storage=self.storage,
            session_id=session_id,
            tracer=tracer,
            messages_snapshot=api_messages,
        )

        final_response = ""
        calculator_results: list[str] = []

        inferred_memory = _infer_memory_updates(user_message)
        if inferred_memory:
            for memory_update in inferred_memory:
                executor.execute("save_memory", memory_update)
            memory_store = self.storage.load_memory(session_id)
            system_prompt = _build_system_prompt(memory_store.entries)

        inferred_calculations = _infer_calculator_expressions(user_message)
        if inferred_calculations:
            for expression in inferred_calculations:
                result_text = executor.execute("calculator", {"expression": expression})
                if result_text:
                    calculator_results.append(result_text)

        model_name = _get_model_name()

        for iteration in range(1, MAX_ITERATIONS + 1):
            tracer.llm_call(iteration, len(api_messages))
            response = self.client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                tools=TOOL_DEFINITIONS,
                messages=api_messages,
            )

            # Always append the assistant turn to maintain valid message history
            api_messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract the text response
                for block in response.content:
                    if hasattr(block, "text"):
                        final_response = block.text
                        break
                break

            if response.stop_reason == "max_tokens":
                # Some local Anthropic-compatible backends stop at the token cap
                # before they emit their final answer or tool-use request.  Treat
                # this as a resumable state and keep looping with the accumulated
                # assistant output so the next pass can continue the turn.
                continue

            if response.stop_reason == "tool_use":
                # Execute all tool calls in this response and collect results
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_text = executor.execute(block.name, block.input)
                        if block.name == "calculator" and result_text:
                            calculator_results.append(result_text)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                # Feed tool results back as a 'user' turn (Anthropic protocol)
                api_messages.append({"role": "user", "content": tool_results})
                # Continue the loop → LLM will reason over tool results
                continue

            # Unexpected stop reason — surface it as an error response
            final_response = f"[Agent stopped unexpectedly: {response.stop_reason}]"
            break

        if not final_response:
            final_response = "[Agent reached max iterations without a final response]"

        if calculator_results and not any(result in final_response for result in calculator_results):
            final_response = f"{final_response}\n\nResult: {calculator_results[-1]}".strip()

        # ── 5. Persist updated history ─────────────────────────────────────
        history.messages.append(
            HistoryMessage(role=MessageRole.USER, content=user_message)
        )
        history.messages.append(
            HistoryMessage(role=MessageRole.ASSISTANT, content=final_response)
        )
        self.storage.save_history(history)
        tracer.save_history(len(history.messages))

        tracer.response(final_response)

        return final_response, tracer.steps()
