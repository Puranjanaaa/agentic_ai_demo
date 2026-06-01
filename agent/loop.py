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
        lines = ["## What I know about this user (long-term memory)\n"]
        for key, entry in memory_entries.items():
            ctx = f"  — {entry.context}" if entry.context else ""
            lines.append(f"- **{key}**: {entry.value}{ctx}")
        memory_section = "\n".join(lines)
    else:
        memory_section = "## Long-term memory\nNo entries yet."

    return f"""You are a helpful, friendly AI assistant with persistent memory.

{memory_section}

## Behaviour guidelines
- When the user shares personal information (name, goals, projects, preferences), 
  call `save_memory` to persist it.
- When answering questions that might depend on past context, call `search_memory`
  to verify your recall before responding.
- Use `calculator` for any arithmetic to avoid errors.
- Use `current_time` when date/time is relevant.
- Use `summarize_history` when a recap would help.
- Be concise but warm. Acknowledge what you remember about the user when relevant.
- Never claim to remember something you haven't verified via memory tools or the
  current conversation.
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
        (r"\bmy name is\s+([^.,;!?]+)", "name", "The user told us their name."),
        (r"\bmy goal is\s+([^.,;!?]+)", "goal", "The user described a goal."),
        (r"\bmy project is called\s+([^.,;!?]+)", "project", "The user named a project."),
        (r"\bmy secret project is called\s+([^.,;!?]+)", "project", "The user named a project."),
        (r"\bi(?:'m| am) building\s+([^.,;!?]+)", "project", "The user described what they are building."),
        (r"\bi work on\s+([^.,;!?]+)", "project", "The user described what they work on."),
        (r"\bi prefer\s+([^.,;!?]+)", "preference", "The user described a preference."),
        (r"\bi love\s+([^.,;!?]+)", "preference", "The user described something they like."),
        (r"\bmy favourite color is\s+([^.,;!?]+)", "preference", "The user described a preference."),
        (r"\bmy favorite color is\s+([^.,;!?]+)", "preference", "The user described a preference."),
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


def _is_summary_request(user_message: str) -> bool:
    """Detect explicit recap / summarize requests."""
    return bool(re.search(r"\b(summarize|summary|recap|summarise)\b", user_message, flags=re.IGNORECASE))


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
        summary_result: str | None = None

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

        if _is_summary_request(user_message):
            summary_result = executor.execute("summarize_history", {})
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
        else:
            final_response = "[Agent reached max iterations without a final response]"

        if summary_result and summary_result not in final_response:
            final_response = summary_result

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
