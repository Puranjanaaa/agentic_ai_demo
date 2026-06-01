from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone
from typing import Any

from agent.tracer import Tracer
from storage.store import StorageManager


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": lambda x, n=0: round(x, int(n)),
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCTIONS:
        return _SAFE_FUNCTIONS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        if callable(func):
            args = [_safe_eval(a) for a in node.args]
            return func(*args)
    raise ValueError(f"Unsafe expression node: {ast.dump(node)}")


# Tool definitions for prompt construction and validation

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "save_memory",
        "description": (
            "Persist important user information to long-term memory. "
            "Use whenever the user shares their name, preferences, goals, projects, "
            "or any detail worth remembering across future sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Canonical category. Use one of: "
                        "'name' (what to call the user), "
                        "'preference' (likes, dislikes, favorites — call once per preference), "
                        "'work' (job, profession, career — call once per job/role), "
                        "'project' (app or thing they are building — call once per project), "
                        "'goal' (aim, objective, plan — call once per goal). "
                        "Each call adds a new entry; multiple values per category are stored separately."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The information to save.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: why this information is relevant.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search the user's long-term memory for relevant information. "
            "Use before answering questions that might rely on what the user told you before."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or topic to search for in memory.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression safely. "
            "Supports +, -, *, /, **, %, //, and functions: sqrt, abs, round, "
            "floor, ceil, log, log2, log10, sin, cos, tan. Constants: pi, e."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g. 'sqrt(144) + 2 ** 10'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "current_time",
        "description": "Return the current UTC date and time. Use when the user asks about the time or date.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "summarize_history",
        "description": (
            "Produce a concise summary of the current conversation. "
            "Useful when the user asks for a recap, or before performing complex reasoning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Optional: a specific aspect to focus on, e.g. 'decisions made' or 'open questions'.",
                },
            },
            "required": [],
        },
    },
]


# ToolExecutor class to route tool calls to implementations and handle side effects


class ToolExecutor:
    def __init__(
        self,
        storage: StorageManager,
        session_id: str,
        tracer: Tracer,
        messages_snapshot: list[dict[str, Any]] | None = None,
    ) -> None:
        self.storage = storage
        self.session_id = session_id
        self.tracer = tracer
        self._messages = messages_snapshot or []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        handlers = {
            "save_memory": self._save_memory,
            "search_memory": self._search_memory,
            "calculator": self._calculator,
            "current_time": self._current_time,
            "summarize_history": self._summarize_history,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            result = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                result = handler(**tool_input)
            except Exception as exc:
                result = f"Tool error: {exc}"

        self.tracer.tool_call(tool_name, tool_input, result)
        return result

    def _save_memory(self, key: str, value: str, context: str | None = None) -> str:
        entry = self.storage.upsert_memory_entry(self.session_id, key, value, context)
        action = "Updated" if entry.updated_at != entry.saved_at else "Saved"
        return f'{action} memory: [{key}] = "{value}"'

    def _search_memory(self, query: str) -> str:
        results = self.storage.search_memory_entries(self.session_id, query)
        if not results:
            return f"No memory entries found matching: {query!r}"
        lines = [f"Memory entries matching '{query}' ({len(results)} entries):"]
        for e in results:
            ctx = f" ({e.context})" if e.context else ""
            lines.append(f'  [{e.key}] = "{e.value}"{ctx}')
        return "\n".join(lines)

    def _calculator(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(round(result, 10))
        except Exception as exc:
            return f"Calculation error: {exc}"

    def _current_time(self) -> str:
        now = datetime.now(timezone.utc)
        return (
            f"Current UTC date: {now:%Y-%m-%d} | "
            f"Current UTC time: {now:%H:%M:%S} | "
            f"Day: {now:%A}"
        )

    def _summarize_history(self, focus: str | None = None) -> str:
        if not self._messages:
            return "No conversation history yet."
        lines: list[str] = []
        for msg in self._messages:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                )
            if content:
                lines.append(f"{role}: {content[:200]}")
        summary = "\n".join(lines[-20:])  # last 20 turns
        if focus:
            summary = f"[Focus: {focus}]\n" + summary
        return summary
