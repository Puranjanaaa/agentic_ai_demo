from __future__ import annotations

from typing import Any

from models.schemas import TraceStep, TraceStepType


class Tracer:
    def __init__(self) -> None:
        self._steps: list[TraceStep] = []

    def load_history(self, message_count: int) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.LOAD_HISTORY,
                detail=f"Loaded {message_count} messages from session history",
                data={"message_count": message_count},
            )
        )

    def load_memory(self, entry_count: int, keys: list[str]) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.LOAD_MEMORY,
                detail=f"Injected {entry_count} long-term memory entries into context",
                data={"entry_count": entry_count, "keys": keys},
            )
        )

    def llm_call(self, iteration: int, message_count: int) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.LLM_CALL,
                detail=f"LLM call #{iteration} with {message_count} messages",
                data={"iteration": iteration, "message_count": message_count},
            )
        )

    def tool_call(
        self, tool_name: str, tool_input: dict[str, Any], tool_result: Any
    ) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.TOOL_CALL,
                detail=f"Executed tool: {tool_name}",
                data={"tool": tool_name, "input": tool_input, "result": tool_result},
            )
        )

    def save_history(self, message_count: int) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.SAVE_HISTORY,
                detail=f"Persisted session history ({message_count} total messages)",
                data={"message_count": message_count},
            )
        )

    def response(self, preview: str) -> None:
        self._steps.append(
            TraceStep(
                step=TraceStepType.RESPONSE,
                detail="Generated final response for user",
                data={
                    "preview": preview[:120] + "…" if len(preview) > 120 else preview
                },
            )
        )

    def steps(self) -> list[TraceStep]:
        return list(self._steps)
