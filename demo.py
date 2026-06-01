import os
import sys
import uuid
from collections import defaultdict

from dotenv import load_dotenv

from agent.loop import AgentLoop
from models.schemas import TraceStep
from storage.store import StorageManager

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_trace(trace: list[TraceStep]):
    print(f"\n{DIM}  ── agent trace ───────────────────────────────{RESET}")
    for step in trace:
        step_type = step.step.value
        icon = {
            "load_history": "📂",
            "load_memory": "🧠",
            "llm_call": "🤖",
            "tool_call": "🔧",
            "save_history": "💾",
            "save_memory": "📌",
            "response": "✅",
        }.get(step_type, "•")

        detail = step.detail

        if step_type == "tool_call" and step.data:
            tool = step.data.get("tool", "")
            result = step.data.get("result", "")
            detail = f"{tool}  →  {result}"

        print(f"{DIM}  {icon}  {detail}{RESET}")
    print(f"{DIM}  ── end trace ─────────────────────────────────{RESET}\n")


def print_session_trace(all_traces: list[list[TraceStep]]):
    if not all_traces:
        return

    llm_calls = 0
    tool_calls: list[tuple[str, str]] = []

    for trace in all_traces:
        for step in trace:
            step_type = step.step.value
            if step_type == "llm_call":
                llm_calls += 1
            elif step_type == "tool_call" and step.data:
                tool_calls.append(
                    (
                        step.data.get("tool", "?"),
                        step.data.get("result", ""),
                    )
                )

    total_turns = len(all_traces)
    print(f"\n{DIM}  ── session trace ─────────────────────────────{RESET}")
    print(
        f"{DIM}  📊  {total_turns} turn(s)  |  {llm_calls} LLM call(s)  |  {len(tool_calls)} tool call(s){RESET}"
    )
    if tool_calls:
        for name, result in tool_calls:
            snippet = result[:80] + "…" if len(result) > 80 else result
            print(f"{DIM}  🔧  {name}  →  {snippet}{RESET}")
    print(f"{DIM}  ── end session trace ─────────────────────────{RESET}\n")


def print_banner(session_id: str):
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════╗
║          Agentic AI  –  Terminal Chat    ║
╚══════════════════════════════════════════╝{RESET}

{DIM}Session : {session_id}
Commands: 'trace on/off'  |  'memory'  |  'history'  |  'quit'{RESET}
""")


def print_memory(storage: StorageManager, session_id: str):
    mem = storage.load_memory(session_id).entries
    if not mem:
        print(f"{DIM}  (no memory entries yet){RESET}\n")
        return
    by_cat: dict[str, list[str]] = defaultdict(list)
    for key, entry in mem.items():
        cat = entry.category or key.split(":")[0]
        by_cat[cat].append(entry.value)
    print(f"\n{YELLOW}{BOLD}  Long-term memory:{RESET}")
    for cat, values in by_cat.items():
        print(f"{YELLOW}  [{cat}]{RESET}  {', '.join(values)}")
    print()


def print_history(storage: StorageManager, session_id: str):
    msgs = storage.load_history(session_id).messages
    if not msgs:
        print(f"{DIM}  (no history yet){RESET}\n")
        return
    print(f"\n{YELLOW}{BOLD}  Conversation history ({len(msgs)} messages):{RESET}")
    for m in msgs:
        role = "You" if m.role.value == "user" else "AI"
        print(f"{YELLOW}  {role}:{RESET} {m.content[:120]}")
    print()


def main():
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY is not set.")
        print("    Set it in your shell or in a .env file.\n")
        sys.exit(1)

    storage = StorageManager(base_dir="data")
    agent = AgentLoop(storage=storage)
    session_id = str(uuid.uuid4())

    show_trace = False
    all_traces: list[list[TraceStep]] = []
    print_banner(session_id)

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{DIM}Goodbye!{RESET}\n")
            print_session_trace(all_traces)
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(f"\n{DIM}Goodbye!{RESET}\n")
            print_session_trace(all_traces)
            break

        if user_input.lower() == "trace on":
            show_trace = True
            print(f"{DIM}  Trace enabled.{RESET}\n")
            continue

        if user_input.lower() == "trace off":
            show_trace = False
            print(f"{DIM}  Trace disabled.{RESET}\n")
            continue

        if user_input.lower() == "memory":
            print_memory(storage, session_id)
            continue

        if user_input.lower() == "history":
            print_history(storage, session_id)
            continue

        try:
            response, trace = agent.run(session_id=session_id, user_message=user_input)
        except Exception as e:
            print(f"\n❌  Error: {e}\n")
            continue

        all_traces.append(trace)
        print(f"\n{CYAN}{BOLD}AI_ASSISTANT:{RESET} {response}\n")

        if show_trace:
            print_trace(trace)


if __name__ == "__main__":
    main()
