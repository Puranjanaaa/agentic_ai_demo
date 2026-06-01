"""
Agentic AI – Terminal Chat Demo

Usage:
    python demo.py

Requirements:
    - Server running at http://localhost:8000
    - pip install requests
"""

import requests
import sys
from collections import defaultdict

BASE = "http://localhost:8000/api/v1"

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def create_session() -> str:
    r = requests.post(f"{BASE}/sessions", json={})
    r.raise_for_status()
    return r.json()["session_id"]


def chat(session_id: str, message: str) -> dict:
    r = requests.post(
        f"{BASE}/chat", json={"session_id": session_id, "message": message}
    )
    r.raise_for_status()
    return r.json()


def print_trace(trace: list):
    print(f"\n{DIM}  ── agent trace ───────────────────────────────{RESET}")
    for step in trace:
        icon = {
            "load_history": "📂",
            "load_memory": "🧠",
            "llm_call": "🤖",
            "tool_call": "🔧",
            "save_history": "💾",
            "save_memory": "📌",
            "response": "✅",
        }.get(step["step"], "•")

        detail = step["detail"]

        # For tool calls, show tool name + result inline
        if step["step"] == "tool_call" and step.get("data"):
            tool = step["data"].get("tool", "")
            result = step["data"].get("result", "")
            detail = f"{tool}  →  {result}"

        print(f"{DIM}  {icon}  {detail}{RESET}")
    print(f"{DIM}  ── end trace ─────────────────────────────────{RESET}\n")


def print_session_trace(all_traces: list[list]):
    """Print a compact summary of everything the agent did across the session."""
    if not all_traces:
        return

    llm_calls = 0
    tool_calls: list[tuple[str, str]] = []  # (tool_name, result)

    for trace in all_traces:
        for step in trace:
            if step["step"] == "llm_call":
                llm_calls += 1
            elif step["step"] == "tool_call" and step.get("data"):
                tool_calls.append(
                    (
                        step["data"].get("tool", "?"),
                        step["data"].get("result", ""),
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


def print_memory(session_id: str):
    r = requests.get(f"{BASE}/sessions/{session_id}/memory")
    mem = r.json().get("memory", {})
    if not mem:
        print(f"{DIM}  (no memory entries yet){RESET}\n")
        return
    by_cat: dict[str, list[str]] = defaultdict(list)
    for key, entry in mem.items():
        cat = entry.get("category") or key.split(":")[0]
        by_cat[cat].append(entry["value"])
    print(f"\n{YELLOW}{BOLD}  Long-term memory:{RESET}")
    for cat, values in by_cat.items():
        print(f"{YELLOW}  [{cat}]{RESET}  {', '.join(values)}")
    print()


def print_history(session_id: str):
    r = requests.get(f"{BASE}/sessions/{session_id}/history")
    msgs = r.json().get("messages", [])
    if not msgs:
        print(f"{DIM}  (no history yet){RESET}\n")
        return
    print(f"\n{YELLOW}{BOLD}  Conversation history ({len(msgs)} messages):{RESET}")
    for m in msgs:
        role = "You" if m["role"] == "user" else "AI"
        print(f"{YELLOW}  {role}:{RESET} {m['content'][:120]}")
    print()


def main():
    # Check server is up
    try:
        requests.get("http://localhost:8000/health", timeout=3).raise_for_status()
    except Exception:
        print("\n❌  Cannot reach the server at http://localhost:8000")
        print("    Start it with:  uvicorn main:app --reload --port 8000\n")
        sys.exit(1)

    session_id = create_session()
    show_trace = False
    all_traces: list[list] = []
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

        # ── built-in commands ──────────────────────────────────────────────
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
            print_memory(session_id)
            continue

        if user_input.lower() == "history":
            print_history(session_id)
            continue

        # ── send to agent ──────────────────────────────────────────────────
        try:
            result = chat(session_id, user_input)
        except requests.HTTPError as e:
            print(f"\n❌  Error: {e}\n")
            continue

        trace = result.get("trace", [])
        all_traces.append(trace)

        print(f"\n{CYAN}{BOLD}AI_ASSISTANT:{RESET} {result['response']}\n")

        if show_trace:
            print_trace(trace)


if __name__ == "__main__":
    main()
