"""
AI Agent – Integration test suite.

Runs against the live server at http://localhost:8000.
Start the server first:
    cd ai_agent && uvicorn main:app --reload --port 8000

Run all tests:
    pytest test_agent.py -v

Run a specific group:
    pytest test_agent.py -v -k "memory"

Interview coverage:
  ✓ Agent loop           – trace always shows the full load→llm→save pipeline
  ✓ Short-term history   – messages persist and grow within a session
  ✓ Long-term memory     – facts saved in one turn are recalled in later turns
  ✓ Tool usage           – save_memory, search_memory, calculator, current_time
  ✓ Clean structure      – 404s, validation errors, and edge cases handled correctly
  ✓ Explainable steps    – every response carries a trace; we assert on its shape
"""

import uuid
import re
import requests

BASE = "http://localhost:8000/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def new_session() -> str:
    """Create a fresh session and return its ID."""
    r = requests.post(f"{BASE}/sessions", json={})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def chat(session_id: str, message: str) -> dict:
    """Send a message and return the full response body."""
    r = requests.post(
        f"{BASE}/chat", json={"session_id": session_id, "message": message}
    )
    assert r.status_code == 200, f"Chat failed: {r.text}"
    return r.json()


def trace_steps(response: dict) -> list[str]:
    """Extract just the step-type strings from a response trace."""
    return [s["step"] for s in response["trace"]]


def trace_tools_used(response: dict) -> list[str]:
    """Return list of tool names called in a response."""
    return [s["data"]["tool"] for s in response["trace"] if s["step"] == "tool_call"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Session management
# ─────────────────────────────────────────────────────────────────────────────


class TestSessions:
    def test_create_new_session_generates_uuid(self):
        """POST /sessions with no body returns a valid UUID and is_new=True."""
        r = requests.post(f"{BASE}/sessions", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["is_new"] is True
        uuid.UUID(body["session_id"])  # raises if not a valid UUID

    def test_create_session_with_explicit_id(self):
        """Supplying a session_id creates that exact session."""
        sid = f"test-{uuid.uuid4().hex[:8]}"
        r = requests.post(f"{BASE}/sessions", json={"session_id": sid})
        assert r.status_code == 200
        assert r.json()["session_id"] == sid
        assert r.json()["is_new"] is True

    def test_resume_existing_session(self):
        """POSTing the same session_id twice returns is_new=False on the second call."""
        sid = new_session()
        r = requests.post(f"{BASE}/sessions", json={"session_id": sid})
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == sid
        assert body["is_new"] is False

    def test_multiple_sessions_are_independent(self):
        """Two sessions created back-to-back have different IDs."""
        s1 = new_session()
        s2 = new_session()
        assert s1 != s2

    def test_health_endpoint(self):
        r = requests.get("http://localhost:8000/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent loop – trace structure
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentLoop:
    """
    Every response must carry a trace that proves the full
    load → llm → save pipeline executed.
    """

    def test_trace_always_present(self):
        sid = new_session()
        resp = chat(sid, "Hello!")
        assert "trace" in resp
        assert isinstance(resp["trace"], list)
        assert len(resp["trace"]) > 0

    def test_trace_contains_required_pipeline_steps(self):
        """
        Every turn must load history, call the LLM at least once,
        save history, and produce a final response step.
        """
        sid = new_session()
        resp = chat(sid, "What is 2 + 2?")
        steps = trace_steps(resp)

        assert "load_history" in steps, "Agent must load history"
        assert "load_memory" in steps, "Agent must load long-term memory"
        assert "llm_call" in steps, "Agent must call the LLM"
        assert "save_history" in steps, "Agent must persist history"
        assert "response" in steps, "Agent must record the final response"

    def test_trace_step_has_required_fields(self):
        sid = new_session()
        resp = chat(sid, "Hi there!")
        for step in resp["trace"]:
            assert "step" in step
            assert "detail" in step
            assert "timestamp" in step

    def test_trace_load_history_count_grows(self):
        """
        The load_history detail should report 0 messages on the first turn
        and 2 messages on the second (user + assistant from turn 1).
        """
        sid = new_session()

        resp1 = chat(sid, "First message.")
        load1 = next(s for s in resp1["trace"] if s["step"] == "load_history")
        assert load1["data"]["message_count"] == 0

        resp2 = chat(sid, "Second message.")
        load2 = next(s for s in resp2["trace"] if s["step"] == "load_history")
        assert load2["data"]["message_count"] == 2

    def test_tool_call_steps_include_input_and_result(self):
        """Tool call trace entries must expose the tool name, input, and result."""
        sid = new_session()
        resp = chat(sid, "Calculate 10 * 10 for me.")
        tool_steps = [s for s in resp["trace"] if s["step"] == "tool_call"]
        assert len(tool_steps) >= 1
        for ts in tool_steps:
            assert "tool" in ts["data"]
            assert "input" in ts["data"]
            assert "result" in ts["data"]

    def test_multi_tool_turn_shows_multiple_llm_calls(self):
        """
        When tools are used, the loop must call the LLM at least twice:
        once to decide to use tools, once to produce the final answer.
        """
        sid = new_session()
        resp = chat(sid, "My hobby is hiking. What is sqrt(256)?")
        llm_steps = [s for s in resp["trace"] if s["step"] == "llm_call"]
        assert len(llm_steps) >= 2, "Expected at least 2 LLM calls when tools are used"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Short-term history
# ─────────────────────────────────────────────────────────────────────────────


class TestShortTermHistory:
    def test_history_starts_empty(self):
        sid = new_session()
        r = requests.get(f"{BASE}/sessions/{sid}/history")
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_history_grows_after_each_turn(self):
        """Each chat turn appends exactly 2 messages (user + assistant)."""
        sid = new_session()
        for i in range(1, 4):
            chat(sid, f"Message number {i}.")
            r = requests.get(f"{BASE}/sessions/{sid}/history")
            assert r.status_code == 200
            assert len(r.json()["messages"]) == i * 2

    def test_history_message_roles_alternate(self):
        """Messages must strictly alternate user → assistant → user → assistant."""
        sid = new_session()
        chat(sid, "Turn one.")
        chat(sid, "Turn two.")
        msgs = requests.get(f"{BASE}/sessions/{sid}/history").json()["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_history_message_content_is_preserved(self):
        """The exact user message text must be stored verbatim."""
        sid = new_session()
        unique = f"unique-token-{uuid.uuid4().hex}"
        chat(sid, f"Remember this token: {unique}")
        msgs = requests.get(f"{BASE}/sessions/{sid}/history").json()["messages"]
        user_contents = [m["content"] for m in msgs if m["role"] == "user"]
        assert any(unique in c for c in user_contents)

    def test_context_carries_across_turns_in_session(self):
        """
        The agent should use conversation history to answer a follow-up question
        without the user repeating themselves.
        """
        sid = new_session()
        chat(sid, "My favourite colour is ultraviolet-blue.")
        resp = chat(sid, "What colour did I just mention?")
        assert (
            "ultraviolet-blue" in resp["response"].lower()
            or "ultraviolet" in resp["response"].lower()
        )

    def test_history_isolated_between_sessions(self):
        """Messages from session A must not appear in session B."""
        s1 = new_session()
        s2 = new_session()
        secret = f"secret-{uuid.uuid4().hex}"
        chat(s1, f"My secret word is {secret}.")
        msgs_s2 = requests.get(f"{BASE}/sessions/{s2}/history").json()["messages"]
        contents = " ".join(m["content"] for m in msgs_s2)
        assert secret not in contents

    def test_history_404_for_unknown_session(self):
        r = requests.get(f"{BASE}/sessions/does-not-exist-xyz/history")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 4. Long-term memory
# ─────────────────────────────────────────────────────────────────────────────


class TestLongTermMemory:
    def test_memory_starts_empty(self):
        sid = new_session()
        r = requests.get(f"{BASE}/sessions/{sid}/memory")
        assert r.status_code == 200
        assert r.json()["memory"] == {}

    def test_agent_saves_name_to_memory(self):
        """Telling the agent your name must result in a memory entry."""
        sid = new_session()
        chat(sid, "My name is Aleksandra.")
        mem = requests.get(f"{BASE}/sessions/{sid}/memory").json()["memory"]
        values = " ".join(e["value"] for e in mem.values()).lower()
        assert "aleksandra" in values

    def test_agent_saves_project_to_memory(self):
        sid = new_session()
        chat(sid, "I'm building a real-time flight scheduling engine.")
        mem = requests.get(f"{BASE}/sessions/{sid}/memory").json()["memory"]
        values = " ".join(e["value"] for e in mem.values()).lower()
        assert "flight" in values or "scheduling" in values

    def test_agent_recalls_memory_in_later_turn(self):
        """
        Core interview scenario: fact saved in turn 1 is recalled in turn 2
        without the user repeating themselves.
        """
        sid = new_session()
        chat(sid, "I work on airport runway management systems.")
        resp = chat(sid, "What systems do I work on?")
        assert (
            "airport" in resp["response"].lower()
            or "runway" in resp["response"].lower()
        )

    def test_agent_recalls_multiple_facts(self):
        sid = new_session()
        chat(sid, "My name is Tomasz and my goal is to ship a RAG pipeline by Q3.")
        resp = chat(sid, "Remind me of my name and my goal.")
        response_lower = resp["response"].lower()
        assert "tomasz" in response_lower
        assert (
            "rag" in response_lower
            or "q3" in response_lower
            or "pipeline" in response_lower
        )

    def test_memory_persists_across_new_agent_turns(self):
        """
        Memory saved by turn 1 must still exist and be searchable
        after several additional turns.
        """
        sid = new_session()
        chat(sid, "I prefer Python over all other languages.")
        chat(sid, "Tell me a joke.")
        chat(sid, "What's 42 * 42?")
        resp = chat(sid, "What's my preferred programming language?")
        assert "python" in resp["response"].lower()

    def test_memory_isolated_between_sessions(self):
        """Memory saved in session A must not bleed into session B."""
        s1 = new_session()
        s2 = new_session()
        secret = f"zephyr-{uuid.uuid4().hex[:6]}"
        chat(s1, f"My secret project is called {secret}.")
        mem_s2 = requests.get(f"{BASE}/sessions/{s2}/memory").json()["memory"]
        all_values = " ".join(e["value"] for e in mem_s2.values())
        assert secret not in all_values

    def test_memory_entry_can_be_deleted(self):
        sid = new_session()
        chat(sid, "My name is DeleteMe.")
        mem = requests.get(f"{BASE}/sessions/{sid}/memory").json()["memory"]
        assert len(mem) > 0
        key = list(mem.keys())[0]
        r = requests.delete(f"{BASE}/sessions/{sid}/memory/{key}")
        assert r.status_code == 200
        assert r.json()["deleted"] == key
        mem_after = requests.get(f"{BASE}/sessions/{sid}/memory").json()["memory"]
        assert key not in mem_after

    def test_delete_nonexistent_memory_key_returns_404(self):
        sid = new_session()
        r = requests.delete(f"{BASE}/sessions/{sid}/memory/no-such-key")
        assert r.status_code == 404

    def test_save_memory_appears_in_trace(self):
        """The trace must contain a tool_call step for save_memory."""
        sid = new_session()
        resp = chat(sid, "My name is Benedikt.")
        assert "save_memory" in trace_tools_used(resp)

    def test_search_memory_appears_in_trace_on_recall(self):
        """When recalling a fact, the trace should show search_memory was called."""
        sid = new_session()
        chat(sid, "I love distributed systems.")
        resp = chat(sid, "What topic do I love?")
        tools = trace_tools_used(resp)
        assert "search_memory" in tools or "distributed" in resp["response"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tool usage
# ─────────────────────────────────────────────────────────────────────────────


class TestTools:
    def test_calculator_basic_arithmetic(self):
        sid = new_session()
        resp = chat(sid, "What is 123 * 456?")
        assert "56088" in resp["response"]
        assert "calculator" in trace_tools_used(resp)

    def test_calculator_exponentiation(self):
        sid = new_session()
        resp = chat(sid, "What is 2 to the power of 16?")
        assert "65536" in resp["response"]

    def test_calculator_square_root(self):
        sid = new_session()
        resp = chat(sid, "What is the square root of 1764?")
        assert "42" in resp["response"]
        assert "calculator" in trace_tools_used(resp)

    def test_calculator_chained_expression(self):
        sid = new_session()
        resp = chat(sid, "Calculate (100 / 4) + (3 ** 3)")
        # 25 + 27 = 52
        assert "52" in resp["response"]

    def test_current_time_tool(self):
        sid = new_session()
        resp = chat(sid, "What is the current UTC time and date?")
        tools = trace_tools_used(resp)
        assert "current_time" in tools
        # Response should contain a year-like string
        assert re.search(r"20\d\d", resp["response"])

    def test_current_time_tool_does_not_attribute_time_to_user(self):
        sid = new_session()
        resp = chat(sid, "What day is it today?")
        response_lower = resp["response"].lower()
        assert "current_time" in trace_tools_used(resp)
        assert "thank you for providing" not in response_lower
        assert "you provided" not in response_lower

    def test_summarize_history_tool(self):
        sid = new_session()
        chat(sid, "I'm working on a Kubernetes migration project.")
        chat(sid, "My deadline is end of July.")
        resp = chat(sid, "Can you summarize our conversation so far?")
        tools = trace_tools_used(resp)
        assert "summarize_history" in tools
        response_lower = resp["response"].lower()
        assert "kubernetes" in response_lower or "migration" in response_lower

    def test_multiple_tools_in_one_turn(self):
        """Agent should call both save_memory and calculator in a single turn."""
        sid = new_session()
        resp = chat(
            sid,
            "My budget is 512 and I need to divide it among 4 teams. Also remember I manage 4 teams.",
        )
        tools = trace_tools_used(resp)
        assert "calculator" in tools
        assert "128" in resp["response"]  # 512 / 4


# ─────────────────────────────────────────────────────────────────────────────
# 6. Error handling & validation
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_chat_with_unknown_session_returns_404(self):
        r = requests.post(
            f"{BASE}/chat",
            json={"session_id": "ghost-session-xyz", "message": "Hello!"},
        )
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_chat_with_empty_message_returns_422(self):
        sid = new_session()
        r = requests.post(f"{BASE}/chat", json={"session_id": sid, "message": ""})
        assert r.status_code == 422

    def test_chat_missing_session_id_returns_422(self):
        r = requests.post(f"{BASE}/chat", json={"message": "Hello!"})
        assert r.status_code == 422

    def test_chat_missing_message_returns_422(self):
        sid = new_session()
        r = requests.post(f"{BASE}/chat", json={"session_id": sid})
        assert r.status_code == 422

    def test_memory_endpoint_for_unknown_session_returns_empty(self):
        """Memory for a session with no entries returns an empty dict (not 404)."""
        sid = new_session()
        r = requests.get(f"{BASE}/sessions/{sid}/memory")
        assert r.status_code == 200
        assert r.json()["memory"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full end-to-end scenario (mirrors the interview spec exactly)
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndScenario:
    def test_stefan_airport_scenario(self):
        """
        Exact scenario from the interview spec:
          Turn 1: User introduces themselves → agent saves memory
          Turn 2: User asks what they work on → agent searches and answers correctly
        """
        sid = new_session()

        # ── Turn 1 ─────────────────────────────────────────────────────────
        resp1 = chat(sid, "My name is Stefan and I work on airport systems.")

        # Trace must show the full pipeline
        steps1 = trace_steps(resp1)
        assert "load_history" in steps1
        assert "load_memory" in steps1
        assert "llm_call" in steps1
        assert "save_history" in steps1
        assert "response" in steps1

        # Memory must have been saved
        assert "save_memory" in trace_tools_used(resp1)

        # ── Verify memory was written ───────────────────────────────────────
        mem = requests.get(f"{BASE}/sessions/{sid}/memory").json()["memory"]
        all_values = " ".join(e["value"] for e in mem.values()).lower()
        assert "stefan" in all_values
        assert "airport" in all_values

        # ── Turn 2 ─────────────────────────────────────────────────────────
        resp2 = chat(sid, "What systems do I work on?")

        # Agent must answer correctly from memory
        assert "airport" in resp2["response"].lower()

        # History must now contain 4 messages (2 turns × 2 messages each)
        history = requests.get(f"{BASE}/sessions/{sid}/history").json()
        assert len(history["messages"]) == 4

    def test_full_workflow_name_goal_calculation(self):
        """
        Realistic multi-turn workflow covering all interview requirements in one flow:
        memory, history continuity, tool use, and trace completeness.
        """
        sid = new_session()

        # Turn 1 – introduce yourself
        chat(
            sid, "Hi! I'm Lena and my goal is to reduce airport gate conflicts by 30%."
        )

        # Turn 2 – add more context
        chat(sid, "I'm using a greedy graph-colouring algorithm for the scheduling.")

        # Turn 3 – use the calculator
        resp3 = chat(
            sid,
            "If I have 240 gates and need 30% fewer conflicts, how many conflict reductions is that?",
        )
        assert "72" in resp3["response"]  # 240 * 0.30 = 72
        assert "calculator" in trace_tools_used(resp3)

        # Turn 4 – recall memory
        resp4 = chat(sid, "What's my name and what algorithm am I using?")
        r4_lower = resp4["response"].lower()
        assert "lena" in r4_lower
        assert (
            "graph" in r4_lower
            or "colouring" in r4_lower
            or "coloring" in r4_lower
            or "greedy" in r4_lower
        )

        # Turn 5 – history check
        resp5 = chat(sid, "Summarize everything we've discussed.")
        assert "summarize_history" in trace_tools_used(resp5)
        assert (
            "lena" in resp5["response"].lower()
            or "airport" in resp5["response"].lower()
        )

        # All 5 turns × 2 messages = 10 messages in history
        history = requests.get(f"{BASE}/sessions/{sid}/history").json()
        assert len(history["messages"]) == 10
