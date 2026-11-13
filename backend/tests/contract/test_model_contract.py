"""What every model must do for the agent loop to work. See conftest.py for how to run."""
from __future__ import annotations

import re

import pytest

from src import usage
from src.agent.runner import TaskAgent, with_status_arg
from src.agent.tools import Tool, ToolResult

pytestmark = pytest.mark.contract

WEATHER = {"name": "get_weather", "description": "Current weather for a city.",
           "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}
SEARCH = {"name": "web_search", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use tools when they help."}


def _calc_tool(calls: list) -> Tool:
    def run(a):
        calls.append(a)
        expr = str(a.get("expression", ""))
        if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
            return ToolResult(text="Error: only arithmetic is allowed", summary="failed")
        return ToolResult(text=str(eval(expr)), summary="ok")  # noqa: S307 — digits/operators only
    return Tool(name="calculator", description="Evaluate an arithmetic expression exactly.",
                parameters={"type": "object", "properties": {"expression": {"type": "string"}},
                            "required": ["expression"]},
                run=run, running=lambda a: "Calculating", done=lambda a: "Calculated", read_only=False)


# ── text ─────────────────────────────────────────────────────────────────

def test_complete(svc, model, record):
    out = svc.complete([{"role": "user", "content": "Reply with exactly the word: banana"}], model.id, max_tokens=50)
    ok = "banana" in out.lower()
    record("complete+instruction", ok)
    assert ok, out


def test_stream_and_usage(svc, model, record):
    with usage.meter() as m:
        text = "".join(svc.stream([{"role": "user", "content": "Count from 1 to 5, comma separated."}], model.id))
    ok = "3" in text and m.output_tokens > 0
    record("stream+usage", ok)
    assert ok, (text, m.snapshot())


# ── tools (each in native and prompted mode) ─────────────────────────────

def test_required_tool_call(svc, model, tools_as, record):
    turn = svc.complete_with_tools([SYSTEM, {"role": "user", "content": "What's the weather in Paris right now?"}],
                                   model.id, [with_status_arg(WEATHER), with_status_arg(SEARCH)], require=True)
    ok = any(c.name == "get_weather" and "paris" in str(c.arguments.get("location", "")).lower()
             for c in turn.tool_calls)
    record("tool_call", ok)
    assert ok, turn


def test_status_line_filled(svc, model, tools_as, record):
    turn = svc.complete_with_tools([SYSTEM, {"role": "user", "content": "Search the web for the tallest building."}],
                                   model.id, [with_status_arg(SEARCH)], require=True)
    ok = bool(turn.tool_calls) and bool(str(turn.tool_calls[0].arguments.get("status", "")).strip())
    record("status_line", ok)
    if not ok:
        pytest.xfail("model leaves `status` empty — the UI falls back to tool templates")


def test_parallel_calls(svc, model, tools_as, record):
    turn = svc.complete_with_tools(
        [SYSTEM, {"role": "user", "content": "What's the weather in Paris and in Tokyo? Call the tool for both cities at once."}],
        model.id, [with_status_arg(WEATHER)], require=True)
    cities = {str(c.arguments.get("location", "")).lower() for c in turn.tool_calls if c.name == "get_weather"}
    ok = any("paris" in c for c in cities) and any("tokyo" in c for c in cities)
    record("parallel_calls", ok)
    if not ok:
        pytest.xfail("no parallel calls — tasks still work, just with more steps")


def test_stops_when_done(svc, model, tools_as, record):
    msgs = [SYSTEM, {"role": "user", "content":
                     "Question: what's the weather in Paris?\n<work_log>\n### [1] Checked the weather in Paris\n"
                     "18°C, sunny, light wind.\n</work_log>\nThe log fully answers the question. "
                     "Which tools should be called next? Call none if nothing more is needed."}]
    turn = svc.complete_with_tools(msgs, model.id, [with_status_arg(WEATHER), with_status_arg(SEARCH)])
    ok = not turn.tool_calls
    record("stops_when_done", ok)
    assert ok, turn


# ── the real loop ─────────────────────────────────────────────────────────

def test_agent_loop_end_to_end(svc, model, tools_as, record):
    calls: list = []
    agent = TaskAgent(svc, None, [_calc_tool(calls)], max_steps=4)
    q = "What is 48271 * 9137? Use the calculator."
    events = list(agent.run([{"role": "user", "content": q}], q, model.id))
    done = events[-1]
    ok = bool(calls) and "441052127" in done.notes
    record("agent_loop", ok)
    assert ok, (calls, done.notes[:500], done.stopped)
