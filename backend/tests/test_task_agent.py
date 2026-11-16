"""TaskAgent end-to-end over the real provider → registry → LLMService stack (ScriptedProvider)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src import usage
from src.agent.runner import RunControls, TaskAgent, with_status_arg
from src.agent.sandbox import LocalSandbox
from src.agent.tools import Tool, ToolResult
from src.agent.workspace_tools import workspace_tools
from src.providers.base import ToolTurn
from tests.fakes import ScriptedProvider, calls, service


def _tool(name="lookup", read_only=True, confirm=False, text="found it", run=None) -> Tool:
    return Tool(name=name, description="d", parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                run=run or MagicMock(return_value=ToolResult(text=text, summary="1 result")),
                running=lambda a: f"Looking up {a.get('q')}", done=lambda a: f"Looked up {a.get('q')}",
                read_only=read_only, confirm=confirm)


def _run(provider, tools, controls=None, max_steps=6, question="do it"):
    agent = TaskAgent(service(provider), None, tools, max_steps=max_steps)
    with usage.meter() as m:
        if controls is not None and controls.meter is None:
            controls.meter = m
        events = list(agent.run([{"role": "user", "content": question}], question, "fake", controls=controls))
    return events, m


def _actions(events):
    return [e.action for e in events if e.type == "action"]


class TestStatusLines:
    def test_status_arg_added_to_every_spec_and_stripped_before_running(self):
        tool = _tool()
        p = ScriptedProvider(turns=[calls(("lookup", {"q": "x", "status": "Checking the x registry"}))])
        events, _ = _run(p, [tool])
        assert all("status" in t["parameters"]["properties"] for t in p.seen_tools[0] if t["name"] == "lookup")
        tool.run.assert_called_once_with({"q": "x"})
        acts = _actions(events)
        assert acts[0]["label"] == acts[-1]["label"] == "Checking the x registry"

    def test_template_used_when_model_gives_no_status(self):
        p = ScriptedProvider(turns=[calls(("lookup", {"q": "x"}))])
        acts = _actions(_run(p, [_tool()])[0])
        assert acts[0]["label"] == "Looking up x" and acts[-1]["label"] == "Looked up x"

    def test_with_status_arg_keeps_existing_schema(self):
        spec = with_status_arg({"name": "t", "description": "d",
                                "parameters": {"type": "object", "properties": {"a": {"type": "string"}},
                                               "required": ["a"]}})
        assert spec["parameters"]["required"] == ["a"] and set(spec["parameters"]["properties"]) == {"a", "status"}


class TestLimits:
    def test_budget_stops_before_next_turn(self):
        # each call costs (100*1 + 20*2)/1e6 = $0.00014
        p = ScriptedProvider(turns=[calls(("lookup", {"q": str(i)})) for i in range(10)])
        events, m = _run(p, [_tool()], RunControls(budget_usd=0.0002))
        assert events[-1].stopped == "budget"
        assert m.cost_usd >= 0.0002 and m.calls == 2

    def test_deadline_stops(self):
        import time
        p = ScriptedProvider(turns=[calls(("lookup", {"q": "a"}))])
        events, _ = _run(p, [_tool()], RunControls(deadline=time.monotonic() - 1))
        assert events[-1].stopped == "time" and not _actions(events)

    def test_cancel_stops_at_next_checkpoint(self):
        flag = {"stop": False}
        tool = _tool(run=lambda a: (flag.update(stop=True), ToolResult(text="r"))[1])
        p = ScriptedProvider(turns=[calls(("lookup", {"q": "a"})), calls(("lookup", {"q": "b"}))])
        events, _ = _run(p, [tool], RunControls(cancelled=lambda: flag["stop"]))
        assert events[-1].stopped == "cancelled" and len(p.turns) == 1

    def test_repeated_failures_stop(self):
        bad = _tool(run=MagicMock(side_effect=RuntimeError("nope")))
        p = ScriptedProvider(turns=[calls(("lookup", {"q": str(i)})) for i in range(10)])
        events, _ = _run(p, [bad], max_steps=10)
        assert events[-1].stopped == "failures"
        assert bad.run.call_count == 4


class TestPlanAndQuestions:
    def test_update_plan_is_internal_and_shown_to_planner(self):
        p = ScriptedProvider(turns=[calls(("update_plan", {"steps": ["find it", "summarize"]})),
                                    calls(("lookup", {"q": "x"})), ToolTurn()])
        events, _ = _run(p, [_tool()])
        assert all(a["tool"] != "update_plan" for a in _actions(events))
        assert "1. find it\n2. summarize" in p.seen[1][1]["content"]

    def test_turn_after_plan_must_act(self):
        # Regression: a model re-planning every turn used up all steps without calling a real tool.
        p = ScriptedProvider(turns=[calls(("update_plan", {"steps": ["search"]})),
                                    calls(("lookup", {"q": "x"})), ToolTurn()])
        events, _ = _run(p, [_tool()])
        assert "update_plan" in {t["name"] for t in p.seen_tools[0]}
        assert "update_plan" not in {t["name"] for t in p.seen_tools[1]}
        assert "update_plan" in {t["name"] for t in p.seen_tools[2]}  # offered again once it acted
        assert [a["tool"] for a in _actions(events) if a["status"] == "done"] == ["lookup"]

    def test_ask_user_offered_only_with_handler(self):
        p = ScriptedProvider(turns=[ToolTurn()])
        _run(p, [_tool()])
        assert "ask_user" not in {t["name"] for t in p.seen_tools[0]}

    def test_ask_user_waits_for_reply_and_logs_answer(self):
        asked = []
        ctl = RunControls(ask=lambda req: (asked.append(req), {"answer": "blue"})[1])
        p = ScriptedProvider(turns=[calls(("ask_user", {"question": "Which color?"})), ToolTurn()])
        events, _ = _run(p, [_tool()], ctl)
        assert asked[0]["kind"] == "question" and asked[0]["question"] == "Which color?"
        assert "blue" in events[-1].notes


class TestApprovals:
    def test_confirm_tool_denied_does_not_run(self):
        send = _tool("send_email", read_only=False, confirm=True)
        ctl = RunControls(ask=lambda req: {"approved": False})
        p = ScriptedProvider(turns=[calls(("send_email", {"q": "hi"})), ToolTurn()])
        events, _ = _run(p, [send], ctl)
        send.run.assert_not_called()
        assert [a["status"] for a in _actions(events)] == ["waiting", "denied"]
        assert "declined" in events[-1].notes

    def test_confirm_tool_approved_runs(self):
        send = _tool("send_email", read_only=False, confirm=True)
        requests = []
        ctl = RunControls(ask=lambda req: (requests.append(req), {"approved": True})[1])
        p = ScriptedProvider(turns=[calls(("send_email", {"q": "hi"})), ToolTurn()])
        events, _ = _run(p, [send], ctl)
        send.run.assert_called_once_with({"q": "hi"})
        assert requests[0]["kind"] == "approval" and requests[0]["arguments"] == {"q": "hi"}
        assert [a["status"] for a in _actions(events)] == ["waiting", "running", "done"]

    def test_confirm_tool_without_handler_is_denied(self):
        send = _tool("send_email", read_only=False, confirm=True)
        p = ScriptedProvider(turns=[calls(("send_email", {"q": "hi"})), ToolTurn()])
        _run(p, [send])
        send.run.assert_not_called()


class TestModelAgnostic:
    def test_prompted_json_model_runs_the_same_loop(self):
        tool = _tool()
        p = ScriptedProvider(turns=[calls(("lookup", {"q": "x", "status": "Looking"})), ToolTurn()],
                             tools_mode="prompted")
        events, _ = _run(p, [tool])
        tool.run.assert_called_once_with({"q": "x"})
        assert "You can call these tools" in p.seen[0][0]["content"]
        assert events[-1].actions[0]["status"] == "done"

    def test_prompted_repair_retry_on_garbage(self):
        tool = _tool()
        p = ScriptedProvider(turns=["I think I should search.", calls(("lookup", {"q": "x"})), ToolTurn()],
                             tools_mode="prompted")
        _run(p, [tool])
        tool.run.assert_called_once()

    def test_auto_mode_falls_back_to_prompted_after_native_error(self):
        tool = _tool()
        p = ScriptedProvider(turns=[RuntimeError("tools not supported"), calls(("lookup", {"q": "x"})), ToolTurn()],
                             tools_mode="auto")
        _run(p, [tool])
        assert p.models[0].tools == "prompted"
        tool.run.assert_called_once()


class TestCompaction:
    def test_old_entries_summarized_when_log_overflows(self, monkeypatch):
        import src.agent.runner as runner
        monkeypatch.setattr(runner, "_MIN_LOG_CHARS", 2000)
        monkeypatch.setattr(runner, "_MAX_LOG_CHARS", 2000)
        tool = _tool(text="x" * 900)
        p = ScriptedProvider(turns=[calls(("lookup", {"q": str(i)})) for i in range(14)],
                             texts=["SUMMARY ONE", "SUMMARY TWO", "SUMMARY THREE"])
        events, _ = _run(p, [tool], max_steps=16)
        notes = events[-1].notes
        assert notes.startswith("### [1] Summary of") and "SUMMARY" in notes
        assert "Looked up 13" in notes  # newest result kept
        summaries = [m for m in p.seen if m[0]["content"].startswith("Condense")]
        assert 1 <= len(summaries) <= 3  # folded in batches, not every step
        if len(summaries) > 1:
            assert "SUMMARY ONE" in summaries[1][1]["content"]  # earlier summary carried forward


class TestCodeTools:
    def test_run_code_and_files_in_local_sandbox(self, tmp_path):
        tools = workspace_tools(LocalSandbox(), "t1")
        by = {t.name: t for t in tools}
        r = by["run_code"].run({"code": "open('out.txt','w').write('42')\nprint(6*7)"})
        assert "42" in r.text and "out.txt" in r.text and r.summary == "ok"
        assert by["read_file"].run({"path": "out.txt"}).text == "42"
        assert "out.txt" in by["list_files"].run({}).text

    def test_workspace_paths_cannot_escape(self):
        by = {t.name: t for t in workspace_tools(LocalSandbox(), "t2")}
        with pytest.raises(ValueError):
            by["write_file"].run({"path": "../../etc/x", "content": "no"})

    def test_failing_code_reports_exit_code(self):
        by = {t.name: t for t in workspace_tools(LocalSandbox(), "t3")}
        r = by["run_code"].run({"code": "raise SystemExit(3)"})
        assert r.summary == "exit 3" and r.ok is False

    def test_timeout(self):
        by = {t.name: t for t in workspace_tools(LocalSandbox(), "t4", timeout=1)}
        r = by["run_code"].run({"code": "import time\ntime.sleep(5)"})
        assert r.summary == "timed out"


class TestRequiredCallFallback:
    def test_auto_mode_retries_prompted_when_native_returns_nothing(self):
        tool = _tool()
        p = ScriptedProvider(turns=[ToolTurn(), calls(("lookup", {"q": "x"})), ToolTurn()], tools_mode="auto")
        _run(p, [tool])
        tool.run.assert_called_once()
        assert p.models[0].tools == "auto"  # a soft miss doesn't permanently switch modes

    def test_native_mode_does_not_retry(self):
        tool = _tool()
        p = ScriptedProvider(turns=[ToolTurn(), calls(("lookup", {"q": "x"}))], tools_mode="native")
        _run(p, [tool])
        tool.run.assert_not_called()
