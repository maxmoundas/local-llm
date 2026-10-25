"""Tests for the research loop: prompted tool parsing, tools, router, and ResearchAgent."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import tools as tools_mod
from src.agent.router import ENOUGH_THRESHOLD, ResearchRouter, RouteDecision
from src.agent.runner import ResearchAgent, build_answer_system
from src.agent.tools import Source, Tool, ToolResult
from src.decision_service import Answer, DecisionError, DecisionService
from src.llm_service import LLMService
from src.providers.base import ToolCall, ToolTurn, parse_prompted_tool_turn


# ── Helpers ────────────────────────────────────────────────────────────────

def _decisions(p: float | Exception) -> MagicMock:
    d = MagicMock(spec=DecisionService)
    d.available = True
    if isinstance(p, Exception):
        d.ask.side_effect = p
    else:
        d.ask.side_effect = lambda state, qs, call_type=None: {k: Answer("noul", p) for k in qs}
    return d


def _fake_tool(name: str = "web_search", text: str = "result", url: str = "https://a.com") -> Tool:
    return Tool(
        name=name, description="d", parameters={"type": "object", "properties": {}},
        run=MagicMock(return_value=ToolResult(text=text, sources=[Source("A", url)], summary="1 result")),
        running=lambda a: f"Running {name}", done=lambda a: f"Ran {name}",
    )


def _events(agent, question="q"):
    return list(agent.run([{"role": "user", "content": question}], question, "m"))


# ── parse_prompted_tool_turn ───────────────────────────────────────────────

class TestPromptedToolParsing:
    def test_parses_calls(self):
        turn = parse_prompted_tool_turn('{"tool_calls": [{"name": "web_search", "arguments": {"query": "x"}}]}',
                                        {"web_search"})
        assert turn.tool_calls[0].name == "web_search"
        assert turn.tool_calls[0].arguments == {"query": "x"}

    def test_tolerates_fences_and_prose(self):
        raw = 'Sure:\n```json\n{"tool_calls": [{"name": "get_weather", "arguments": {"location": "Paris"}}]}\n```'
        assert parse_prompted_tool_turn(raw, {"get_weather"}).tool_calls[0].arguments["location"] == "Paris"

    def test_empty_list_means_done(self):
        assert parse_prompted_tool_turn('{"tool_calls": []}', {"web_search"}).tool_calls == []

    def test_unknown_tool_dropped(self):
        turn = parse_prompted_tool_turn('{"tool_calls": [{"name": "rm_rf", "arguments": {}}]}', {"web_search"})
        assert turn.tool_calls == []

    def test_non_json_returns_text(self):
        turn = parse_prompted_tool_turn("I have enough info.", {"web_search"})
        assert turn.tool_calls == [] and turn.text == "I have enough info."


class TestDefaultCompleteWithTools:
    def test_appends_tool_listing_to_system_prompt(self):
        from src.providers.base import BaseProvider

        class P(BaseProvider):
            provider_name = "p"
            models = []
            def stream(self, *a, **k): return iter([])
            def complete(self, messages, model_id):
                self.seen = messages
                return '{"tool_calls": [{"name": "web_search", "arguments": {"query": "q"}}]}'

        p = P()
        spec = _fake_tool().spec
        turn = p.complete_with_tools([{"role": "system", "content": "SYS"}, {"role": "user", "content": "u"}], "m", [spec])
        assert turn.tool_calls[0].name == "web_search"
        assert p.seen[0]["content"].startswith("SYS") and "web_search" in p.seen[0]["content"]


# ── tools ──────────────────────────────────────────────────────────────────

class TestTools:
    def test_default_tool_names(self):
        assert [t.name for t in tools_mod.default_tools()] == ["web_search", "news_search", "fetch_page", "get_weather"]

    def test_specs_are_json_schema_objects(self):
        for t in tools_mod.default_tools():
            assert t.spec["parameters"]["type"] == "object"
            assert t.running({}) and t.done({})

    def test_status_lines_quote_query(self):
        t = next(t for t in tools_mod.default_tools() if t.name == "web_search")
        assert t.running({"query": "nashville weather"}) == "Searching the web for “nashville weather”"

    @pytest.mark.parametrize("url", ["http://localhost/", "http://127.0.0.1:8000/x", "http://10.0.0.5/",
                                     "http://192.168.1.1/", "file:///etc/passwd", "ftp://example.com/"])
    def test_fetch_page_refuses_non_public_urls(self, url):
        result = tools_mod._fetch_page({"url": url})
        assert result.text.startswith("Error")
        assert result.sources == []

    def test_html_to_text_strips_scripts_and_chrome(self):
        html = ("<html><head><title>T</title><script>evil()</script></head><body><nav>menu</nav>"
                "<article><p>Hello   world</p></article><footer>foot</footer></body></html>")
        title, text = tools_mod._html_to_text(html)
        assert title == "T" and text == "Hello world"


# ── ResearchRouter ─────────────────────────────────────────────────────────

class TestRouter:
    def test_jev_above_threshold_researches(self):
        r = ResearchRouter(MagicMock(spec=LLMService), _decisions(0.9))
        assert r.decide("weather?", []) == RouteDecision(research=True, probability=0.9, source="jev")

    def test_jev_below_threshold_answers_directly(self):
        assert not ResearchRouter(MagicMock(spec=LLMService), _decisions(0.1)).decide("hi", []).research

    def test_jev_state_includes_recent_conversation(self):
        d = _decisions(0.9)
        history = [{"role": "user", "content": "weather in Paris?"}, {"role": "assistant", "content": "Sunny"}]
        ResearchRouter(MagicMock(spec=LLMService), d).decide("and tomorrow?", history)
        state = d.ask.call_args[0][0]
        assert state["latest_message"] == "and tomorrow?"
        assert state["recent_conversation"][0]["content"] == "weather in Paris?"

    def test_falls_back_to_llm_when_jev_errors(self):
        llm = MagicMock(spec=LLMService)
        llm.complete.return_value = "Yes"
        d = ResearchRouter(llm, _decisions(DecisionError("down"))).decide("news?", [])
        assert d.research and d.source == "llm"

    def test_uses_llm_when_jev_routing_disabled(self):
        llm = MagicMock(spec=LLMService)
        llm.complete.return_value = "no"
        jev = _decisions(0.99)
        assert not ResearchRouter(llm, jev, use_jev=False).decide("news?", []).research
        jev.ask.assert_not_called()

    def test_llm_failure_means_no_research(self):
        llm = MagicMock(spec=LLMService)
        llm.complete.side_effect = RuntimeError("down")
        assert not ResearchRouter(llm, None).decide("news?", []).research

    def test_enough_none_without_jev(self):
        assert ResearchRouter(MagicMock(spec=LLMService), None).enough("q", "notes") is None


# ── ResearchAgent ──────────────────────────────────────────────────────────

class TestResearchAgent:
    def _agent(self, turns, enough=None, tools=None, max_steps=4):
        llm = MagicMock(spec=LLMService)
        llm.complete_with_tools.side_effect = turns
        router = MagicMock(spec=ResearchRouter)
        router.enough.return_value = enough
        return ResearchAgent(llm, router, tools or [_fake_tool()], max_steps=max_steps), llm, router

    def test_emits_status_step_done(self):
        agent, _, _ = self._agent([ToolTurn(tool_calls=[ToolCall("web_search", {"query": "x"})]), ToolTurn()])
        events = _events(agent)
        types = [e.type for e in events]
        assert types[0] == "status" and "step" in types and types[-1] == "done"
        assert events[1].text == "Running web_search"

    def test_done_carries_notes_sources_steps(self):
        agent, _, _ = self._agent([ToolTurn(tool_calls=[ToolCall("web_search", {"query": "x"})]), ToolTurn()])
        done = _events(agent)[-1]
        assert "result" in done.notes and "Ran web_search" in done.notes
        assert done.sources == [{"title": "A", "url": "https://a.com"}]
        assert done.steps[0]["label"] == "Ran web_search"

    def test_stops_when_jev_says_enough(self):
        turns = [ToolTurn(tool_calls=[ToolCall("web_search", {"query": "x"})])] * 4
        agent, llm, _ = self._agent(turns, enough=ENOUGH_THRESHOLD + 0.01)
        _events(agent)
        assert llm.complete_with_tools.call_count == 1

    def test_continues_when_not_enough(self):
        turns = [ToolTurn(tool_calls=[ToolCall("web_search", {"query": f"q{i}"})]) for i in range(4)]
        agent, llm, _ = self._agent(turns, enough=0.1, max_steps=3)
        _events(agent)
        assert llm.complete_with_tools.call_count == 3

    def test_skips_repeated_calls(self):
        same = ToolTurn(tool_calls=[ToolCall("web_search", {"query": "x"})])
        tool = _fake_tool()
        agent, llm, _ = self._agent([same, same, same], tools=[tool])
        _events(agent)
        assert tool.run.call_count == 1
        assert llm.complete_with_tools.call_count == 2  # second turn repeats → loop ends

    def test_parallel_calls_capped(self):
        calls = [ToolCall("web_search", {"query": str(i)}) for i in range(10)]
        tool = _fake_tool()
        agent, _, _ = self._agent([ToolTurn(tool_calls=calls), ToolTurn()], tools=[tool])
        events = _events(agent)
        assert tool.run.call_count == 4
        assert events[1].text.endswith("and 3 more")

    def test_tool_exception_becomes_error_note(self):
        tool = _fake_tool()
        tool.run.side_effect = RuntimeError("boom")
        agent, _, _ = self._agent([ToolTurn(tool_calls=[ToolCall("web_search", {})]), ToolTurn()], tools=[tool])
        assert "Error: boom" in _events(agent)[-1].notes

    def test_planner_failure_still_finishes(self):
        agent, _, _ = self._agent(RuntimeError("api down"))
        assert _events(agent)[-1].type == "done"

    def test_answer_system_wraps_notes(self):
        text = build_answer_system("BASE", "NOTE")
        assert text.startswith("BASE") and "<research_notes>\nNOTE\n</research_notes>" in text


class TestSearchErrors:
    def test_no_results_becomes_advice(self, monkeypatch):
        from ddgs.exceptions import DDGSException
        result = tools_mod._ddgs_search(lambda: (_ for _ in ()).throw(DDGSException("No results found.")))[1]
        assert result.summary == "0 results" and "shorter query" in result.text

    def test_rate_limit_becomes_advice(self):
        from ddgs.exceptions import RatelimitException
        result = tools_mod._ddgs_search(lambda: (_ for _ in ()).throw(RatelimitException("429")))[1]
        assert result.summary == "rate-limited"

    def test_other_errors_propagate(self):
        from ddgs.exceptions import DDGSException
        with pytest.raises(DDGSException):
            tools_mod._ddgs_search(lambda: (_ for _ in ()).throw(DDGSException("boom")))


class TestFirstStepRequiresTools:
    def test_first_step_requires_then_optional(self):
        llm = MagicMock(spec=LLMService)
        llm.complete_with_tools.side_effect = [ToolTurn(tool_calls=[ToolCall("web_search", {"query": "a"})]), ToolTurn()]
        router = MagicMock(spec=ResearchRouter)
        router.enough.return_value = None
        list(ResearchAgent(llm, router, [_fake_tool()]).run([{"role": "user", "content": "q"}], "q", "m"))
        requires = [c.kwargs["require"] for c in llm.complete_with_tools.call_args_list]
        assert requires == [True, False]

    def test_anthropic_sets_tool_choice(self):
        from unittest.mock import patch
        with patch("anthropic.Anthropic") as cls:
            from src.providers.anthropic_provider import AnthropicProvider
            cls.return_value.messages.create.return_value = MagicMock(content=[])
            p = AnthropicProvider("k")
            p.complete_with_tools([{"role": "user", "content": "q"}], "m", [_fake_tool().spec], require=True)
            assert cls.return_value.messages.create.call_args.kwargs["tool_choice"] == {"type": "any"}

    def test_openai_sets_tool_choice(self):
        from unittest.mock import patch
        with patch("src.providers.openai_provider.OpenAI") as cls:
            from src.providers.openai_provider import OpenAIProvider
            msg = MagicMock(content="", tool_calls=[])
            cls.return_value.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=msg)])
            OpenAIProvider("k").complete_with_tools([{"role": "user", "content": "q"}], "m", [_fake_tool().spec], require=True)
            assert cls.return_value.chat.completions.create.call_args.kwargs["tool_choice"] == "required"

    def test_prompted_fallback_adds_instruction(self):
        from src.providers.base import BaseProvider

        class P(BaseProvider):
            provider_name = "p"
            models = []
            def stream(self, *a, **k): return iter([])
            def complete(self, messages, model_id):
                self.sys = messages[0]["content"]
                return '{"tool_calls": []}'

        p = P()
        p.complete_with_tools([{"role": "user", "content": "q"}], "m", [_fake_tool().spec], require=True)
        assert "must call at least one tool" in p.sys
