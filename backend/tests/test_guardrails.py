"""Tests for GuardrailService and guardrail integration in the message endpoint.

Structure:
  TestGuardrailResult       — dataclass sanity checks
  TestBuildPrompt           — _build_prompt() helper
  TestGuardrailServiceCheck — GuardrailService.check() unit tests (mock LLM)
  TestInputGuardrailIntegration  — /api/chats/{id}/message input blocking (test_client)
  TestOutputGuardrailIntegration — /api/chats/{id}/message output blocking (test_client)
  TestGuardrailSettings     — settings persistence and API
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.guardrail_service import GuardrailResult, GuardrailService, _build_prompt
from src.llm_service import CALL_TYPE_GUARD, LLMService
from src.settings_store import DEFAULT_GUARDRAIL_PROMPT


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_guardrail(blocked: bool, reason: str | None = None) -> GuardrailService:
    """Return a GuardrailService whose LLM always returns the given result."""
    data: dict = {"blocked": blocked}
    if reason is not None:
        data["reason"] = reason
    mock_svc = MagicMock(spec=LLMService)
    mock_svc.complete.return_value = json.dumps(data)
    return GuardrailService(mock_svc, "check {denied_section}", [])


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE response body into a list of event dicts."""
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:].strip()))
            except json.JSONDecodeError:
                pass
    return events


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_guardrail_state():
    """Restore guardrail module-level state after every test."""
    from src import server
    orig_svc = server._guardrail_svc
    orig_input = server._guardrails_input
    orig_output = server._guardrails_output
    yield
    server._guardrail_svc = orig_svc
    server._guardrails_input = orig_input
    server._guardrails_output = orig_output


@pytest.fixture(autouse=True)
def fast_llm_stream(monkeypatch):
    """Replace LocalLLM.stream with a simple iterator.

    LocalLLM.stream() uses TextIteratorStreamer + a background thread. The
    mock model's generate() doesn't feed the streamer, causing tests that reach
    the actual stream to hang. This fixture short-circuits the whole mechanism.
    """
    from src.model import LocalLLM
    monkeypatch.setattr(LocalLLM, "stream", lambda self, *a, **kw: iter(["hello ", "world"]))


@pytest.fixture(autouse=True)
def tmp_settings(tmp_path, monkeypatch):
    import src.settings_store as ss
    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "settings.json")


# ── GuardrailResult ────────────────────────────────────────────────────────

class TestGuardrailResult:
    def test_not_blocked_has_no_reason(self):
        r = GuardrailResult(blocked=False)
        assert r.blocked is False
        assert r.reason is None

    def test_blocked_with_reason(self):
        r = GuardrailResult(blocked=True, reason="hate speech")
        assert r.blocked is True
        assert r.reason == "hate speech"

    def test_blocked_without_reason(self):
        r = GuardrailResult(blocked=True)
        assert r.blocked is True
        assert r.reason is None


# ── _build_prompt() ────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_no_topics_replaces_placeholder_with_empty(self):
        result = _build_prompt("check {denied_section} end", [])
        assert result == "check  end"

    def test_topics_appear_in_output(self):
        result = _build_prompt("{denied_section}", ["politics", "sports"])
        assert "politics" in result
        assert "sports" in result

    def test_topics_formatted_as_bullets(self):
        result = _build_prompt("{denied_section}", ["topic-a"])
        assert "- topic-a" in result

    def test_multiple_topics_all_present(self):
        topics = ["cats", "dogs", "birds"]
        result = _build_prompt("{denied_section}", topics)
        for t in topics:
            assert t in result

    def test_no_topics_no_bullet_chars(self):
        result = _build_prompt("base {denied_section}", [])
        assert "- " not in result

    def test_prompt_without_placeholder_unchanged(self):
        result = _build_prompt("no placeholder here", ["topic"])
        assert result == "no placeholder here"


# ── GuardrailService.check() ──────────────────────────────────────────────

class TestGuardrailServiceCheck:
    def _svc(self, llm_response: str, topics: list[str] | None = None) -> GuardrailService:
        mock = MagicMock(spec=LLMService)
        mock.complete.return_value = llm_response
        return GuardrailService(mock, "prompt {denied_section}", topics or [])

    def test_not_blocked_when_llm_returns_false(self):
        result = self._svc('{"blocked": false}').check("hello")
        assert result.blocked is False

    def test_blocked_with_reason_when_llm_returns_true(self):
        result = self._svc('{"blocked": true, "reason": "hate speech"}').check("bad")
        assert result.blocked is True
        assert result.reason == "hate speech"

    def test_blocked_without_reason_field(self):
        result = self._svc('{"blocked": true}').check("bad")
        assert result.blocked is True
        assert result.reason is None

    def test_not_blocked_reason_is_none(self):
        result = self._svc('{"blocked": false}').check("good")
        assert result.reason is None

    def test_fails_open_on_llm_exception(self):
        mock = MagicMock(spec=LLMService)
        mock.complete.side_effect = RuntimeError("api down")
        svc = GuardrailService(mock, "prompt {denied_section}", [])
        result = svc.check("any text")
        assert result.blocked is False

    def test_fails_open_on_malformed_json(self):
        result = self._svc("not json at all").check("text")
        assert result.blocked is False

    def test_fails_open_on_empty_string(self):
        result = self._svc("").check("text")
        assert result.blocked is False

    def test_handles_markdown_fenced_json(self):
        result = self._svc('```json\n{"blocked": true, "reason": "test"}\n```').check("t")
        assert result.blocked is True
        assert result.reason == "test"

    def test_handles_fenced_json_without_lang_tag(self):
        result = self._svc('```\n{"blocked": false}\n```').check("t")
        assert result.blocked is False

    def test_text_truncated_to_2000_chars(self):
        mock = MagicMock(spec=LLMService)
        mock.complete.return_value = '{"blocked": false}'
        svc = GuardrailService(mock, "p {denied_section}", [])
        svc.check("x" * 5000)
        call_args = mock.complete.call_args[0][0]
        user_msg = next(m for m in call_args if m["role"] == "user")
        # Content is "Text to evaluate:\n" + truncated text; total < 2100
        assert len(user_msg["content"]) < 2100

    def test_uses_call_type_guard(self):
        mock = MagicMock(spec=LLMService)
        mock.complete.return_value = '{"blocked": false}'
        svc = GuardrailService(mock, "p {denied_section}", [])
        svc.check("test")
        _, kwargs = mock.complete.call_args
        assert kwargs.get("call_type") == CALL_TYPE_GUARD

    def test_denied_topics_in_system_prompt(self):
        mock = MagicMock(spec=LLMService)
        mock.complete.return_value = '{"blocked": false}'
        svc = GuardrailService(mock, "check {denied_section}", ["politics"])
        svc.check("test")
        call_args = mock.complete.call_args[0][0]
        sys_msg = next(m for m in call_args if m["role"] == "system")
        assert "politics" in sys_msg["content"]

    def test_no_topics_no_bullet_in_system_prompt(self):
        mock = MagicMock(spec=LLMService)
        mock.complete.return_value = '{"blocked": false}'
        svc = GuardrailService(mock, "check {denied_section}", [])
        svc.check("test")
        call_args = mock.complete.call_args[0][0]
        sys_msg = next(m for m in call_args if m["role"] == "system")
        assert "- " not in sys_msg["content"]

    def test_blocked_null_treated_as_false(self):
        result = self._svc('{"blocked": null}').check("text")
        assert result.blocked is False

    def test_reason_coerced_to_string(self):
        # reason value is not a string but should be converted
        result = self._svc('{"blocked": true, "reason": 42}').check("text")
        assert result.blocked is True
        assert result.reason == "42"


# ── Input guardrail integration ────────────────────────────────────────────

class TestInputGuardrailIntegration:
    def _chat_id(self, test_client) -> str:
        return test_client.post("/api/chats").json()["id"]

    def _events(self, test_client, chat_id: str, content: str = "hi") -> list[dict]:
        resp = test_client.post(
            f"/api/chats/{chat_id}/message",
            json={"content": content},
        )
        return _parse_sse(resp.text)

    def test_blocked_input_emits_blocked_event(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True, reason="test")
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id, "bad message")
        blocked = [e for e in events if e.get("type") == "blocked"]
        assert len(blocked) == 1

    def test_blocked_input_target_is_input(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        blocked = next(e for e in events if e.get("type") == "blocked")
        assert blocked["target"] == "input"

    def test_blocked_input_reason_in_event(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True, reason="hate speech")
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        blocked = next(e for e in events if e.get("type") == "blocked")
        assert blocked["reason"] == "hate speech"

    def test_blocked_input_no_reason_field_when_none(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True, reason=None)
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        blocked = next(e for e in events if e.get("type") == "blocked")
        assert "reason" not in blocked

    def test_blocked_input_user_message_not_stored(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        self._events(test_client, chat_id, "blocked message")
        stored = test_client.get(f"/api/chats/{chat_id}").json()
        assert len(stored["messages"]) == 0

    def test_allowed_input_emits_done_not_blocked(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=False)
        server._guardrails_input = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id, "safe message")
        types = [e.get("type") for e in events]
        assert "blocked" not in types
        assert "done" in types

    def test_no_guardrail_svc_skips_check(self, test_client):
        from src import server
        server._guardrail_svc = None
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        types = [e.get("type") for e in events]
        assert "blocked" not in types

    def test_guardrails_input_false_skips_check(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = False  # check disabled
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        types = [e.get("type") for e in events]
        assert "blocked" not in types
        assert "done" in types


# ── Output guardrail integration ───────────────────────────────────────────

class TestOutputGuardrailIntegration:
    def _chat_id(self, test_client) -> str:
        return test_client.post("/api/chats").json()["id"]

    def _events(self, test_client, chat_id: str) -> list[dict]:
        resp = test_client.post(
            f"/api/chats/{chat_id}/message",
            json={"content": "hi"},
        )
        return _parse_sse(resp.text)

    def test_blocked_output_emits_blocked_event(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True, reason="unsafe")
        server._guardrails_input = False
        server._guardrails_output = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        blocked = [e for e in events if e.get("type") == "blocked"]
        assert len(blocked) == 1

    def test_blocked_output_target_is_output(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = False
        server._guardrails_output = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        blocked = next(e for e in events if e.get("type") == "blocked")
        assert blocked["target"] == "output"

    def test_blocked_output_assistant_message_not_stored(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = False
        server._guardrails_output = True
        chat_id = self._chat_id(test_client)
        self._events(test_client, chat_id)
        stored = test_client.get(f"/api/chats/{chat_id}").json()
        messages = [m for m in stored["messages"] if m["role"] == "assistant"]
        assert len(messages) == 0

    def test_allowed_output_stored_and_done_emitted(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=False)
        server._guardrails_input = False
        server._guardrails_output = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        types = [e.get("type") for e in events]
        assert "done" in types
        assert "blocked" not in types

    def test_guardrails_output_false_skips_check(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = False
        server._guardrails_output = False  # output check disabled
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        types = [e.get("type") for e in events]
        assert "blocked" not in types
        assert "done" in types

    def test_tokens_emitted_before_blocked_event(self, test_client):
        from src import server
        server._guardrail_svc = _make_guardrail(blocked=True)
        server._guardrails_input = False
        server._guardrails_output = True
        chat_id = self._chat_id(test_client)
        events = self._events(test_client, chat_id)
        # Tokens should be present (model generated them) before the blocked event
        token_events = [e for e in events if e.get("type") == "token"]
        blocked_events = [e for e in events if e.get("type") == "blocked"]
        assert len(token_events) > 0
        assert len(blocked_events) == 1
        # Blocked event comes after tokens
        token_idx = max(i for i, e in enumerate(events) if e.get("type") == "token")
        blocked_idx = next(i for i, e in enumerate(events) if e.get("type") == "blocked")
        assert blocked_idx > token_idx


# ── Guardrail settings ─────────────────────────────────────────────────────

class TestGuardrailSettings:
    def test_get_settings_includes_guardrails_enabled(self, test_client):
        data = test_client.get("/api/settings").json()
        assert "guardrails_enabled" in data

    def test_get_settings_default_guardrails_disabled(self, test_client):
        data = test_client.get("/api/settings").json()
        assert data["guardrails_enabled"] is False

    def test_get_settings_includes_default_guardrail_prompt(self, test_client):
        data = test_client.get("/api/settings").json()
        assert "default_guardrail_prompt" in data
        assert isinstance(data["default_guardrail_prompt"], str)
        assert len(data["default_guardrail_prompt"]) > 0

    def test_get_settings_includes_guardrail_topics(self, test_client):
        data = test_client.get("/api/settings").json()
        assert "guardrails_topics" in data
        assert data["guardrails_topics"] == []

    def test_patch_enables_guardrails(self, test_client):
        resp = test_client.patch("/api/settings", json={"guardrails_enabled": True})
        assert resp.json()["guardrails_enabled"] is True

    def test_patch_persists_guardrails_enabled(self, test_client):
        test_client.patch("/api/settings", json={"guardrails_enabled": True})
        data = test_client.get("/api/settings").json()
        assert data["guardrails_enabled"] is True

    def test_patch_persists_guardrail_topics(self, test_client):
        test_client.patch("/api/settings", json={"guardrails_topics": ["topic-a", "topic-b"]})
        data = test_client.get("/api/settings").json()
        assert data["guardrails_topics"] == ["topic-a", "topic-b"]

    def test_patch_persists_guardrail_prompt(self, test_client):
        test_client.patch("/api/settings", json={"guardrails_prompt": "custom prompt {denied_section}"})
        data = test_client.get("/api/settings").json()
        assert data["guardrails_prompt"] == "custom prompt {denied_section}"

    def test_patch_enables_creates_guardrail_svc(self, test_client):
        from src import server
        test_client.patch("/api/settings", json={"guardrails_enabled": True})
        assert server._guardrail_svc is not None

    def test_patch_disables_clears_guardrail_svc(self, test_client):
        from src import server
        test_client.patch("/api/settings", json={"guardrails_enabled": True})
        test_client.patch("/api/settings", json={"guardrails_enabled": False})
        assert server._guardrail_svc is None

    def test_patch_guardrails_input_updates_server_state(self, test_client):
        from src import server
        test_client.patch("/api/settings", json={"guardrails_input": False})
        assert server._guardrails_input is False

    def test_patch_guardrails_output_updates_server_state(self, test_client):
        from src import server
        test_client.patch("/api/settings", json={"guardrails_output": True})
        assert server._guardrails_output is True

    def test_default_guardrail_prompt_contains_placeholder(self):
        assert "{denied_section}" in DEFAULT_GUARDRAIL_PROMPT

    def test_default_guardrail_prompt_nonempty(self):
        assert len(DEFAULT_GUARDRAIL_PROMPT) > 0
