"""A provider whose API key is rejected: flagged unavailable, utility calls fall back, errors say what to fix."""
from __future__ import annotations

import pytest

from src.llm_service import LLMService
from src.providers.base import ProviderAuthError
from src.providers.registry import ProviderRegistry, check_api_keys
from tests.fakes import ScriptedProvider


class _HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class RejectingProvider(ScriptedProvider):
    """Every call fails with the given HTTP status, as an SDK would raise it."""

    def __init__(self, status: int = 401, **kw) -> None:
        super().__init__(**kw)
        self.status = status
        self.calls = 0

    def _fail(self):
        self.calls += 1
        raise _HTTPError(self.status)

    def complete(self, messages, model_id, max_tokens=None) -> str:
        self._fail()

    def stream(self, messages, model_id, image_base64=None, options=None):
        self._fail()
        yield ""

    def _native_tool_turn(self, messages, model_id, tools, max_tokens, require):
        self._fail()

    def check_auth(self) -> None:
        self._probe_auth(self._fail)


def _service(*providers) -> LLMService:
    reg = ProviderRegistry()
    for p in providers:
        reg.register(p)
    return LLMService(reg)


def _bad(status=401) -> RejectingProvider:
    p = RejectingProvider(status, model_id="cloud-cheap")
    p.api_key_env = "ANTHROPIC_API_KEY"
    return p


class TestUtilityFallback:
    def test_utility_call_moves_to_next_ready_model(self):
        bad, good = _bad(), ScriptedProvider(texts=["My title"], model_id="local")
        svc = _service(bad, good)
        assert svc.complete([{"role": "user", "content": "hi"}]) == "My title"
        assert not bad.ready and "ANTHROPIC_API_KEY" in bad.auth_error
        # Later utility calls skip the flagged provider without trying it again.
        good.texts.append("Second")
        assert svc.complete([{"role": "user", "content": "hi"}]) == "Second"
        assert bad.calls == 1

    def test_title_uses_fallback_model(self):
        svc = _service(_bad(), ScriptedProvider(texts=["Weekend weather"], model_id="local"))
        assert svc.generate_title("compare the weather in three cities") == "Weekend weather"

    def test_no_ready_model_raises_clear_error(self):
        svc = _service(_bad())
        with pytest.raises(ProviderAuthError, match="ANTHROPIC_API_KEY in backend/.env"):
            svc.complete([{"role": "user", "content": "hi"}])

    def test_other_errors_do_not_flag_provider(self):
        bad = _bad(status=500)
        svc = _service(bad, ScriptedProvider(model_id="local"))
        with pytest.raises(_HTTPError):
            svc.complete([{"role": "user", "content": "hi"}])
        assert bad.ready

    def test_explicit_model_is_not_swapped(self):
        bad, good = _bad(), ScriptedProvider(texts=["x"], model_id="local")
        svc = _service(bad, good)
        with pytest.raises(ProviderAuthError):
            svc.complete([{"role": "user", "content": "hi"}], model_id="cloud-cheap")
        assert good.texts == ["x"]


class TestChatCalls:
    def test_stream_raises_clear_error_and_flags(self):
        bad = _bad()
        svc = _service(bad)
        with pytest.raises(ProviderAuthError, match="rejected its key"):
            list(svc.stream([{"role": "user", "content": "hi"}], "cloud-cheap"))
        assert not bad.ready

    def test_tool_turn_raises_clear_error(self):
        svc = _service(_bad())
        with pytest.raises(ProviderAuthError):
            svc.complete_with_tools([{"role": "user", "content": "hi"}], "cloud-cheap", [])


class TestStartupCheck:
    def test_rejected_key_flagged(self):
        bad = _bad()
        reg = ProviderRegistry()
        reg.register(bad)
        check_api_keys(reg)
        assert not bad.ready

    def test_network_failure_does_not_flag(self):
        bad = _bad(status=None)
        reg = ProviderRegistry()
        reg.register(bad)
        check_api_keys(reg)
        assert bad.ready

    def test_keyless_server_not_checked(self):
        from src.providers.openai_provider import OpenAICompatibleProvider
        p = OpenAICompatibleProvider("ollama", base_url="http://127.0.0.1:9/v1")
        p.check_auth()  # no key: returns without any request
        assert p.ready


class TestModelsEndpoint:
    def test_flagged_model_reports_error(self, test_client):
        from src import server
        provider = server._registry.providers[0]
        provider.mark_auth_failed()
        try:
            models = test_client.get("/api/models").json()
            flagged = [m for m in models if m["provider"] == provider.provider_name]
            assert flagged and all("rejected its key" in m["error"] for m in flagged)
        finally:
            provider._auth_error = None
