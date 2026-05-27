"""Tests for LLMService — routing, logging, title generation, cheapest-model resolution."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.llm_service import (
    CALL_TYPE_CHAT,
    CALL_TYPE_TITLE,
    LLMService,
    _short,
    _title_fallback,
)
from src.providers.base import ModelOption


# ── Fixtures ────────────────────────────────────────────────────────────

def _make_mock_registry(
    model_id: str = "test-model",
    is_cheapest: bool = True,
    stream_chunks: list[str] | None = None,
    complete_result: str = "completed text",
):
    """Return (registry_mock, provider_mock) with sensible defaults."""
    if stream_chunks is None:
        stream_chunks = ["hello", " world"]

    mock_provider = MagicMock()
    mock_provider.count_tokens.side_effect = lambda text: max(1, len(text) // 4)
    mock_provider.stream.return_value = iter(stream_chunks)
    mock_provider.complete.return_value = complete_result

    cheapest_model = ModelOption(id=model_id, name="Test Model", provider="test", is_cheapest=is_cheapest)

    registry = MagicMock()
    registry.get_provider.return_value = mock_provider
    registry.cheapest_model = cheapest_model

    return registry, mock_provider


# ── _title_fallback() ────────────────────────────────────────────────────

class TestTitleFallback:
    def test_short_text_returned_as_is(self):
        assert _title_fallback("Hello world") == "Hello world"

    def test_exactly_45_chars_returned_as_is(self):
        text = "a" * 45
        assert _title_fallback(text) == text

    def test_long_text_truncated_with_ellipsis(self):
        text = "word " * 20  # well over 45 chars
        result = _title_fallback(text)
        assert result.endswith("…")

    def test_truncation_respects_word_boundary(self):
        text = "short words and more words and more words here"  # > 45 chars
        result = _title_fallback(text)
        # Result must not cut in the middle of a word
        assert not result.rstrip("…").endswith(" ")
        # All prefix words should be complete (no partial words)
        body = result.rstrip("… ")
        assert all(w in text for w in body.split())

    def test_strips_leading_trailing_whitespace(self):
        assert _title_fallback("  hello  ") == "hello"

    def test_empty_string_returns_empty(self):
        assert _title_fallback("") == ""


# ── _short() ────────────────────────────────────────────────────────────

class TestShort:
    def test_strips_org_prefix(self):
        assert _short("google/gemma-4-E4B-it") == "gemma-4-E4B-it"

    def test_no_slash_returns_full_id(self):
        assert _short("local") == "local"

    def test_multiple_slashes_returns_last_segment(self):
        assert _short("org/team/model-name") == "model-name"


# ── LLMService construction and configuration ────────────────────────────

class TestLLMServiceInit:
    def test_default_cheapest_override_is_none(self):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        assert svc._cheapest_override is None

    def test_cheapest_override_stored(self):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry, cheapest_model_id="my-model")
        assert svc._cheapest_override == "my-model"

    def test_set_cheapest_model_updates_override(self):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        svc.set_cheapest_model("new-model")
        assert svc._cheapest_override == "new-model"

    def test_set_cheapest_model_accepts_none(self):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry, cheapest_model_id="old")
        svc.set_cheapest_model(None)
        assert svc._cheapest_override is None


# ── _resolve_cheapest_id() ───────────────────────────────────────────────

class TestResolveCheapestId:
    def test_returns_registry_cheapest_when_no_override(self):
        registry, _ = _make_mock_registry(model_id="registry-cheap")
        svc = LLMService(registry)
        assert svc._resolve_cheapest_id() == "registry-cheap"

    def test_returns_override_when_valid(self):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry, cheapest_model_id="test-model")
        assert svc._resolve_cheapest_id() == "test-model"

    def test_falls_back_to_registry_when_override_not_found(self):
        registry, _ = _make_mock_registry(model_id="registry-cheap")
        registry.get_provider.side_effect = [KeyError("not found"), MagicMock()]
        svc = LLMService(registry, cheapest_model_id="missing-model")
        assert svc._resolve_cheapest_id() == "registry-cheap"

    def test_falls_back_when_override_is_none(self):
        registry, _ = _make_mock_registry(model_id="the-cheapest")
        svc = LLMService(registry, cheapest_model_id=None)
        assert svc._resolve_cheapest_id() == "the-cheapest"


# ── LLMService.stream() ──────────────────────────────────────────────────

class TestLLMServiceStream:
    def test_yields_all_chunks(self):
        registry, _ = _make_mock_registry(stream_chunks=["a", "b", "c"])
        svc = LLMService(registry)
        result = list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))
        assert result == ["a", "b", "c"]

    def test_logs_start_on_call(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))
        assert any("start" in r.getMessage() for r in caplog.records)

    def test_logs_done_after_full_stream(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))
        assert any("done" in r.getMessage() for r in caplog.records)

    def test_log_includes_call_type(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model", call_type="my_call_type"))
        assert any("my_call_type" in r.getMessage() for r in caplog.records)

    def test_log_includes_context_id(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model", context_id="abc12345"))
        assert any("abc12345" in r.getMessage() for r in caplog.records)

    def test_no_context_id_in_log_when_empty(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model", context_id=""))
        # "id=" prefix should not appear when context_id is empty
        assert not any("id=" in r.getMessage() for r in caplog.records)

    def test_logs_error_on_exception(self, caplog):
        registry, provider = _make_mock_registry()
        provider.stream.side_effect = RuntimeError("boom")
        svc = LLMService(registry)
        with caplog.at_level(logging.ERROR, logger="src.llm_service"):
            with pytest.raises(RuntimeError):
                list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))
        assert any("error" in r.getMessage() for r in caplog.records)

    def test_propagates_exception(self):
        registry, provider = _make_mock_registry()
        provider.stream.side_effect = ValueError("provider broke")
        svc = LLMService(registry)
        with pytest.raises(ValueError, match="provider broke"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))

    def test_default_call_type_is_chat(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            list(svc.stream([{"role": "user", "content": "hi"}], "test-model"))
        assert any(CALL_TYPE_CHAT in r.getMessage() for r in caplog.records)

    def test_passes_image_base64_to_provider(self):
        registry, provider = _make_mock_registry()
        provider.stream.return_value = iter(["ok"])
        svc = LLMService(registry)
        list(svc.stream([{"role": "user", "content": "hi"}], "test-model", image_base64="abc123"))
        _, kwargs = provider.stream.call_args
        assert kwargs.get("image_base64") == "abc123"


# ── LLMService.complete() ────────────────────────────────────────────────

class TestLLMServiceComplete:
    def test_returns_provider_result(self):
        registry, _ = _make_mock_registry(complete_result="the answer")
        svc = LLMService(registry)
        result = svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")
        assert result == "the answer"

    def test_uses_cheapest_model_when_no_model_id(self):
        registry, provider = _make_mock_registry(model_id="cheap-model")
        provider.complete.return_value = "cheap result"
        svc = LLMService(registry)
        svc.complete([{"role": "user", "content": "hi"}])
        registry.get_provider.assert_called_with("cheap-model")

    def test_logs_start(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")
        assert any("start" in r.getMessage() for r in caplog.records)

    def test_logs_done(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")
        assert any("done" in r.getMessage() for r in caplog.records)

    def test_logs_error_on_exception(self, caplog):
        registry, provider = _make_mock_registry()
        provider.complete.side_effect = RuntimeError("api down")
        svc = LLMService(registry)
        with caplog.at_level(logging.ERROR, logger="src.llm_service"):
            with pytest.raises(RuntimeError):
                svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")
        assert any("error" in r.getMessage() for r in caplog.records)

    def test_propagates_exception(self):
        registry, provider = _make_mock_registry()
        provider.complete.side_effect = ConnectionError("network fail")
        svc = LLMService(registry)
        with pytest.raises(ConnectionError, match="network fail"):
            svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")

    def test_default_call_type_is_title(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            svc.complete([{"role": "user", "content": "hi"}], model_id="test-model")
        assert any(CALL_TYPE_TITLE in r.getMessage() for r in caplog.records)

    def test_custom_call_type_appears_in_log(self, caplog):
        registry, _ = _make_mock_registry()
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            svc.complete([{"role": "user", "content": "hi"}], model_id="test-model", call_type="guardrail")
        assert any("guardrail" in r.getMessage() for r in caplog.records)


# ── LLMService.count_tokens() ────────────────────────────────────────────

class TestLLMServiceCountTokens:
    def test_delegates_to_provider(self):
        registry, provider = _make_mock_registry()
        # Clear side_effect so return_value is honoured
        provider.count_tokens.side_effect = None
        provider.count_tokens.return_value = 42
        svc = LLMService(registry)
        assert svc.count_tokens("test-model", "hello world") == 42

    def test_passes_correct_text(self):
        registry, provider = _make_mock_registry()
        svc = LLMService(registry)
        svc.count_tokens("test-model", "some text here")
        provider.count_tokens.assert_called_with("some text here")


# ── LLMService.generate_title() ──────────────────────────────────────────

class TestGenerateTitle:
    def test_returns_title_from_llm(self):
        registry, _ = _make_mock_registry(complete_result="Python Basics")
        svc = LLMService(registry)
        assert svc.generate_title("Tell me about Python") == "Python Basics"

    def test_strips_surrounding_quotes(self):
        registry, _ = _make_mock_registry(complete_result='"My Title"')
        svc = LLMService(registry)
        assert svc.generate_title("some message") == "My Title"

    def test_strips_single_quotes(self):
        registry, _ = _make_mock_registry(complete_result="'My Title'")
        svc = LLMService(registry)
        assert svc.generate_title("some message") == "My Title"

    def test_truncates_to_60_chars(self):
        long_title = "A" * 80
        registry, _ = _make_mock_registry(complete_result=long_title)
        svc = LLMService(registry)
        assert len(svc.generate_title("hi")) <= 60

    def test_falls_back_on_empty_llm_response(self):
        registry, _ = _make_mock_registry(complete_result="")
        svc = LLMService(registry)
        # Should use _title_fallback on the original user message
        title = svc.generate_title("Tell me about Python programming")
        assert len(title) > 0

    def test_falls_back_on_llm_exception(self):
        registry, provider = _make_mock_registry()
        provider.complete.side_effect = RuntimeError("api error")
        svc = LLMService(registry)
        # Should not raise — uses truncation fallback
        title = svc.generate_title("Tell me about Python")
        assert isinstance(title, str)
        assert len(title) > 0

    def test_uses_cheapest_model(self):
        registry, provider = _make_mock_registry(model_id="cheap-model")
        provider.complete.return_value = "Short Title"
        svc = LLMService(registry)
        svc.generate_title("some user message")
        registry.get_provider.assert_called_with("cheap-model")

    def test_logs_result(self, caplog):
        registry, _ = _make_mock_registry(complete_result="My Title")
        svc = LLMService(registry)
        with caplog.at_level(logging.INFO, logger="src.llm_service"):
            svc.generate_title("some message")
        assert any("My Title" in r.getMessage() for r in caplog.records)

    def test_truncates_user_message_to_300_chars(self):
        """Long user messages must not be sent in full to the title prompt."""
        registry, provider = _make_mock_registry(complete_result="Title")
        provider.complete.return_value = "Title"
        svc = LLMService(registry)
        svc.generate_title("x" * 1000)
        messages_sent = provider.complete.call_args[0][0]
        user_msg = next(m for m in messages_sent if m["role"] == "user")
        # The content is "Title for: " + truncated message; inner text ≤ 300
        inner = user_msg["content"].replace("Title for: ", "", 1)
        assert len(inner) <= 300


# ── Call-type constants ──────────────────────────────────────────────────

class TestCallTypeConstants:
    def test_call_type_chat_is_snake_case(self):
        assert CALL_TYPE_CHAT == CALL_TYPE_CHAT.lower()
        assert " " not in CALL_TYPE_CHAT

    def test_call_type_title_is_snake_case(self):
        assert CALL_TYPE_TITLE == CALL_TYPE_TITLE.lower()
        assert " " not in CALL_TYPE_TITLE

    def test_constants_are_distinct(self):
        assert CALL_TYPE_CHAT != CALL_TYPE_TITLE
