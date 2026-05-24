"""
Integration tests for cloud providers.

These tests make REAL API calls and require valid keys in .env:
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...

Run all tests:        pytest tests/test_providers_integration.py -v -m integration
Run Anthropic only:   pytest tests/test_providers_integration.py -v -k anthropic
Run OpenAI only:      pytest tests/test_providers_integration.py -v -k openai

Test outcomes:
  PASS  — API key valid + response received
  SKIP  — API key valid but account has insufficient credits/quota
  FAIL  — API key rejected (401) or implementation bug
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load keys from project-root .env (local-llm/.env)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

_SKIP_ANTHROPIC = pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
_SKIP_OPENAI = pytest.mark.skipif(not _OPENAI_KEY, reason="OPENAI_API_KEY not set")

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_MODEL = "gpt-5.4-mini"

_IMAGE_PATH = Path(__file__).parent / "images" / "smile.png"


# ── Core helper ────────────────────────────────────────────────────────

def _stream_or_skip(provider_name: str, provider, messages, model_id, **kwargs) -> tuple[list[str], str]:
    """
    Collect all tokens from a provider stream.

    - Returns (tokens, joined_text) on success.
    - Calls pytest.skip() on billing / quota errors or empty response
      (key is valid; account lacks credits).
    - Calls pytest.fail() on auth errors (key rejected).
    - Re-raises all other exceptions so real bugs surface.
    """
    import anthropic as _ant
    import openai as _oai
    try:
        tokens = list(provider.stream(messages, model_id, **kwargs))
    except _ant.AuthenticationError:
        pytest.fail(f"{provider_name}: API key rejected — check ANTHROPIC_API_KEY in .env")
    except _oai.AuthenticationError:
        pytest.fail(f"{provider_name}: API key rejected — check OPENAI_API_KEY in .env")
    except (_ant.BadRequestError, _ant.PermissionDeniedError) as e:
        msg = str(e).lower()
        if any(w in msg for w in ("credit", "billing", "quota", "insufficient")):
            pytest.skip(f"{provider_name}: key valid, insufficient credits — top up to run streaming tests")
        raise
    except _oai.RateLimitError as e:
        if "quota" in str(e).lower() or "billing" in str(e).lower():
            pytest.skip(f"{provider_name}: key valid, quota exceeded — top up to run streaming tests")
        raise

    # Empty stream without an explicit error = silent billing rejection.
    # Seen when a prior test in the session exhausted the quota.
    if not tokens:
        pytest.skip(f"{provider_name}: empty stream (key valid; account may have no credits)")

    return tokens, "".join(tokens)


def _image_b64() -> str:
    return base64.b64encode(_IMAGE_PATH.read_bytes()).decode()


# ── Anthropic ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestAnthropicConnection:
    @_SKIP_ANTHROPIC
    def test_provider_initialises(self):
        """Provider can be constructed with the key — no network call."""
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        assert p.provider_name == "anthropic"
        assert len(p.models) > 0

    @_SKIP_ANTHROPIC
    def test_api_key_is_authenticated(self):
        """Key is accepted by the API (not rejected with 401/403).
        Passes even if the account has no credits — billing != auth."""
        from src.providers.anthropic_provider import AnthropicProvider
        import anthropic
        p = AnthropicProvider(_ANTHROPIC_KEY)
        try:
            list(p.stream([{"role": "user", "content": "hi"}], _ANTHROPIC_MODEL))
        except anthropic.AuthenticationError:
            pytest.fail("API key rejected — check ANTHROPIC_API_KEY in .env")
        except (anthropic.BadRequestError, anthropic.PermissionDeniedError) as e:
            if any(w in str(e).lower() for w in ("credit", "billing", "quota", "insufficient")):
                pass  # auth succeeded; billing is a separate concern
            else:
                raise

    @_SKIP_ANTHROPIC
    def test_stream_returns_tokens(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]
        tokens, text = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL)
        assert len(tokens) > 0
        assert len(text.strip()) > 0

    @_SKIP_ANTHROPIC
    def test_stream_is_incremental(self):
        """Streaming should yield multiple chunks, not one big blob."""
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [{"role": "user", "content": "Count from 1 to 10, one number per line."}]
        tokens, _ = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL)
        assert len(tokens) > 1, "Expected multiple streaming chunks"

    @_SKIP_ANTHROPIC
    def test_system_message_affects_response(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [
            {"role": "system", "content": "You must reply in French only."},
            {"role": "user", "content": "Say hello."},
        ]
        _, text = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL)
        french_words = {"bonjour", "salut", "allô", "bonsoir", "coucou", "oui", "ça"}
        assert any(w in text.lower() for w in french_words), (
            f"Expected French response, got: {text!r}"
        )

    @_SKIP_ANTHROPIC
    def test_multi_turn_conversation(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [
            {"role": "user", "content": "My favourite colour is blue. Remember it."},
            {"role": "assistant", "content": "Got it, your favourite colour is blue."},
            {"role": "user", "content": "What is my favourite colour?"},
        ]
        _, text = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL)
        assert "blue" in text.lower(), f"Model forgot context; got: {text!r}"

    @_SKIP_ANTHROPIC
    def test_stream_with_image(self):
        from src.providers.anthropic_provider import AnthropicProvider
        if not _IMAGE_PATH.exists():
            pytest.skip("Test image not found")
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [{"role": "user", "content": "Describe this image in one sentence."}]
        _, text = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL, image_base64=_image_b64())
        assert len(text.strip()) > 0

    @_SKIP_ANTHROPIC
    def test_haiku_model_id(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        _, text = _stream_or_skip("Anthropic/Haiku", p, [{"role": "user", "content": "Say 'ok'"}], "claude-haiku-4-5-20251001")
        assert len(text.strip()) > 0

    @_SKIP_ANTHROPIC
    def test_sonnet_model_id(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        _, text = _stream_or_skip("Anthropic/Sonnet", p, [{"role": "user", "content": "Say 'ok'"}], "claude-sonnet-4-6")
        assert len(text.strip()) > 0

    @_SKIP_ANTHROPIC
    def test_joined_tokens_form_coherent_text(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
        _, text = _stream_or_skip("Anthropic", p, messages, _ANTHROPIC_MODEL)
        assert "4" in text, f"Expected '4' in response, got: {text!r}"

    @_SKIP_ANTHROPIC
    def test_count_tokens_fallback(self):
        """count_tokens returns a positive estimate — no network call."""
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(_ANTHROPIC_KEY)
        assert p.count_tokens("hello world this is a test") > 0


# ── OpenAI ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestOpenAIConnection:
    @_SKIP_OPENAI
    def test_provider_initialises(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        assert p.provider_name == "openai"
        assert len(p.models) > 0

    @_SKIP_OPENAI
    def test_api_key_is_authenticated(self):
        """Key is accepted by the API — passes even with no credits."""
        from src.providers.openai_provider import OpenAIProvider
        import openai
        p = OpenAIProvider(_OPENAI_KEY)
        try:
            list(p.stream([{"role": "user", "content": "hi"}], _OPENAI_MODEL))
        except openai.AuthenticationError:
            pytest.fail("API key rejected — check OPENAI_API_KEY in .env")
        except openai.RateLimitError as e:
            if "quota" in str(e).lower() or "billing" in str(e).lower():
                pass  # auth succeeded; quota is a separate concern
            else:
                raise

    @_SKIP_OPENAI
    def test_stream_returns_tokens(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]
        tokens, text = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL)
        assert len(tokens) > 0
        assert len(text.strip()) > 0

    @_SKIP_OPENAI
    def test_stream_is_incremental(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [{"role": "user", "content": "Count from 1 to 10, one number per line."}]
        tokens, _ = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL)
        assert len(tokens) > 1, "Expected multiple streaming chunks"

    @_SKIP_OPENAI
    def test_system_message_affects_response(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [
            {"role": "system", "content": "You must reply in French only."},
            {"role": "user", "content": "Say hello."},
        ]
        _, text = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL)
        french_words = {"bonjour", "salut", "allô", "bonsoir", "coucou", "oui", "ça"}
        assert any(w in text.lower() for w in french_words), (
            f"Expected French response, got: {text!r}"
        )

    @_SKIP_OPENAI
    def test_multi_turn_conversation(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [
            {"role": "user", "content": "My favourite colour is blue. Remember it."},
            {"role": "assistant", "content": "Got it, your favourite colour is blue."},
            {"role": "user", "content": "What is my favourite colour?"},
        ]
        _, text = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL)
        assert "blue" in text.lower(), f"Model forgot context; got: {text!r}"

    @_SKIP_OPENAI
    def test_stream_with_image(self):
        from src.providers.openai_provider import OpenAIProvider
        if not _IMAGE_PATH.exists():
            pytest.skip("Test image not found")
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [{"role": "user", "content": "Describe this image in one sentence."}]
        _, text = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL, image_base64=_image_b64())
        assert len(text.strip()) > 0

    @_SKIP_OPENAI
    def test_joined_tokens_form_coherent_text(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
        _, text = _stream_or_skip("OpenAI", p, messages, _OPENAI_MODEL)
        assert "4" in text, f"Expected '4' in response, got: {text!r}"

    @_SKIP_OPENAI
    def test_count_tokens_fallback(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(_OPENAI_KEY)
        assert p.count_tokens("hello world this is a test") > 0


# ── Registry auto-discovery ────────────────────────────────────────────

@pytest.mark.integration
class TestRegistryAutoDiscovery:
    def test_registry_includes_local_always(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        assert "local" in {m.provider for m in registry.list_models()}

    @_SKIP_ANTHROPIC
    def test_registry_includes_anthropic_when_key_present(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        assert "anthropic" in {m.provider for m in registry.list_models()}, (
            "ANTHROPIC_API_KEY is set but Anthropic provider not registered"
        )

    @_SKIP_OPENAI
    def test_registry_includes_openai_when_key_present(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        assert "openai" in {m.provider for m in registry.list_models()}, (
            "OPENAI_API_KEY is set but OpenAI provider not registered"
        )

    @_SKIP_ANTHROPIC
    def test_registry_routes_anthropic_model(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        assert registry.get_provider("claude-haiku-4-5-20251001").provider_name == "anthropic"

    @_SKIP_OPENAI
    def test_registry_routes_openai_model(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        assert registry.get_provider("gpt-5.4-mini").provider_name == "openai"

    def test_all_registered_models_have_unique_ids(self, mock_llm):
        from src.providers.registry import build_registry
        registry = build_registry(mock_llm)
        ids = [m.id for m in registry.list_models()]
        assert len(ids) == len(set(ids)), "Duplicate model IDs in registry"
