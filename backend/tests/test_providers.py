"""
Unit tests for the provider abstraction layer.

All external SDK calls (anthropic, openai) are mocked — no network required.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.providers.base import BaseProvider, ModelOption
from src.providers.registry import ProviderRegistry


# ── Helpers ────────────────────────────────────────────────────────────

def _make_stream_ctx(tokens: list[str]):
    """Build a mock Anthropic stream context manager."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.text_stream = iter(tokens)
    return ctx


def _make_oai_chunks(tokens: list[str]):
    """Build a list of mock OpenAI stream chunks."""
    chunks = []
    for t in tokens:
        chunk = MagicMock()
        chunk.choices[0].delta.content = t
        chunks.append(chunk)
    return iter(chunks)


# ── ModelOption ────────────────────────────────────────────────────────

class TestModelOption:
    def test_fields(self):
        m = ModelOption(id="model-id", name="My Model", provider="myprovider")
        assert m.id == "model-id"
        assert m.name == "My Model"
        assert m.provider == "myprovider"

    def test_equality(self):
        a = ModelOption(id="x", name="X", provider="p")
        b = ModelOption(id="x", name="X", provider="p")
        assert a == b

    def test_inequality(self):
        a = ModelOption(id="x", name="X", provider="p")
        b = ModelOption(id="y", name="Y", provider="p")
        assert a != b


# ── BaseProvider default count_tokens ──────────────────────────────────

class TestBaseProviderDefaults:
    def test_count_tokens_rough_estimate(self):
        class MinimalProvider(BaseProvider):
            @property
            def provider_name(self): return "test"
            @property
            def models(self): return []
            def stream(self, messages, model_id, image_base64=None): return iter([])
            def complete(self, messages, model_id): return ""

        p = MinimalProvider()
        assert p.count_tokens("hello world") > 0
        assert p.count_tokens("a" * 400) > p.count_tokens("a" * 40)


# ── LocalProvider ──────────────────────────────────────────────────────

class TestLocalProvider:
    def test_provider_name(self, mock_llm):
        from src.providers.local import LocalProvider
        p = LocalProvider(mock_llm)
        assert p.provider_name == "local"

    def test_models_uses_repo_id(self, mock_llm):
        from src.providers.local import LocalProvider
        p = LocalProvider(mock_llm)
        models = p.models
        assert len(models) == 1
        assert models[0].id == "google/gemma-4-E4B-it"
        assert models[0].provider == "local"

    def test_model_name_is_short_form(self, mock_llm):
        from src.providers.local import LocalProvider
        p = LocalProvider(mock_llm)
        assert "/" not in p.models[0].name

    def test_is_cheapest_true(self, mock_llm):
        from src.providers.local import LocalProvider
        p = LocalProvider(mock_llm)
        assert p.models[0].is_cheapest is True

    def test_stream_delegates_to_llm(self, mock_llm):
        from src.providers.local import LocalProvider
        mock_llm.stream = MagicMock(return_value=iter(["hi", " there"]))
        p = LocalProvider(mock_llm)
        tokens = list(p.stream([{"role": "user", "content": "hello"}], mock_llm.repo_id))
        assert tokens == ["hi", " there"]
        mock_llm.stream.assert_called_once()

    def test_complete_delegates_to_llm(self, mock_llm):
        from src.providers.local import LocalProvider
        mock_llm.generate = MagicMock(return_value="Generated text")
        p = LocalProvider(mock_llm)
        result = p.complete([{"role": "user", "content": "hello"}], mock_llm.repo_id)
        assert result == "Generated text"
        mock_llm.generate.assert_called_once()

    def test_complete_raises_when_not_loaded(self, mock_llm):
        from src.providers.local import LocalProvider
        mock_llm.model = None  # mark as not loaded
        p = LocalProvider(mock_llm)
        with pytest.raises(RuntimeError, match="still loading"):
            p.complete([{"role": "user", "content": "hello"}], mock_llm.repo_id)

    def test_stream_with_image_calls_vision(self, mock_llm):
        from src.providers.local import LocalProvider
        mock_llm.stream_with_image = MagicMock(return_value=iter(["a cat"]))
        p = LocalProvider(mock_llm)
        img_b64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 16).decode()
        with patch("src.providers.local.PILImage") as mock_pil:
            mock_pil.open.return_value.convert.return_value = MagicMock()
            tokens = list(p.stream([{"role": "user", "content": "what is this?"}], mock_llm.repo_id, image_base64=img_b64))
        assert tokens == ["a cat"]
        mock_llm.stream_with_image.assert_called_once()

    def test_count_tokens_delegates_to_llm(self, mock_llm):
        from src.providers.local import LocalProvider
        mock_llm.count_tokens = MagicMock(return_value=42)
        p = LocalProvider(mock_llm)
        assert p.count_tokens("some text") == 42


# ── AnthropicProvider ──────────────────────────────────────────────────

class TestAnthropicProvider:
    def test_provider_name(self):
        with patch("anthropic.Anthropic"):
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("test-key")
            assert p.provider_name == "anthropic"

    def test_models_list(self):
        with patch("anthropic.Anthropic"):
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("test-key")
            ids = [m.id for m in p.models]
            assert "claude-haiku-5-5" in ids
            assert "claude-sonnet-5-5" in ids
            # assert "claude-opus-4-7" in ids  # expensive — uncomment when re-enabling

    def test_all_models_have_anthropic_provider(self):
        with patch("anthropic.Anthropic"):
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("test-key")
            assert all(m.provider == "anthropic" for m in p.models)

    def test_stream_yields_tokens(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_cls.return_value.messages.stream.return_value = _make_stream_ctx(["Hello", " world"])
            p = AnthropicProvider("test-key")
            tokens = list(p.stream([{"role": "user", "content": "hi"}], "claude-haiku-5-5"))
            assert tokens == ["Hello", " world"]

    def test_stream_separates_system_message(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_client = mock_cls.return_value
            mock_client.messages.stream.return_value = _make_stream_ctx(["ok"])
            p = AnthropicProvider("test-key")
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ]
            list(p.stream(messages, "claude-haiku-5-5"))
            call_kwargs = mock_client.messages.stream.call_args.kwargs
            assert call_kwargs["system"][0]["text"] == "Be brief."
            assert all(m["role"] != "system" for m in call_kwargs["messages"])

    def test_stream_without_system_message(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_client = mock_cls.return_value
            mock_client.messages.stream.return_value = _make_stream_ctx(["ok"])
            p = AnthropicProvider("test-key")
            list(p.stream([{"role": "user", "content": "hi"}], "claude-haiku-5-5"))
            call_kwargs = mock_client.messages.stream.call_args.kwargs
            assert "system" not in call_kwargs

    def test_stream_with_image_injects_content_block(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_client = mock_cls.return_value
            mock_client.messages.stream.return_value = _make_stream_ctx(["a cat"])
            p = AnthropicProvider("test-key")
            img_b64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 16).decode()
            messages = [{"role": "user", "content": "what is this?"}]
            list(p.stream(messages, "claude-haiku-5-5", image_base64=img_b64))
            call_kwargs = mock_client.messages.stream.call_args.kwargs
            last_msg = call_kwargs["messages"][-1]
            assert isinstance(last_msg["content"], list)
            types = [block["type"] for block in last_msg["content"]]
            assert "image" in types
            assert "text" in types

    def test_stream_jpeg_media_type_detected(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_client = mock_cls.return_value
            mock_client.messages.stream.return_value = _make_stream_ctx(["ok"])
            p = AnthropicProvider("test-key")
            jpeg_b64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 16).decode()
            list(p.stream([{"role": "user", "content": "hi"}], "claude-haiku-5-5", image_base64=jpeg_b64))
            call_kwargs = mock_client.messages.stream.call_args.kwargs
            img_block = next(b for b in call_kwargs["messages"][-1]["content"] if b["type"] == "image")
            assert img_block["source"]["media_type"] == "image/jpeg"

    def test_stream_png_media_type_detected(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_client = mock_cls.return_value
            mock_client.messages.stream.return_value = _make_stream_ctx(["ok"])
            p = AnthropicProvider("test-key")
            png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
            list(p.stream([{"role": "user", "content": "hi"}], "claude-haiku-5-5", image_base64=png_b64))
            call_kwargs = mock_client.messages.stream.call_args.kwargs
            img_block = next(b for b in call_kwargs["messages"][-1]["content"] if b["type"] == "image")
            assert img_block["source"]["media_type"] == "image/png"

    def test_haiku_is_cheapest(self):
        with patch("anthropic.Anthropic"):
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("test-key")
            haiku = next(m for m in p.models if "haiku" in m.id)
            assert haiku.is_cheapest is True

    def test_non_haiku_models_not_cheapest(self):
        with patch("anthropic.Anthropic"):
            from src.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("test-key")
            others = [m for m in p.models if "haiku" not in m.id]
            assert all(not m.is_cheapest for m in others)

    def test_complete_returns_text(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(type="text", text="Short answer")]
            mock_cls.return_value.messages.create.return_value = mock_msg
            p = AnthropicProvider("test-key")
            result = p.complete([{"role": "user", "content": "hi"}], "claude-haiku-5-5")
            assert result == "Short answer"

    def test_complete_separates_system_message(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text="ok")]
            mock_cls.return_value.messages.create.return_value = mock_msg
            p = AnthropicProvider("test-key")
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ]
            p.complete(messages, "claude-haiku-5-5")
            call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
            assert call_kwargs["system"][0]["text"] == "Be brief."
            assert all(m["role"] != "system" for m in call_kwargs["messages"])

    def test_complete_uses_non_streaming_api(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text="ok")]
            mock_cls.return_value.messages.create.return_value = mock_msg
            p = AnthropicProvider("test-key")
            p.complete([{"role": "user", "content": "hi"}], "claude-haiku-5-5")
            # create() not stream() should be called
            mock_cls.return_value.messages.create.assert_called_once()
            mock_cls.return_value.messages.stream.assert_not_called()

    def test_uses_api_key(self):
        with patch("anthropic.Anthropic") as mock_cls:
            from src.providers.anthropic_provider import AnthropicProvider
            AnthropicProvider("my-secret-key")
            assert mock_cls.call_args.kwargs["api_key"] == "my-secret-key"


# ── OpenAIProvider ─────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_provider_name(self):
        with patch("openai.OpenAI"):
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("test-key")
            assert p.provider_name == "openai"

    def test_models_list(self):
        with patch("openai.OpenAI"):
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("test-key")
            ids = [m.id for m in p.models]
            # assert "gpt-5.5" in ids      # expensive — uncomment when re-enabling
            # assert "gpt-5.5-pro" in ids  # expensive — uncomment when re-enabling
            assert "gpt-5.4-mini" in ids

    def test_all_models_have_openai_provider(self):
        with patch("openai.OpenAI"):
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("test-key")
            assert all(m.provider == "openai" for m in p.models)

    def test_stream_yields_tokens(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_cls.return_value.chat.completions.create.return_value = _make_oai_chunks(["Hello", " world"])
            p = OpenAIProvider("test-key")
            tokens = list(p.stream([{"role": "user", "content": "hi"}], "gpt-5.4-mini"))
            assert tokens == ["Hello", " world"]

    def test_stream_passes_model_id(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _make_oai_chunks(["ok"])
            p = OpenAIProvider("test-key")
            list(p.stream([{"role": "user", "content": "hi"}], "gpt-5.4-mini"))
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-5.4-mini"
            assert call_kwargs["stream"] is True

    def test_stream_preserves_message_roles(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _make_oai_chunks(["ok"])
            p = OpenAIProvider("test-key")
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "bye"},
            ]
            list(p.stream(messages, "gpt-5.4-mini"))
            sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
            assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]

    def test_stream_with_image_uses_image_url(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _make_oai_chunks(["a dog"])
            p = OpenAIProvider("test-key")
            img_b64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 16).decode()
            messages = [{"role": "user", "content": "what is this?"}]
            list(p.stream(messages, "gpt-5.4-mini", image_base64=img_b64))
            sent = mock_client.chat.completions.create.call_args.kwargs["messages"]
            last_content = sent[-1]["content"]
            assert isinstance(last_content, list)
            types = [block["type"] for block in last_content]
            assert "image_url" in types
            assert "text" in types

    def test_stream_skips_none_content_chunks(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_client = mock_cls.return_value
            chunk_none = MagicMock()
            chunk_none.choices[0].delta.content = None
            chunk_text = MagicMock()
            chunk_text.choices[0].delta.content = "hello"
            mock_client.chat.completions.create.return_value = iter([chunk_none, chunk_text])
            p = OpenAIProvider("test-key")
            tokens = list(p.stream([{"role": "user", "content": "hi"}], "gpt-5.4-mini"))
            assert tokens == ["hello"]

    def test_gpt_mini_is_cheapest(self):
        with patch("openai.OpenAI"):
            from src.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("test-key")
            mini = next(m for m in p.models if "mini" in m.id)
            assert mini.is_cheapest is True

    def test_complete_returns_text(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Short answer"
            mock_cls.return_value.chat.completions.create.return_value = mock_response
            p = OpenAIProvider("test-key")
            result = p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
            assert result == "Short answer"

    def test_complete_uses_non_streaming_api(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "ok"
            mock_cls.return_value.chat.completions.create.return_value = mock_response
            p = OpenAIProvider("test-key")
            p.complete([{"role": "user", "content": "hi"}], "gpt-5.4-mini")
            call_kwargs = mock_cls.return_value.chat.completions.create.call_args.kwargs
            assert call_kwargs["stream"] is False

    def test_uses_api_key(self):
        with patch("src.providers.openai_provider.OpenAI") as mock_cls:
            from src.providers.openai_provider import OpenAIProvider
            OpenAIProvider("my-secret-key")
            assert mock_cls.call_args.kwargs["api_key"] == "my-secret-key"


# ── ProviderRegistry ───────────────────────────────────────────────────

class TestProviderRegistry:
    def _make_provider(self, provider_name: str, model_ids: list[str], cheapest: str | None = None) -> BaseProvider:
        class FakeProvider(BaseProvider):
            @property
            def provider_name(self): return provider_name
            @property
            def models(self):
                return [
                    ModelOption(id=m, name=m, provider=provider_name, is_cheapest=(m == cheapest))
                    for m in model_ids
                ]
            def stream(self, messages, model_id, image_base64=None): return iter([])
            def complete(self, messages, model_id): return "ok"
        return FakeProvider()

    def test_register_and_list(self):
        registry = ProviderRegistry()
        p = self._make_provider("test", ["model-a", "model-b"])
        registry.register(p)
        ids = [m.id for m in registry.list_models()]
        assert "model-a" in ids
        assert "model-b" in ids

    def test_multiple_providers(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("p1", ["m1"]))
        registry.register(self._make_provider("p2", ["m2", "m3"]))
        assert len(registry.list_models()) == 3

    def test_get_provider_correct_routing(self):
        registry = ProviderRegistry()
        p1 = self._make_provider("first", ["alpha"])
        p2 = self._make_provider("second", ["beta"])
        registry.register(p1)
        registry.register(p2)
        assert registry.get_provider("alpha").provider_name == "first"
        assert registry.get_provider("beta").provider_name == "second"

    def test_get_provider_raises_for_unknown(self):
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="unknown-model"):
            registry.get_provider("unknown-model")

    def test_default_model_id_is_first_registered(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("first", ["first-model"]))
        registry.register(self._make_provider("second", ["second-model"]))
        assert registry.default_model_id == "first-model"

    def test_list_models_preserves_registration_order(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("a", ["m1"]))
        registry.register(self._make_provider("b", ["m2"]))
        registry.register(self._make_provider("c", ["m3"]))
        assert [m.id for m in registry.list_models()] == ["m1", "m2", "m3"]

    def test_cheapest_model_returns_flagged_model(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("test", ["m1", "m2"], cheapest="m2"))
        assert registry.cheapest_model.id == "m2"

    def test_cheapest_model_returns_first_flagged_across_providers(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("first", ["a"], cheapest="a"))
        registry.register(self._make_provider("second", ["b"], cheapest="b"))
        # Registration order: first provider's cheapest wins
        assert registry.cheapest_model.id == "a"

    def test_cheapest_model_falls_back_to_default_when_none_flagged(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("test", ["m1", "m2"]))  # none marked cheapest
        assert registry.cheapest_model.id == "m1"  # first model of first provider

    def test_cheapest_model_is_cheapest_flag_set(self):
        registry = ProviderRegistry()
        registry.register(self._make_provider("test", ["cheap", "expensive"], cheapest="cheap"))
        assert registry.cheapest_model.is_cheapest is True


# ── build_registry ─────────────────────────────────────────────────────

class TestBuildRegistry:
    def test_always_includes_local(self, mock_llm, tmp_path):
        from src.providers.registry import build_registry
        cat = tmp_path / "m.yaml"
        cat.write_text("providers:\n  local: {type: builtin}\n")
        registry = build_registry(mock_llm, cat)
        providers = {m.provider for m in registry.list_models()}
        assert "local" in providers

    def test_anthropic_registered_when_key_present(self, mock_llm, tmp_path):
        from src.providers.registry import build_registry
        cat = tmp_path / "m.yaml"
        cat.write_text("providers:\n  anthropic:\n    type: anthropic\n    api_key_env: ANTHROPIC_API_KEY\n"
                       "    models: [{id: claude-haiku-5-5, input_price: 0.1, output_price: 0.5}]\n")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}), patch("anthropic.Anthropic") as cls:
            registry = build_registry(mock_llm, cat)
            cls.assert_called_once_with(api_key="sk-ant-test")
        assert registry.get_model("claude-haiku-5-5").input_price == 0.1

    def test_openai_registered_when_key_present(self, mock_llm, tmp_path):
        from src.providers.registry import build_registry
        cat = tmp_path / "m.yaml"
        cat.write_text("providers:\n  openai:\n    type: openai\n    api_key_env: OPENAI_API_KEY\n"
                       "    models: [gpt-5.4-mini]\n")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai-test"}), \
                patch("src.providers.openai_provider.OpenAI") as cls:
            registry = build_registry(mock_llm, cat)
            assert cls.call_args.kwargs["api_key"] == "sk-openai-test"
        assert registry.get_provider("gpt-5.4-mini").provider_name == "openai"

    def test_cloud_providers_not_registered_without_keys(self, mock_llm, tmp_path):
        from src.providers.registry import build_registry
        cat = tmp_path / "m.yaml"
        cat.write_text("providers:\n  local: {type: builtin}\n"
                       "  anthropic: {type: anthropic, api_key_env: ANTHROPIC_API_KEY, models: [a]}\n"
                       "  openai: {type: openai, api_key_env: OPENAI_API_KEY, models: [b]}\n")
        env = {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}
        with patch.dict("os.environ", env):
            registry = build_registry(mock_llm, cat)
        providers = {m.provider for m in registry.list_models()}
        assert "anthropic" not in providers
        assert "openai" not in providers


# ── /api/models endpoint ───────────────────────────────────────────────

class TestApiModelsEndpoint:
    def test_returns_200(self, test_client):
        resp = test_client.get("/api/models")
        assert resp.status_code == 200

    def test_returns_list(self, test_client):
        resp = test_client.get("/api/models")
        assert isinstance(resp.json(), list)

    def test_each_model_has_required_fields(self, test_client):
        resp = test_client.get("/api/models")
        for model in resp.json():
            assert "id" in model
            assert "name" in model
            assert "provider" in model

    def test_local_model_present(self, test_client):
        resp = test_client.get("/api/models")
        providers = {m["provider"] for m in resp.json()}
        assert "local" in providers

    def test_503_when_registry_not_ready(self):
        from src import server
        original = server._registry
        server._registry = None
        try:
            from fastapi.testclient import TestClient
            client = TestClient(server.app, raise_server_exceptions=False)
            resp = client.get("/api/models")
            assert resp.status_code == 503
        finally:
            server._registry = original


# ── /api/chats/{id}/message with model_id ─────────────────────────────

class TestSendMessageWithModelId:
    def test_unknown_model_returns_400(self, test_client):
        chat = test_client.post("/api/chats").json()
        resp = test_client.post(
            f"/api/chats/{chat['id']}/message",
            json={"content": "hi", "model_id": "nonexistent-model-xyz"},
        )
        assert resp.status_code == 400

    def test_valid_local_model_id_accepted(self, test_client, mock_llm):
        mock_llm.stream = MagicMock(return_value=iter(["hello"]))
        chat = test_client.post("/api/chats").json()
        model_id = "google/gemma-4-E4B-it"
        resp = test_client.post(
            f"/api/chats/{chat['id']}/message",
            json={"content": "hi", "model_id": model_id},
        )
        assert resp.status_code == 200

    def test_null_model_id_uses_default(self, test_client, mock_llm):
        mock_llm.stream = MagicMock(return_value=iter(["hello"]))
        chat = test_client.post("/api/chats").json()
        resp = test_client.post(
            f"/api/chats/{chat['id']}/message",
            json={"content": "hi", "model_id": None},
        )
        assert resp.status_code == 200


class TestCheapestModelReadiness:
    def _provider(self, name: str, model_id: str, ready: bool) -> BaseProvider:
        class P(BaseProvider):
            provider_name = name
            models = [ModelOption(id=model_id, name=model_id, provider=name, is_cheapest=True)]
            def stream(self, *a, **k): return iter([])
            def complete(self, messages, model_id): return ""
        P.ready = property(lambda self: ready)
        return P()

    def test_skips_provider_that_is_not_ready(self):
        registry = ProviderRegistry()
        registry.register(self._provider("local", "gemma", ready=False))
        registry.register(self._provider("anthropic", "haiku", ready=True))
        assert registry.cheapest_model.id == "haiku"

    def test_prefers_first_ready_provider(self):
        registry = ProviderRegistry()
        registry.register(self._provider("local", "gemma", ready=True))
        registry.register(self._provider("anthropic", "haiku", ready=True))
        assert registry.cheapest_model.id == "gemma"

    def test_falls_back_to_first_flagged_when_none_ready(self):
        registry = ProviderRegistry()
        registry.register(self._provider("local", "gemma", ready=False))
        assert registry.cheapest_model.id == "gemma"

    def test_local_provider_ready_tracks_model_load(self, mock_llm):
        from src.providers.local import LocalProvider
        assert LocalProvider(mock_llm).ready is True
        mock_llm.model = None
        assert LocalProvider(mock_llm).ready is False

    def test_llm_service_ignores_override_that_is_not_ready(self):
        from src.llm_service import LLMService
        registry = ProviderRegistry()
        registry.register(self._provider("local", "gemma", ready=False))
        registry.register(self._provider("anthropic", "haiku", ready=True))
        assert LLMService(registry, cheapest_model_id="gemma")._resolve_cheapest_id() == "haiku"


class TestCheapestFallback:
    def test_uses_cheapest_ready_model_when_flagged_ones_are_not_ready(self):
        from src.providers.registry import ProviderRegistry
        from tests.fakes import ScriptedProvider

        class NotReady(ScriptedProvider):
            ready = False

        reg = ProviderRegistry()
        reg.register(NotReady(model_id="local-loading"))
        free = ScriptedProvider(model_id="ollama-model", price=(0.0, 0.0))
        free.models[0].is_cheapest = False
        reg.register(free)
        assert reg.cheapest_model.id == "ollama-model"
