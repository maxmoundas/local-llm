"""Tests for the FastAPI server endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_ok_when_loaded(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_returns_model_name(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert data["model"] == "google/gemma-4-E4B-it"

    def test_health_loading_when_not_ready(self):
        from unittest.mock import MagicMock
        from src import server
        original = server._llm
        server._llm = MagicMock(is_loaded=False, repo_id="google/gemma-4-E4B-it")
        try:
            data = TestClient(server.app).get("/health").json()
            assert data["status"] == "loading" and data["local_model_enabled"] is True
        finally:
            server._llm = original

    def test_health_ok_when_local_model_disabled(self):
        from src import server
        original = server._llm
        server._llm = None  # model.enabled: false → never created
        try:
            data = TestClient(server.app).get("/health").json()
            assert data["status"] == "ok"
            assert data["local_model_enabled"] is False and data["local_model_loaded"] is False
            resp = TestClient(server.app).post("/v1/vision", json={"image_base64": "x", "prompt": "hi"})
            assert resp.status_code == 503 and "turned off" in resp.json()["detail"]
        finally:
            server._llm = original


class TestLocalModelConfig:
    def test_disabled_by_default(self, monkeypatch):
        from src.config import load_config, local_model_enabled
        monkeypatch.delenv("LOCAL_MODEL_ENABLED", raising=False)
        assert local_model_enabled(load_config()) is False

    @pytest.mark.parametrize("val,expected", [("true", True), ("1", True), ("yes", True),
                                              ("false", False), ("0", False), ("", False)])
    def test_env_override(self, monkeypatch, val, expected):
        from src.config import load_config, local_model_enabled
        monkeypatch.setenv("LOCAL_MODEL_ENABLED", val)
        assert local_model_enabled(load_config()) is expected

    def test_missing_key_means_disabled(self):
        from src.config import local_model_enabled
        assert local_model_enabled({"model": {"repo_id": "x"}}) is False

    def test_registry_without_local_model(self, tmp_path, monkeypatch):
        from src.providers.registry import build_registry
        monkeypatch.setenv("TEST_KEY", "k")
        cat = tmp_path / "models.yaml"
        cat.write_text("providers:\n  local: {type: builtin}\n"
                       "  cloud: {type: openai_compatible, api_key_env: TEST_KEY, base_url: http://127.0.0.1:9/v1,"
                       " models: [{id: m1}]}\n")
        reg = build_registry(None, cat)
        assert [m.id for m in reg.list_models()] == ["m1"]

    def test_no_models_at_all_explains_how_to_fix(self, tmp_path):
        from src.providers.registry import build_registry
        cat = tmp_path / "models.yaml"
        cat.write_text("providers:\n  local: {type: builtin}\n")
        with pytest.raises(RuntimeError, match="model.enabled"):
            build_registry(None, cat)


class TestModelsEndpoint:
    def test_list_models_returns_200(self, test_client):
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200

    def test_list_models_schema(self, test_client):
        resp = test_client.get("/v1/models")
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1

    def test_list_models_contains_repo_id(self, test_client):
        resp = test_client.get("/v1/models")
        model = resp.json()["data"][0]
        assert model["id"] == "google/gemma-4-E4B-it"
        assert model["object"] == "model"


class TestChatCompletions:
    def test_basic_completion(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

    def test_completion_schema(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        data = resp.json()
        assert "id" in data
        assert "choices" in data
        assert "usage" in data
        assert data["object"] == "chat.completion"

    def test_completion_choice_structure(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        choice = resp.json()["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)

    def test_usage_counts_present(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_with_system_message(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a pirate."},
                    {"role": "user", "content": "hello"},
                ]
            },
        )
        assert resp.status_code == 200

    def test_with_multi_turn(self, test_client):
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "what is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "and 3+3?"},
                ]
            },
        )
        assert resp.status_code == 200

    def test_max_tokens_passed_through(self, test_client, mock_llm):
        test_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        )
        mock_llm.model.generate.assert_called()

    def test_503_when_model_not_loaded(self):
        from src import server
        original = server._llm
        server._llm = None
        try:
            client = TestClient(server.app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 503
        finally:
            server._llm = original

    def test_id_is_unique_per_request(self, test_client):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        id1 = test_client.post("/v1/chat/completions", json=payload).json()["id"]
        id2 = test_client.post("/v1/chat/completions", json=payload).json()["id"]
        assert id1 != id2
