"""Tests for settings_store persistence and /api/settings endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import src.settings_store as ss


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_settings(tmp_path, monkeypatch):
    """Redirect SETTINGS_PATH to a temp file so tests don't touch real data."""
    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "settings.json")


# ── settings_store.load() ────────────────────────────────────────────────

class TestLoad:
    def test_returns_defaults_when_no_file(self):
        result = ss.load()
        assert result["cheapest_model_id"] is None
        assert isinstance(result["system_prompt"], str)
        assert len(result["system_prompt"]) > 0

    def test_loads_existing_file(self):
        ss.SETTINGS_PATH.write_text(
            json.dumps({"cheapest_model_id": "my-model"}), encoding="utf-8"
        )
        assert ss.load()["cheapest_model_id"] == "my-model"

    def test_known_key_from_file_is_present(self):
        ss.SETTINGS_PATH.write_text(
            json.dumps({"cheapest_model_id": "m"}), encoding="utf-8"
        )
        result = ss.load()
        assert result["cheapest_model_id"] == "m"

    def test_new_default_keys_appear_even_if_not_in_file(self):
        # Simulates adding a new setting key in code after a settings.json was written
        ss.SETTINGS_PATH.write_text(json.dumps({}), encoding="utf-8")
        result = ss.load()
        assert "cheapest_model_id" in result

    def test_returns_defaults_on_malformed_json(self):
        ss.SETTINGS_PATH.write_text("not valid json!!!", encoding="utf-8")
        result = ss.load()
        assert result["cheapest_model_id"] is None
        assert result["system_prompt"] == ss.DEFAULT_SYSTEM_PROMPT

    def test_returns_defaults_on_empty_file(self):
        ss.SETTINGS_PATH.write_text("", encoding="utf-8")
        result = ss.load()
        assert result["cheapest_model_id"] is None
        assert result["system_prompt"] == ss.DEFAULT_SYSTEM_PROMPT

    def test_cheapest_model_id_defaults_to_none(self):
        result = ss.load()
        assert result["cheapest_model_id"] is None


# ── settings_store.save() ────────────────────────────────────────────────

class TestSave:
    def test_creates_file_on_disk(self):
        ss.save({"cheapest_model_id": "gpt-4"})
        assert ss.SETTINGS_PATH.exists()

    def test_returns_merged_settings(self):
        result = ss.save({"cheapest_model_id": "haiku"})
        assert result["cheapest_model_id"] == "haiku"

    def test_persists_to_disk(self):
        ss.save({"cheapest_model_id": "sonnet"})
        on_disk = json.loads(ss.SETTINGS_PATH.read_text(encoding="utf-8"))
        assert on_disk["cheapest_model_id"] == "sonnet"

    def test_merges_with_existing_settings(self):
        ss.save({"cheapest_model_id": "first"})
        ss.save({"cheapest_model_id": "second"})
        assert ss.load()["cheapest_model_id"] == "second"

    def test_ignores_unknown_keys(self):
        result = ss.save({"cheapest_model_id": "m", "rogue_key": "value"})
        assert "rogue_key" not in result

    def test_save_none_clears_value(self):
        ss.save({"cheapest_model_id": "model-x"})
        ss.save({"cheapest_model_id": None})
        assert ss.load()["cheapest_model_id"] is None

    def test_save_without_cheapest_preserves_existing(self):
        ss.save({"cheapest_model_id": "preserved"})
        ss.save({})
        assert ss.load()["cheapest_model_id"] == "preserved"


# ── /api/settings endpoints ──────────────────────────────────────────────

class TestGetSettingsEndpoint:
    def test_returns_200(self, test_client):
        resp = test_client.get("/api/settings")
        assert resp.status_code == 200

    def test_includes_cheapest_model_id(self, test_client):
        resp = test_client.get("/api/settings")
        assert "cheapest_model_id" in resp.json()

    def test_includes_models_list(self, test_client):
        resp = test_client.get("/api/settings")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_have_required_fields(self, test_client):
        resp = test_client.get("/api/settings")
        model = resp.json()["models"][0]
        for field in ("id", "name", "provider", "is_cheapest"):
            assert field in model

    def test_default_cheapest_model_id_is_none_or_string(self, test_client):
        data = test_client.get("/api/settings").json()
        assert data["cheapest_model_id"] is None or isinstance(data["cheapest_model_id"], str)


class TestPatchSettingsEndpoint:
    def test_returns_200(self, test_client):
        resp = test_client.patch("/api/settings", json={"cheapest_model_id": None})
        assert resp.status_code == 200

    def test_updates_cheapest_model_id(self, test_client, mock_registry):
        model_id = mock_registry.list_models()[0].id
        resp = test_client.patch("/api/settings", json={"cheapest_model_id": model_id})
        assert resp.json()["cheapest_model_id"] == model_id

    def test_setting_none_clears_override(self, test_client):
        test_client.patch("/api/settings", json={"cheapest_model_id": "some-model"})
        resp = test_client.patch("/api/settings", json={"cheapest_model_id": None})
        assert resp.json()["cheapest_model_id"] is None

    def test_updates_llm_service_override(self, test_client):
        from src import server
        test_client.patch("/api/settings", json={"cheapest_model_id": "google/gemma-4-E4B-it"})
        assert server._llm_service._cheapest_override == "google/gemma-4-E4B-it"

    def test_persists_across_get(self, test_client, mock_registry):
        model_id = mock_registry.list_models()[0].id
        test_client.patch("/api/settings", json={"cheapest_model_id": model_id})
        get_resp = test_client.get("/api/settings")
        assert get_resp.json()["cheapest_model_id"] == model_id


class TestApiModelsEndpoint:
    def test_returns_200(self, test_client):
        resp = test_client.get("/api/models")
        assert resp.status_code == 200

    def test_includes_is_cheapest_field(self, test_client):
        resp = test_client.get("/api/models")
        model = resp.json()[0]
        assert "is_cheapest" in model

    def test_model_fields_present(self, test_client):
        resp = test_client.get("/api/models")
        model = resp.json()[0]
        for field in ("id", "name", "provider", "is_cheapest"):
            assert field in model
