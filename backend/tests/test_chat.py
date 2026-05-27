"""Tests for chat_store persistence."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import src.chat_store as cs


@pytest.fixture(autouse=True)
def tmp_chats_dir(tmp_path, monkeypatch):
    """Redirect CHATS_DIR to a temp dir so tests don't touch real data."""
    monkeypatch.setattr(cs, "CHATS_DIR", tmp_path / "chats")


class TestCreateChat:
    def test_returns_dict_with_id(self):
        chat = cs.create_chat()
        assert "id" in chat
        assert isinstance(chat["id"], str)

    def test_default_name(self):
        chat = cs.create_chat()
        assert chat["name"] == "New Chat"

    def test_starts_with_empty_messages(self):
        chat = cs.create_chat()
        assert chat["messages"] == []

    def test_creates_file_on_disk(self):
        chat = cs.create_chat()
        assert (cs.CHATS_DIR / f"{chat['id']}.json").exists()

    def test_unique_ids(self):
        ids = {cs.create_chat()["id"] for _ in range(5)}
        assert len(ids) == 5


class TestGetChat:
    def test_returns_existing_chat(self):
        chat = cs.create_chat()
        loaded = cs.get_chat(chat["id"])
        assert loaded is not None
        assert loaded["id"] == chat["id"]

    def test_returns_none_for_missing(self):
        assert cs.get_chat("nonexistent-id") is None


class TestListChats:
    def test_empty_when_no_chats(self):
        assert cs.list_chats() == []

    def test_lists_created_chats(self):
        cs.create_chat()
        cs.create_chat()
        assert len(cs.list_chats()) == 2

    def test_list_entries_have_required_fields(self):
        cs.create_chat()
        entry = cs.list_chats()[0]
        for key in ("id", "name", "created_at", "updated_at", "preview"):
            assert key in entry


class TestRenameChat:
    def test_rename_changes_name(self):
        chat = cs.create_chat()
        updated = cs.rename_chat(chat["id"], "My Chat")
        assert updated["name"] == "My Chat"

    def test_rename_persists(self):
        chat = cs.create_chat()
        cs.rename_chat(chat["id"], "Persistent Name")
        assert cs.get_chat(chat["id"])["name"] == "Persistent Name"

    def test_rename_nonexistent_returns_none(self):
        assert cs.rename_chat("bad-id", "name") is None

    def test_rename_empty_string_uses_fallback(self):
        chat = cs.create_chat()
        updated = cs.rename_chat(chat["id"], "")
        assert updated["name"] == "New Chat"


class TestAppendMessages:
    def test_appends_user_message(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "hello")
        messages = cs.get_chat(chat["id"])["messages"]
        assert messages[-1] == {"role": "user", "content": "hello"}

    def test_appends_assistant_message(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "assistant", "Hi there!")
        messages = cs.get_chat(chat["id"])["messages"]
        assert messages[-1]["role"] == "assistant"

    def test_name_stays_new_chat_after_first_user_message(self):
        # Auto-naming moved to LLMService.generate_title(); append_messages is name-neutral.
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "Tell me about Python")
        assert cs.get_chat(chat["id"])["name"] == "New Chat"

    def test_model_id_stored_on_user_message(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "hello", model_id="my-model")
        assert cs.get_chat(chat["id"])["model_id"] == "my-model"

    def test_model_id_not_stored_for_assistant_message(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "assistant", "hi", model_id="my-model")
        assert "model_id" not in cs.get_chat(chat["id"])

    def test_does_not_rename_if_already_named(self):
        chat = cs.create_chat()
        cs.rename_chat(chat["id"], "Custom Name")
        cs.append_messages(chat["id"], "user", "first message")
        assert cs.get_chat(chat["id"])["name"] == "Custom Name"

    def test_returns_none_for_missing_chat(self):
        assert cs.append_messages("bad-id", "user", "hi") is None

    def test_multiple_messages_accumulate(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "msg1")
        cs.append_messages(chat["id"], "assistant", "resp1")
        cs.append_messages(chat["id"], "user", "msg2")
        assert len(cs.get_chat(chat["id"])["messages"]) == 3


class TestDeleteChat:
    def test_delete_removes_file(self):
        chat = cs.create_chat()
        path = cs.CHATS_DIR / f"{chat['id']}.json"
        assert path.exists()
        cs.delete_chat(chat["id"])
        assert not path.exists()

    def test_delete_returns_true_on_success(self):
        chat = cs.create_chat()
        assert cs.delete_chat(chat["id"]) is True

    def test_delete_returns_false_for_missing(self):
        assert cs.delete_chat("nonexistent") is False

    def test_get_after_delete_returns_none(self):
        chat = cs.create_chat()
        cs.delete_chat(chat["id"])
        assert cs.get_chat(chat["id"]) is None


class TestPreview:
    def test_preview_shows_last_message_snippet(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "first")
        cs.append_messages(chat["id"], "assistant", "second response")
        entry = cs.list_chats()[0]
        assert "second" in entry["preview"]

    def test_preview_truncates_long_content(self):
        chat = cs.create_chat()
        cs.append_messages(chat["id"], "user", "x" * 100)
        entry = cs.list_chats()[0]
        assert len(entry["preview"]) <= 81
