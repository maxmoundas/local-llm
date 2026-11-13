"""Versions (regenerate/edit), attachments, context fitting, search/export, projects, usage and budget,
schedules, per-chat options, voice, and the network-security middleware."""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime

import pytest

from src import attachments, chat_store, context, project_store, scheduler
from src.providers.base import ModelOption

PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82").decode()


@pytest.fixture(autouse=True)
def plain_turns(monkeypatch):
    """Answer directly (no agent, no memory) and record what the local model was sent."""
    from src import server
    from src.model import LocalLLM
    seen: list = []

    def stream(self, messages, *a, **kw):
        seen.append({"messages": messages, **kw})
        return iter(["hello ", "world"])

    monkeypatch.setattr(LocalLLM, "stream", stream)
    for k in ("_router", "_agent", "_guardrail_svc"):
        monkeypatch.setattr(server, k, None)
    monkeypatch.setattr(server, "_memory_enabled", False)
    monkeypatch.setattr(server, "_monthly_budget", None)
    return seen


def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.split("\n") if line.startswith("data: ")]


def _send(client, chat_id, content="hi", **body):
    resp = client.post(f"/api/chats/{chat_id}/message", json={"content": content, **body})
    assert resp.status_code == 200, resp.text
    return _events(resp)


# ── Versions ────────────────────────────────────────────────────────────

class TestVersionsStore:
    def _chat(self):
        chat = chat_store.create_chat()
        for role, text in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]:
            chat_store.append_messages(chat["id"], role, text)
        return chat["id"]

    def test_fork_then_append_keeps_old_tail_as_version(self):
        cid = self._chat()
        chat_store.fork(cid, 1)
        chat = chat_store.append_messages(cid, "assistant", "a1-new")
        msgs = chat["messages"]
        assert [m["content"] for m in msgs] == ["q1", "a1-new"]
        assert msgs[1]["version_index"] == 1
        assert [m["content"] for m in msgs[1]["versions"][0]] == ["a1", "q2", "a2"]

    def test_switch_version_restores_the_whole_tail(self):
        cid = self._chat()
        chat_store.fork(cid, 1)
        chat_store.append_messages(cid, "assistant", "a1-new")
        chat = chat_store.switch_version(cid, 1, 0)
        assert [m["content"] for m in chat["messages"]] == ["q1", "a1", "q2", "a2"]
        assert chat["messages"][1]["version_index"] == 0
        assert [m["content"] for m in chat["messages"][1]["versions"][0]] == ["a1-new"]
        back = chat_store.switch_version(cid, 1, 1)
        assert [m["content"] for m in back["messages"]] == ["q1", "a1-new"]

    def test_restore_fork_when_nothing_was_written(self):
        cid = self._chat()
        chat_store.fork(cid, 3)
        assert chat_store.restore_fork(cid)
        msgs = chat_store.get_chat(cid)["messages"]
        assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2"]
        assert "versions" not in msgs[3] and "pending_fork" not in chat_store.get_chat(cid)

    def test_three_versions_keep_order(self):
        cid = self._chat()
        for text in ("b", "c"):
            chat_store.fork(cid, 3)
            chat_store.append_messages(cid, "assistant", text)
        msg = chat_store.get_chat(cid)["messages"][3]
        assert msg["content"] == "c" and msg["version_index"] == 2
        assert [v[0]["content"] for v in msg["versions"]] == ["a2", "b"]


class TestRegenerateAndEdit:
    def test_regenerate_adds_a_version(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        _send(test_client, cid)
        resp = test_client.post(f"/api/chats/{cid}/regenerate", json={"index": 1})
        assert resp.status_code == 200
        msgs = chat_store.get_chat(cid)["messages"]
        assert len(msgs) == 2 and msgs[1]["version_index"] == 1 and len(msgs[1]["versions"]) == 1

    def test_regenerate_rejects_user_index(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        _send(test_client, cid)
        assert test_client.post(f"/api/chats/{cid}/regenerate", json={"index": 0}).status_code == 400

    def test_failed_regenerate_restores_previous_reply(self, test_client, monkeypatch):
        from src.model import LocalLLM
        cid = test_client.post("/api/chats").json()["id"]
        _send(test_client, cid)

        def boom(self, *a, **kw):
            raise RuntimeError("model crashed")
            yield  # pragma: no cover
        monkeypatch.setattr(LocalLLM, "stream", boom)
        events = _events(test_client.post(f"/api/chats/{cid}/regenerate", json={"index": 1}))
        assert any(e["type"] == "error" for e in events)
        msgs = chat_store.get_chat(cid)["messages"]
        assert [m["content"] for m in msgs] == ["hi", "hello world"]

    def test_edit_replaces_user_message_and_keeps_old_branch(self, test_client, plain_turns):
        cid = test_client.post("/api/chats").json()["id"]
        _send(test_client, cid, "first")
        resp = test_client.post(f"/api/chats/{cid}/edit", json={"index": 0, "content": "second"})
        assert resp.status_code == 200
        msgs = chat_store.get_chat(cid)["messages"]
        assert [m["content"] for m in msgs] == ["second", "hello world"]
        assert msgs[0]["versions"][0][0]["content"] == "first"
        assert plain_turns[-1]["messages"][-1]["content"] == "second"

    def test_switch_endpoint(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        _send(test_client, cid, "first")
        test_client.post(f"/api/chats/{cid}/edit", json={"index": 0, "content": "second"})
        chat = test_client.post(f"/api/chats/{cid}/versions", json={"index": 0, "version": 0}).json()
        assert chat["messages"][0]["content"] == "first"
        assert test_client.post(f"/api/chats/{cid}/versions", json={"index": 0, "version": 5}).status_code == 400


# ── Attachments ─────────────────────────────────────────────────────────

class TestAttachments:
    def _upload(self, client, cid, name, data: bytes, mime=None):
        resp = client.post(f"/api/chats/{cid}/files",
                           json={"name": name, "data_base64": base64.b64encode(data).decode(), "mime": mime})
        assert resp.status_code == 201, resp.text
        return resp.json()

    def test_document_text_reaches_the_model(self, test_client, plain_turns):
        cid = test_client.post("/api/chats").json()["id"]
        meta = self._upload(test_client, cid, "notes.txt", b"The launch code is PURPLE.")
        assert meta["kind"] == "document"
        _send(test_client, cid, "what is the code?", attachments=[meta["id"]])
        last_user = plain_turns[-1]["messages"][-1]["content"]
        assert '<document name="notes.txt">' in last_user and "PURPLE" in last_user
        saved = chat_store.get_chat(cid)["messages"][0]
        assert saved["content"] == "what is the code?" and saved["attachments"][0]["name"] == "notes.txt"

    def test_image_is_saved_and_served(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        meta = self._upload(test_client, cid, "dot.png", base64.b64decode(PNG), "image/png")
        assert meta["kind"] == "image"
        resp = test_client.get(f"/api/chats/{cid}/files/{meta['id']}")
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"

    def test_docx_and_pdf_extraction(self, tmp_path):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        d.add_paragraph("Quarterly revenue rose")
        buf = io.BytesIO()
        d.save(buf)
        meta = attachments.save(tmp_path, "report.docx", buf.getvalue())
        assert "Quarterly revenue rose" in attachments.text_of(tmp_path, meta)

    def test_unsupported_type_rejected(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        resp = test_client.post(f"/api/chats/{cid}/files", json={"name": "a.exe", "data_base64": "AAAA"})
        assert resp.status_code == 400

    def test_unknown_attachment_id_rejected(self, test_client):
        cid = test_client.post("/api/chats").json()["id"]
        resp = test_client.post(f"/api/chats/{cid}/message", json={"content": "x", "attachments": ["f" * 32]})
        assert resp.status_code == 400

    def test_only_recent_images_are_sent(self, tmp_path):
        chat = chat_store.create_chat()
        root = chat_store.files_dir(chat["id"])
        for i in range(5):
            meta = attachments.save_base64(root, f"{i}.png", PNG, "image/png")
            chat_store.append_messages(chat["id"], "user", f"pic {i}", extra={"attachments": [meta]})
            chat_store.append_messages(chat["id"], "assistant", "ok")
        model = ModelOption(id="m", name="m", provider="x", vision=True)
        msgs = context.build(chat_store.get_chat(chat["id"]), "sys", model, files_root=root)
        with_images = [m for m in msgs if m.get("images")]
        assert len(with_images) == context.RECENT_IMAGE_MESSAGES
        assert "no longer shown" in msgs[1]["content"]
        text_only = ModelOption(id="t", name="t", provider="x", vision=False)
        msgs = context.build(chat_store.get_chat(chat["id"]), "sys", text_only, files_root=root)
        assert not any(m.get("images") for m in msgs)
        assert "can't view images" in msgs[-2]["content"]


# ── Context window ──────────────────────────────────────────────────────

class TestContextFitting:
    def _long_chat(self, n=40):
        chat = chat_store.create_chat()
        for i in range(n):
            chat_store.append_messages(chat["id"], "user", f"question {i} " + "x" * 400)
            chat_store.append_messages(chat["id"], "assistant", f"answer {i} " + "y" * 400)
        return chat_store.get_chat(chat["id"])

    def test_short_chat_untouched(self):
        chat = self._long_chat(2)
        msgs = context.build(chat, "sys", ModelOption(id="m", name="m", provider="x"))
        assert len(msgs) == 5 and msgs[0]["content"] == "sys"

    def test_long_chat_is_trimmed_and_summarized(self):
        chat = self._long_chat()
        small = ModelOption(id="m", name="m", provider="x", context_window=6000, max_output_tokens=1000)
        calls, saved = [], []

        def summarize(dropped, previous):
            calls.append((len(dropped), previous))
            return "SUMMARY"

        msgs = context.build(chat, "sys", small, summarize=summarize, save_summary=saved.append)
        assert "SUMMARY" in msgs[0]["content"] and "<earlier_conversation_summary>" in msgs[0]["content"]
        assert msgs[1]["role"] == "user" and msgs[-1]["content"].startswith("answer 39")
        total = sum(context.estimate_tokens(m["content"]) for m in msgs)
        assert total <= context.budget_for(small)
        assert calls and saved and saved[0]["upto"] == calls[0][0]

        # The cached summary is reused, then extended — not rewritten — as the chat grows.
        chat["context_summary"] = saved[0]
        context.build(chat, "sys", small, summarize=summarize, save_summary=saved.append)
        assert len(calls) == 1
        chat["messages"] += [{"role": "user", "content": "more " + "z" * 2000},
                             {"role": "assistant", "content": "ok " + "z" * 2000}]
        context.build(chat, "sys", small, summarize=summarize, save_summary=saved.append)
        assert len(calls) == 2 and calls[1][1] == "SUMMARY"

    def test_summary_failure_drops_with_note(self):
        chat = self._long_chat()
        small = ModelOption(id="m", name="m", provider="x", context_window=6000, max_output_tokens=1000)

        def fail(*_):
            raise RuntimeError("down")
        msgs = context.build(chat, "sys", small, summarize=fail)
        assert "omitted" in msgs[0]["content"]

    def test_huge_last_message_is_trimmed(self):
        chat = chat_store.create_chat()
        chat_store.append_messages(chat["id"], "user", "start " + "q" * 100_000 + " end")
        small = ModelOption(id="m", name="m", provider="x", context_window=6000, max_output_tokens=1000)
        msgs = context.build(chat_store.get_chat(chat["id"]), "sys", small)
        assert "omitted" in msgs[-1]["content"] and msgs[-1]["content"].endswith("end")


# ── Search, export, title ───────────────────────────────────────────────

class TestSearchExport:
    def test_search_finds_message_content(self, test_client):
        a = chat_store.create_chat()
        chat_store.append_messages(a["id"], "user", "Let's talk about the Riemann hypothesis today")
        b = chat_store.create_chat()
        chat_store.append_messages(b["id"], "user", "unrelated")
        results = test_client.get("/api/search", params={"q": "riemann"}).json()
        assert [r["id"] for r in results] == [a["id"]]
        assert "Riemann" in results[0]["snippet"] and results[0]["index"] == 0

    def test_export_markdown_and_json(self, test_client):
        chat = chat_store.create_chat()
        chat_store.rename_chat(chat["id"], "My chat")
        chat_store.append_messages(chat["id"], "user", "hello")
        chat_store.append_messages(chat["id"], "assistant", "hi there",
                                   extra={"sources": [{"title": "A", "url": "https://a.com"}]})
        md = test_client.get(f"/api/chats/{chat['id']}/export?format=md")
        assert md.status_code == 200 and "attachment" in md.headers["content-disposition"]
        assert "# My chat" in md.text and "## Assistant" in md.text and "[A](https://a.com)" in md.text
        js = test_client.get(f"/api/chats/{chat['id']}/export?format=json").json()
        assert js["messages"][1]["content"] == "hi there"

    def test_generate_title_from_history(self, test_client, monkeypatch):
        from src import server
        chat = chat_store.create_chat()
        chat_store.append_messages(chat["id"], "user", "how do I bake bread")
        monkeypatch.setattr(server._llm_service, "complete", lambda *a, **kw: "Baking Bread Basics")
        assert test_client.post(f"/api/chats/{chat['id']}/title").json()["name"] == "Baking Bread Basics"


# ── Per-chat settings ───────────────────────────────────────────────────

class TestChatSettings:
    def test_system_prompt_override_and_temperature(self, test_client, plain_turns):
        cid = test_client.post("/api/chats").json()["id"]
        resp = test_client.patch(f"/api/chats/{cid}", json={"settings": {"system_prompt": "Talk like a pirate",
                                                                          "temperature": 0.2}})
        assert resp.json()["settings"] == {"system_prompt": "Talk like a pirate", "temperature": 0.2}
        _send(test_client, cid)
        assert plain_turns[-1]["messages"][0]["content"].startswith("Talk like a pirate")
        assert plain_turns[-1]["temperature"] == 0.2

    def test_anthropic_effort_and_no_temperature(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="x")
        kw = p._request([{"role": "user", "content": "hi"}], "claude-sonnet-5-5", None,
                        {"effort": "low", "temperature": 0.5})
        assert kw["extra_body"]["output_config"]["effort"] == "low"
        assert "temperature" not in kw

    def test_anthropic_images_from_history(self):
        from src.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="x")
        kw = p._request([{"role": "user", "content": "look", "images": [PNG]}], "claude-haiku-5-5", None)
        blocks = kw["messages"][0]["content"]
        assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"

    def test_openai_reasoning_effort(self):
        from src.providers.openai_provider import OpenAIProvider
        p = OpenAIProvider(api_key="x")
        assert p._options("gpt-5.4-mini", {"effort": "high", "temperature": 1.0}) == {"reasoning_effort": "high"}


# ── Projects ────────────────────────────────────────────────────────────

class TestProjects:
    def test_project_context_reaches_model(self, test_client, plain_turns, monkeypatch):
        from src import server
        monkeypatch.setattr(server, "_memory_enabled", True)
        monkeypatch.setattr(server, "_maybe_update_memory", lambda *a, **kw: None)
        proj = test_client.post("/api/projects", json={"name": "Garden", "instructions": "Answer in haiku"}).json()
        up = test_client.post(f"/api/projects/{proj['id']}/files",
                              json={"name": "plants.md", "data_base64": base64.b64encode(b"Tomatoes on the left").decode()})
        assert up.status_code == 201
        test_client.patch(f"/api/projects/{proj['id']}/memory", json={"content": "- Lives in zone 7"})
        cid = test_client.post("/api/chats", json={"project_id": proj["id"]}).json()["id"]
        _send(test_client, cid)
        system = plain_turns[-1]["messages"][0]["content"]
        assert 'project name="Garden"' in system and "Answer in haiku" in system
        assert "Tomatoes on the left" in system and "zone 7" in system
        detail = test_client.get(f"/api/projects/{proj['id']}").json()
        assert detail["chat_count"] == 1 and detail["chats"][0]["id"] == cid

    def test_project_chat_memory_goes_to_project(self, test_client, monkeypatch):
        from src import memory_store, server
        monkeypatch.setattr(server, "_memory_enabled", True)
        monkeypatch.setattr(server, "_jev_memory", False)
        monkeypatch.setattr(server._llm_service, "extract_memory", lambda *a, **kw: "- Uses raised beds")
        proj = project_store.create("P")
        chat = chat_store.create_chat(project_id=proj["id"])
        server._turn_deps(server._llm_service, chat).update_memory("I use raised beds", "")
        assert "raised beds" in project_store.memory_path(proj["id"]).read_text()
        assert memory_store.load() == ""  # the global profile is untouched

    def test_delete_project_keeps_chats(self, test_client):
        proj = project_store.create("P")
        chat = chat_store.create_chat(project_id=proj["id"])
        assert test_client.delete(f"/api/projects/{proj['id']}").status_code == 204
        assert "project_id" not in chat_store.get_chat(chat["id"])

    def test_move_chat_into_project(self, test_client):
        proj = project_store.create("P")
        cid = test_client.post("/api/chats").json()["id"]
        assert test_client.patch(f"/api/chats/{cid}", json={"project_id": proj["id"]}).json()["project_id"] == proj["id"]
        assert "project_id" not in test_client.patch(f"/api/chats/{cid}", json={"project_id": ""}).json()


# ── Usage and budget ────────────────────────────────────────────────────

class TestUsage:
    def test_usage_report_and_budget(self, test_client, monkeypatch):
        from src import server
        from src.tasks import store
        store.create("t1", None, "claude-sonnet-5-5")
        store.set_status("t1", "done", usage={"cost_usd": 1.5, "input_tokens": 100, "output_tokens": 50, "calls": 1,
                                              "by_model": {"claude-sonnet-5-5": {"cost_usd": 1.5, "input_tokens": 100,
                                                                                 "output_tokens": 50, "calls": 1}}})
        report = test_client.get("/api/usage?days=7").json()
        assert report["total"]["cost_usd"] == 1.5 and report["month_to_date_usd"] == 1.5
        assert report["models"][0]["model_id"] == "claude-sonnet-5-5"
        assert len(report["days"]) == 7 and report["days"][-1]["cost_usd"] == 1.5

        monkeypatch.setattr(server, "_monthly_budget", 1.0)
        cid = test_client.post("/api/chats").json()["id"]
        resp = test_client.post(f"/api/chats/{cid}/message", json={"content": "x"})
        assert resp.status_code == 402 and "limit" in resp.json()["detail"]


# ── Schedules ───────────────────────────────────────────────────────────

class TestSchedules:
    def test_next_run_kinds(self):
        now = datetime(2026, 11, 13, 9, 30)  # a Friday
        assert scheduler.next_run({"kind": "daily", "time": "08:00"}, now) == datetime(2026, 11, 14, 8, 0)
        assert scheduler.next_run({"kind": "daily", "time": "10:00"}, now) == datetime(2026, 11, 13, 10, 0)
        assert scheduler.next_run({"kind": "weekly", "time": "08:00", "days": [0]}, now) == datetime(2026, 11, 16, 8, 0)
        assert scheduler.next_run({"kind": "interval", "every_minutes": 90}, now) == datetime(2026, 11, 13, 11, 0)

    def test_validation(self):
        with pytest.raises(ValueError):
            scheduler.create({"kind": "daily", "time": "25:00", "prompt": "x"})
        with pytest.raises(ValueError):
            scheduler.create({"kind": "interval", "every_minutes": 1, "prompt": "x"})
        with pytest.raises(ValueError):
            scheduler.create({"kind": "weekly", "time": "08:00", "prompt": "x"})

    def test_tick_runs_due_schedule_into_its_chat(self, test_client):
        from src import server
        s = test_client.post("/api/schedules", json={"name": "News", "prompt": "Summarize the news",
                                                      "kind": "daily", "time": "07:00"}).json()
        assert s["description"] == "Daily at 07:00"
        sch = scheduler.Scheduler(server._start_scheduled_run)
        started = sch.tick(datetime.fromisoformat(s["next_run_at"]))
        assert started == [s["id"]]
        after = scheduler.get(s["id"])
        assert after["chat_id"] and after["next_run_at"] > s["next_run_at"]
        for _ in server._tasks.subscribe(after["last_task_id"]):
            pass
        chat = chat_store.get_chat(after["chat_id"])
        assert chat["name"] == "News" and chat["schedule_id"] == s["id"]
        assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]

    def test_unattended_runs_cannot_ask(self, test_client):
        from src import server
        chat = chat_store.create_chat()
        assert server._turn_deps(server._llm_service, chat, unattended=True).unattended

    def test_crud(self, test_client):
        s = test_client.post("/api/schedules", json={"prompt": "x", "kind": "interval", "every_minutes": 60}).json()
        upd = test_client.patch(f"/api/schedules/{s['id']}", json={"enabled": False}).json()
        assert upd["enabled"] is False
        assert test_client.patch(f"/api/schedules/{s['id']}", json={"every_minutes": 2}).status_code == 400
        assert test_client.delete(f"/api/schedules/{s['id']}").status_code == 204
        assert test_client.get("/api/schedules").json() == []


# ── Voice ───────────────────────────────────────────────────────────────

class TestTranscribe:
    def test_unavailable(self, test_client, monkeypatch):
        from src import transcribe
        monkeypatch.setattr(transcribe, "available", lambda: False)
        assert test_client.post("/api/transcribe", json={"audio_base64": "AAAA"}).status_code == 503

    def test_success(self, test_client, monkeypatch):
        from src import transcribe
        monkeypatch.setattr(transcribe, "available", lambda: True)
        monkeypatch.setattr(transcribe, "transcribe", lambda audio, suffix, model: f"heard {len(audio)} bytes{suffix}")
        resp = test_client.post("/api/transcribe", json={"audio_base64": "AAAA", "mime": "audio/webm;codecs=opus"})
        assert resp.json() == {"text": "heard 3 bytes.webm"}


# ── Security ────────────────────────────────────────────────────────────

class TestSecurity:
    def test_foreign_host_rejected_without_token(self, test_client):
        resp = test_client.get("/api/chats", headers={"host": "evil.example.com"})
        assert resp.status_code == 403

    def test_cross_origin_post_blocked(self, test_client):
        resp = test_client.post("/api/chats", headers={"origin": "https://evil.example.com"})
        assert resp.status_code == 403
        ok = test_client.post("/api/chats", headers={"origin": "http://localhost:5173"})
        assert ok.status_code == 201

    def test_access_token(self, test_client, monkeypatch):
        from src import server
        monkeypatch.setattr(server, "_ACCESS_TOKEN", "s3cret")
        assert test_client.get("/api/chats", headers={"host": "my-box.lan"}).status_code == 401
        assert test_client.get("/api/auth/status").json() == {"required": True, "authenticated": False}
        assert test_client.get("/api/chats", headers={"authorization": "Bearer s3cret"}).status_code == 200
        assert test_client.post("/api/auth/login", json={"token": "nope"}).status_code == 401
        login = test_client.post("/api/auth/login", json={"token": "s3cret"})
        assert login.status_code == 200 and "llm_session" in login.cookies
        assert test_client.get("/api/chats").status_code == 200  # cookie kept by the client
        assert test_client.get("/health").status_code == 200


class TestAgentSeesAttachments:
    def test_current_message_files_are_staged_and_text_is_given_to_planner(self, test_client, monkeypatch):
        from unittest.mock import MagicMock
        from src import server
        from src.agent import sandbox as sb
        from src.agent.router import ResearchRouter, RouteDecision
        from src.agent.runner import AgentEvent

        seen = {}
        router = MagicMock(spec=ResearchRouter)
        router.decide.side_effect = lambda text, history: (seen.setdefault("route", text), RouteDecision(True, 0.9))[1]

        def run(history, content, model_id, **kw):
            seen["content"] = content
            seen["files"] = sorted(p.name for p in sb.WORKSPACES_DIR.rglob("*") if p.is_file())
            yield AgentEvent("done", notes="", sources=[], actions=[])

        agent = MagicMock()
        agent.run.side_effect = run
        sandbox = MagicMock()
        monkeypatch.setattr(server, "_router", router)
        monkeypatch.setattr(server, "_agent", agent)
        monkeypatch.setattr(server, "_sandbox", sandbox)

        cid = test_client.post("/api/chats").json()["id"]
        meta = test_client.post(f"/api/chats/{cid}/files", json={
            "name": "data.csv", "data_base64": base64.b64encode(b"city,temp\nOslo,3\n").decode()}).json()
        _send(test_client, cid, "average temp?", attachments=[meta["id"]])
        assert "data.csv" in seen["route"]
        assert '<document name="data.csv">' in seen["content"] and "Oslo,3" in seen["content"]
        assert "data.csv" in seen["content"].split("</document>")[-1]  # the workspace note
        assert seen["files"] == ["data.csv"]
        # The staged copy isn't reported back as a file the agent created.
        assert "files" not in chat_store.get_chat(cid)["messages"][-1]
