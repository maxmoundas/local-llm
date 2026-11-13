"""fetch_page redirects and size cap, atomic busy-chat check, chat list from the index, session cookie,
and /v1/vision input errors."""
from __future__ import annotations

import json
import sqlite3
import threading

import httpx
import pytest

from src.agent import tools as tools_mod
from src.tasks.manager import ChatBusy, TaskManager


# ── fetch_page ──────────────────────────────────────────────────────────

@pytest.fixture
def fake_web(monkeypatch):
    """Route fetch_page through a MockTransport; hosts named *.public count as public."""
    requested: list[str] = []
    routes: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return routes.get(str(request.url), httpx.Response(404))

    real_client = httpx.Client
    monkeypatch.setattr(tools_mod.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(tools_mod, "_is_public_host", lambda host: host.endswith(".public"))
    return routes, requested


class TestFetchPage:
    def test_redirect_to_private_host_is_never_requested(self, fake_web):
        routes, requested = fake_web
        routes["http://a.public/"] = httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
        result = tools_mod._fetch_page({"url": "http://a.public/"})
        assert result.text.startswith("Error") and "169.254.169.254" in result.text
        assert requested == ["http://a.public/"]

    def test_follows_public_redirects(self, fake_web):
        routes, requested = fake_web
        routes["http://a.public/"] = httpx.Response(301, headers={"location": "/moved"})
        routes["http://a.public/moved"] = httpx.Response(
            200, headers={"content-type": "text/html"}, text="<title>Moved</title><article>Hi there</article>")
        result = tools_mod._fetch_page({"url": "http://a.public/"})
        assert "Hi there" in result.text and result.sources[0].url == "http://a.public/moved"
        assert requested == ["http://a.public/", "http://a.public/moved"]

    def test_redirect_loop_stops(self, fake_web):
        routes, requested = fake_web
        routes["http://a.public/"] = httpx.Response(302, headers={"location": "http://a.public/"})
        result = tools_mod._fetch_page({"url": "http://a.public/"})
        assert "too many redirects" in result.text
        assert len(requested) == tools_mod._MAX_REDIRECTS + 1

    def test_body_is_capped(self, fake_web, monkeypatch):
        routes, _ = fake_web
        monkeypatch.setattr(tools_mod, "_MAX_PAGE_BYTES", 1000)
        routes["http://big.public/"] = httpx.Response(200, headers={"content-type": "text/plain"},
                                                      content=b"x" * 50_000)
        result = tools_mod._fetch_page({"url": "http://big.public/"})
        assert result.ok and result.text.count("x") == 1000

    def test_empty_html_page(self, fake_web):
        routes, _ = fake_web
        routes["http://e.public/"] = httpx.Response(200, headers={"content-type": "text/html"}, content=b"")
        assert not tools_mod._fetch_page({"url": "http://e.public/"}).text.startswith("Error")

    def test_unsupported_type_skips_body(self, fake_web):
        routes, _ = fake_web
        routes["http://f.public/x.zip"] = httpx.Response(200, headers={"content-type": "application/zip"},
                                                         content=b"PK")
        assert tools_mod._fetch_page({"url": "http://f.public/x.zip"}).summary == "unsupported"


# ── One running task per chat ───────────────────────────────────────────

class TestExclusiveStart:
    def test_second_exclusive_start_is_refused(self):
        mgr, gate = TaskManager(), threading.Event()
        mgr.start("c", "m", lambda ctx: gate.wait(3), exclusive=True)
        try:
            with pytest.raises(ChatBusy):
                mgr.start("c", "m", lambda ctx: None, exclusive=True)
            mgr.start("other", "m", lambda ctx: None, exclusive=True)  # other chats are unaffected
        finally:
            gate.set()

    def test_concurrent_starts_admit_one(self):
        mgr, gate, barrier = TaskManager(), threading.Event(), threading.Barrier(8)
        started, refused = [], []

        def attempt():
            barrier.wait()
            try:
                started.append(mgr.start("c", "m", lambda ctx: gate.wait(3), exclusive=True))
            except ChatBusy:
                refused.append(1)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        gate.set()
        assert len(started) == 1 and len(refused) == 7


# ── Chat list from the index ────────────────────────────────────────────

class TestChatListIndex:
    def test_list_comes_from_index_without_reading_files(self, monkeypatch):
        from src import chat_store
        a = chat_store.create_chat(name="first")
        chat_store.append_messages(a["id"], "user", "hello there")
        b = chat_store.create_chat(name="second")
        monkeypatch.setattr(chat_store, "_all", lambda: pytest.fail("list_chats parsed every chat file"))
        chats = chat_store.list_chats()
        assert [c["id"] for c in chats][:2] == [b["id"], a["id"]] or {c["id"] for c in chats} == {a["id"], b["id"]}
        assert next(c for c in chats if c["id"] == a["id"])["preview"] == "hello there"

    def test_project_filter_and_pin(self):
        from src import chat_store
        a = chat_store.create_chat(project_id="p1")
        chat_store.create_chat()
        chat_store.toggle_pin(a["id"])
        listed = chat_store.list_chats("p1")
        assert [c["id"] for c in listed] == [a["id"]] and listed[0]["pinned"] is True

    def test_files_written_outside_the_store_show_up(self):
        from src import chat_store
        chat_store.create_chat()  # creates the dir and index
        cid = "11111111-2222-3333-4444-555555555555"
        (chat_store.CHATS_DIR / f"{cid}.json").write_text(json.dumps({
            "id": cid, "name": "Restored", "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00", "messages": []}))
        assert "Restored" in [c["name"] for c in chat_store.list_chats()]

    def test_deleted_file_disappears(self):
        from src import chat_store
        cid = chat_store.create_chat()["id"]
        (chat_store.CHATS_DIR / f"{cid}.json").unlink()
        assert cid not in [c["id"] for c in chat_store.list_chats()]

    def test_old_index_without_summary_column_is_migrated(self):
        from src import chat_store, search_index
        cid = chat_store.create_chat(name="Old")["id"]
        search_index._conn.close()
        search_index._conn = None
        db = chat_store.CHATS_DIR / "search.db"
        db.unlink()
        conn = sqlite3.connect(db)
        conn.executescript("CREATE TABLE chats (chat_id TEXT PRIMARY KEY, updated_at TEXT NOT NULL, mtime REAL NOT NULL);")
        mtime = (chat_store.CHATS_DIR / f"{cid}.json").stat().st_mtime
        conn.execute("INSERT INTO chats VALUES (?,?,?)", (cid, "", mtime))
        conn.commit()
        conn.close()
        assert [c["name"] for c in chat_store.list_chats()] == ["Old"]

    def test_malformed_chat_does_not_break_list(self):
        from src import chat_store
        good = chat_store.create_chat(name="Good")["id"]
        bad = "99999999-2222-3333-4444-555555555555"
        (chat_store.CHATS_DIR / f"{bad}.json").write_text(json.dumps({"id": bad, "messages": []}))
        assert [c["id"] for c in chat_store.list_chats()] == [good]


# ── Server ──────────────────────────────────────────────────────────────

class TestServerFixes:
    def test_cookie_does_not_hold_the_token(self, test_client, monkeypatch):
        from src import server
        monkeypatch.setattr(server, "_ACCESS_TOKEN", "s3cret")
        login = test_client.post("/api/auth/login", json={"token": "s3cret"})
        assert login.cookies["llm_session"] != "s3cret"
        assert test_client.get("/api/chats").status_code == 200
        # An old-style cookie (the raw token) no longer works. Clear first: newer Starlette/httpx scope the
        # login cookie to "testserver.local", so set() would add a second cookie instead of replacing it.
        test_client.cookies.clear()
        test_client.cookies.set("llm_session", "s3cret")
        assert test_client.get("/api/chats").status_code == 401

    def test_vision_rejects_bad_image(self, test_client, monkeypatch):
        from unittest.mock import MagicMock
        from src import server
        monkeypatch.setattr(server, "_llm", MagicMock(is_loaded=True))
        assert test_client.post("/v1/vision", json={"image_base64": "not base64!", "prompt": "x"}).status_code == 400
        assert test_client.post("/v1/vision", json={"image_base64": "aGVsbG8=", "prompt": "x"}).status_code == 400
