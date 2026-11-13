"""Usage outside turns, workspace/container cleanup, sandbox languages, the untrusted-content guard,
cache-write pricing, multi-model /v1, the search index, backup/restore, and serving the built frontend."""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import usage
from src.agent.router import ResearchRouter
from src.agent.runner import RunControls, TaskAgent, UNTRUSTED_REASON
from src.agent.tools import Tool, ToolResult
from src.llm_service import LLMService
from src.providers.base import ToolCall, ToolTurn
from tests.fakes import ScriptedProvider


def _sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]


# ── Usage outside chat turns ────────────────────────────────────────────

class TestUsageOutsideTurns:
    def test_record_usage_counts_in_dashboard_but_not_as_turn(self, test_client):
        from src.tasks import store
        store.record_usage(None, "m", {"cost_usd": 0.25, "input_tokens": 10, "output_tokens": 5,
                                       "by_model": {"m": {"cost_usd": 0.25, "input_tokens": 10,
                                                          "output_tokens": 5, "calls": 1}}}, "memory")
        data = test_client.get("/api/usage").json()
        assert data["month_to_date_usd"] == pytest.approx(0.25)
        assert data["total"]["cost_usd"] == pytest.approx(0.25)
        assert data["total"]["turns"] == 0
        assert data["models"][0]["model_id"] == "m"

    def test_old_database_gets_kind_column(self, tmp_path, monkeypatch):
        import sqlite3
        from src.tasks import store
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, chat_id TEXT, status TEXT NOT NULL, model_id TEXT, "
                     "created_at REAL NOT NULL, updated_at REAL NOT NULL, usage TEXT, error TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('t1', NULL, 'done', 'm', 1, 1, '{}', NULL)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(store, "DB_PATH", db)
        rows = store.usage_since(0)
        assert rows[0]["kind"] == "turn"

    def test_memory_job_meters_its_own_calls(self, monkeypatch):
        from src.tasks import store, turn
        recorded = []
        monkeypatch.setattr(store, "record_usage", lambda *a: recorded.append(a))

        def update(content, prev):
            usage.record("m", 100, 10)
            return {"added": ["x"], "removed": []}

        with usage.meter() as outer:  # stands in for the turn's meter
            job = turn._MemoryJob(update, "hi", "", "chat-1")
            assert job.wait(5) == {"added": ["x"], "removed": []}
        job._thread.join(5)
        assert outer.calls == 0  # not double-counted on the turn
        assert recorded and recorded[0][0] == "chat-1" and recorded[0][3] == "memory"
        assert recorded[0][2]["input_tokens"] == 100

    def test_title_generation_is_recorded(self, test_client, monkeypatch):
        from src import server
        from src.tasks import store
        chat_id = test_client.post("/api/chats").json()["id"]

        def gen(messages):
            usage.record("m", 50, 5)
            return "A title"

        monkeypatch.setattr(server._llm_service, "generate_title_from_history", gen)
        assert test_client.post(f"/api/chats/{chat_id}/title").json()["name"] == "A title"
        rows = store.usage_since(0)
        assert [(r["kind"], r["chat_id"]) for r in rows] == [("title", chat_id)]


class TestCachePricing:
    def test_cache_writes_cost_125_percent(self):
        usage.set_price("price-test", 1.0, 0.0)
        plain = usage.cost("price-test", 1_000_000, 0)
        written = usage.cost("price-test", 1_000_000, 0, cache_write_tokens=1_000_000)
        read = usage.cost("price-test", 1_000_000, 0, cached_input_tokens=1_000_000)
        assert plain == pytest.approx(1.0)
        assert written == pytest.approx(1.25)
        assert read == pytest.approx(0.1)


# ── Workspaces and containers ───────────────────────────────────────────

class TestWorkspaceCleanup:
    def test_deleting_chat_removes_its_workspaces(self, test_client):
        from src.agent import sandbox as sb
        from src.tasks import store
        chat_id = test_client.post("/api/chats").json()["id"]
        store.create("task-a", chat_id, "m")
        ws = sb.workspace_for("task-a")
        (ws / "out.csv").write_text("x")
        assert test_client.delete(f"/api/chats/{chat_id}").status_code == 204
        assert not ws.exists()

    def test_sweep_removes_only_old_workspaces(self):
        from src.agent import sandbox as sb
        old, new, active = sb.workspace_for("old"), sb.workspace_for("new"), sb.workspace_for("active")
        for d in (old, new, active):
            (d / "f.txt").write_text("x")
        past = time.time() - 40 * 86400
        for d in (old, active):
            os.utime(d / "f.txt", (past, past))
            os.utime(d, (past, past))
        assert sb.sweep_workspaces(30, keep={"active"}) == 1
        assert not old.exists() and new.exists() and active.exists()

    def test_sweep_disabled_with_zero(self):
        from src.agent import sandbox as sb
        sb.workspace_for("x")
        assert sb.sweep_workspaces(0) == 0

    def test_remove_workspace_refuses_paths(self, tmp_path):
        from src.agent import sandbox as sb
        outside = sb.WORKSPACES_DIR.parent / "keep"
        outside.mkdir(parents=True)
        sb.remove_workspace("../keep")
        assert outside.exists()


class TestDockerHousekeeping:
    def test_remove_stale_containers(self):
        from src.agent import sandbox as sb
        ps = MagicMock(stdout="abc\ndef\n", returncode=0)
        with patch.object(sb.subprocess, "run", side_effect=[ps, MagicMock(returncode=0)]) as run:
            assert sb.DockerSandbox.remove_stale() == 2
        assert run.call_args_list[1][0][0] == ["docker", "rm", "-f", "abc", "def"]

    def test_javascript_only_when_image_has_node(self):
        from src.agent import sandbox as sb
        box = sb.DockerSandbox()
        box._image_ready = True
        with patch.object(sb.subprocess, "run", return_value=MagicMock(returncode=1)):
            assert box.languages == ["python", "bash"]
        box2 = sb.DockerSandbox()
        box2._image_ready = True
        with patch.object(sb.subprocess, "run", return_value=MagicMock(returncode=0)):
            assert box2.languages == ["python", "bash", "javascript"]

    def test_run_code_enum_follows_sandbox_languages(self):
        from src.agent.workspace_tools import workspace_tools
        box = MagicMock()
        box.languages = ["python", "bash"]
        box.network = False
        run_code = workspace_tools(box, "t")[0]
        assert run_code.parameters["properties"]["language"]["enum"] == ["python", "bash"]
        assert run_code.egress is False
        result = run_code.run({"language": "javascript", "code": "1"})
        assert not result.ok and "Unsupported" in result.text


# ── Untrusted-content guard ─────────────────────────────────────────────

def _tool(name, untrusted=False, egress=False):
    return Tool(name=name, description=name, parameters={"type": "object", "properties": {}},
                run=MagicMock(return_value=ToolResult(text="result")), running=lambda a: f"Running {name}",
                done=lambda a: f"Ran {name}", untrusted=untrusted, egress=egress, read_only=not egress)


class TestUntrustedGuard:
    def _run(self, turns, tools, guard=True, ask=None):
        llm = MagicMock(spec=LLMService)
        llm.complete_with_tools.side_effect = turns
        router = MagicMock(spec=ResearchRouter)
        router.enough.return_value = None
        agent = TaskAgent(llm, router, [], max_steps=5)
        return list(agent.run([], "q", "m", extra_tools=tools, controls=RunControls(ask=ask, guard_untrusted=guard)))

    def test_egress_after_untrusted_needs_approval(self):
        fetch, code = _tool("fetch", untrusted=True), _tool("code", egress=True)
        asks = []
        events = self._run([ToolTurn(tool_calls=[ToolCall("fetch", {})]), ToolTurn(tool_calls=[ToolCall("code", {})]),
                            ToolTurn()], [fetch, code], ask=lambda r: asks.append(r) or {"approved": False})
        assert len(asks) == 1 and asks[0]["tool"] == "code" and asks[0]["reason"] == UNTRUSTED_REASON
        code.run.assert_not_called()
        assert any(e.type == "action" and e.action["status"] == "denied" for e in events)

    def test_approved_egress_runs(self):
        fetch, code = _tool("fetch", untrusted=True), _tool("code", egress=True)
        self._run([ToolTurn(tool_calls=[ToolCall("fetch", {})]), ToolTurn(tool_calls=[ToolCall("code", {})]),
                   ToolTurn()], [fetch, code], ask=lambda r: {"approved": True})
        code.run.assert_called_once()

    def test_egress_before_untrusted_runs_freely(self):
        code = _tool("code", egress=True)
        ask = MagicMock()
        self._run([ToolTurn(tool_calls=[ToolCall("code", {})]), ToolTurn()], [code], ask=ask)
        ask.assert_not_called()
        code.run.assert_called_once()

    def test_guard_off(self):
        fetch, code = _tool("fetch", untrusted=True), _tool("code", egress=True)
        self._run([ToolTurn(tool_calls=[ToolCall("fetch", {})]), ToolTurn(tool_calls=[ToolCall("code", {})]),
                   ToolTurn()], [fetch, code], guard=False)
        code.run.assert_called_once()

    def test_unattended_declines(self):
        fetch, code = _tool("fetch", untrusted=True), _tool("code", egress=True)
        self._run([ToolTurn(tool_calls=[ToolCall("fetch", {})]), ToolTurn(tool_calls=[ToolCall("code", {})]),
                   ToolTurn()], [fetch, code], ask=None)
        code.run.assert_not_called()

    def test_default_tools_flags(self):
        from src.agent.tools import default_tools
        flags = {t.name: t.untrusted for t in default_tools()}
        assert flags == {"web_search": True, "news_search": True, "fetch_page": True, "get_weather": False}


# ── /v1 across the registry ─────────────────────────────────────────────

@pytest.fixture
def cloud_client(test_client, mock_registry):
    from src import server
    fake = ScriptedProvider(model_id="cloud-model", stream_text="cloud says hi")
    mock_registry.register(fake)
    return test_client


class TestV1Registry:
    def test_models_lists_registry(self, cloud_client):
        ids = [m["id"] for m in cloud_client.get("/v1/models").json()["data"]]
        assert ids == ["google/gemma-4-E4B-it", "cloud-model"]

    def test_completion_with_registry_model(self, cloud_client):
        from src.tasks import store
        data = cloud_client.post("/v1/chat/completions", json={
            "model": "cloud-model", "messages": [{"role": "user", "content": "hi"}]}).json()
        assert data["model"] == "cloud-model"
        assert data["choices"][0]["message"]["content"] == "cloud says hi"
        assert data["usage"]["prompt_tokens"] == 100 and data["usage"]["completion_tokens"] == 20
        assert [r["kind"] for r in store.usage_since(0)] == ["api"]

    def test_streaming_with_registry_model(self, cloud_client):
        resp = cloud_client.post("/v1/chat/completions", json={
            "model": "cloud-model", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        chunks = _sse(resp.text)
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "cloud says hi"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert resp.text.rstrip().endswith("data: [DONE]")

    def test_content_parts_accepted(self, cloud_client):
        resp = cloud_client.post("/v1/chat/completions", json={
            "model": "cloud-model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        assert resp.status_code == 200

    def test_unknown_model_404(self, cloud_client):
        resp = cloud_client.post("/v1/chat/completions", json={
            "model": "nope", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 404

    def test_monthly_limit_applies(self, cloud_client, monkeypatch):
        from src import server
        monkeypatch.setattr(server, "_monthly_budget", 0.0)
        resp = cloud_client.post("/v1/chat/completions", json={
            "model": "cloud-model", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 402


# ── Search index ────────────────────────────────────────────────────────

class TestSearchIndex:
    def _chat(self, name, *texts):
        from src import chat_store
        chat = chat_store.create_chat(name=name)
        for i, t in enumerate(texts):
            chat_store.append_messages(chat["id"], "user" if i % 2 == 0 else "assistant", t)
        return chat["id"]

    def test_substring_case_insensitive(self):
        from src import chat_store
        cid = self._chat("Plans", "tell me about Pythonic code", "Sure")
        hits = chat_store.search("THONIC")
        assert [h["id"] for h in hits] == [cid] and hits[0]["index"] == 0
        assert "Pythonic" in hits[0]["snippet"]

    def test_message_match_beats_title(self):
        from src import chat_store
        self._chat("zebra facts", "hello", "zebras are striped")
        hit = chat_store.search("zebra")[0]
        assert hit["index"] == 1 and hit["role"] == "assistant"

    def test_title_only_match(self):
        from src import chat_store
        self._chat("Quarterly budget", "hi")
        hit = chat_store.search("quarterly")[0]
        assert hit["index"] is None

    def test_delete_removes_from_index(self):
        from src import chat_store, search_index
        cid = self._chat("x", "findable words")
        chat_store.delete_chat(cid)
        assert search_index.search(chat_store.CHATS_DIR, "findable") == []

    def test_short_query_falls_back_to_scan(self):
        from src import chat_store
        cid = self._chat("x", "a b c")
        assert [h["id"] for h in chat_store.search("b")] == [cid]

    def test_sync_picks_up_offline_edits(self):
        from src import chat_store, search_index
        cid = self._chat("x", "original")
        path = chat_store.CHATS_DIR / f"{cid}.json"
        data = json.loads(path.read_text())
        data["messages"][0]["content"] = "edited offline"
        path.write_text(json.dumps(data))
        os.utime(path, (time.time() + 5, time.time() + 5))
        assert search_index.sync(chat_store.CHATS_DIR) == 1
        assert [h["id"] for h in chat_store.search("offline")] == [cid]

    def test_quotes_in_query(self):
        from src import chat_store
        cid = self._chat("x", 'he said "hello there"')
        assert [h["id"] for h in chat_store.search('"hello')] == [cid]


# ── Backup / restore ────────────────────────────────────────────────────

class TestBackup:
    def _seed(self, client):
        from src import memory_store
        chat_id = client.post("/api/chats").json()["id"]
        client.patch(f"/api/chats/{chat_id}", json={"name": "Backed up"})
        client.post(f"/api/chats/{chat_id}/files", json={"name": "notes.txt", "data_base64": "aGVsbG8="})
        project_id = client.post("/api/projects", json={"name": "Proj"}).json()["id"]
        memory_store.save("- likes tea")
        return chat_id, project_id

    def test_roundtrip_merge(self, test_client):
        from src import chat_store, memory_store, project_store
        chat_id, project_id = self._seed(test_client)
        resp = test_client.get("/api/backup")
        assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
        names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
        assert f"chats/{chat_id}.json" in names and "memory.txt" in names and "manifest.json" in names

        chat_store.delete_chat(chat_id)
        project_store.delete(project_id)
        memory_store.save("")
        result = test_client.post("/api/restore", content=resp.content).json()
        assert result["chats_restored"] == 1 and result["projects_restored"] == 1
        assert result["memory_restored"] is True
        assert chat_store.get_chat(chat_id)["name"] == "Backed up"
        assert any(chat_store.files_dir(chat_id).iterdir())
        assert memory_store.load().strip() == "- likes tea"
        assert [h["id"] for h in chat_store.search("Backed")] == [chat_id]

    def test_merge_skips_existing_replace_overwrites(self, test_client):
        from src import chat_store
        chat_id, _ = self._seed(test_client)
        archive = test_client.get("/api/backup").content
        chat_store.rename_chat(chat_id, "Changed here")
        merged = test_client.post("/api/restore", content=archive).json()
        assert merged["chats_skipped"] == 1 and chat_store.get_chat(chat_id)["name"] == "Changed here"
        replaced = test_client.post("/api/restore?mode=replace", content=archive).json()
        assert replaced["chats_restored"] == 1 and chat_store.get_chat(chat_id)["name"] == "Backed up"

    def _zip(self, members: dict[str, bytes | str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"app": "local-llm", "format": 1}))
            for k, v in members.items():
                zf.writestr(k, v)
        return buf.getvalue()

    @pytest.mark.parametrize("name", ["../evil.txt", "chats/../../evil.json",
                                      "chats/00000000-0000-0000-0000-000000000000/files/../x",
                                      ".env", "connectors.yaml"])
    def test_rejects_unexpected_paths(self, test_client, name):
        resp = test_client.post("/api/restore", content=self._zip({name: "x"}))
        assert resp.status_code == 400
        assert not (Path.cwd() / "evil.txt").exists()

    def test_rejects_non_backup(self, test_client):
        assert test_client.post("/api/restore", content=b"not a zip").status_code == 400
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", "x")
        assert test_client.post("/api/restore", content=buf.getvalue()).status_code == 400

    def test_rejects_mismatched_id(self, test_client):
        cid = "00000000-0000-0000-0000-000000000000"
        resp = test_client.post("/api/restore", content=self._zip({f"chats/{cid}.json": json.dumps({"id": "other"})}))
        assert resp.status_code == 400


# ── Misc ────────────────────────────────────────────────────────────────

class TestMisc:
    def test_invalid_chat_id_file_route_404(self, test_client):
        assert test_client.get("/api/chats/..%2E/files/" + "a" * 32).status_code == 404
        assert test_client.get("/api/chats/not-a-uuid/files/" + "a" * 32).status_code == 404

    def test_serves_built_frontend(self, test_client, tmp_path, monkeypatch):
        from src import server
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html>app</html>")
        (dist / "assets" / "app-123.js").write_text("console.log(1)")
        monkeypatch.setattr(server, "_FRONTEND_DIST", dist)
        assert test_client.get("/").text == "<html>app</html>"
        assert test_client.get("/chat/xyz").text == "<html>app</html>"  # SPA fallback
        js = test_client.get("/assets/app-123.js")
        assert js.text == "console.log(1)" and "immutable" in js.headers["cache-control"]
        assert test_client.get("/api/nope").status_code == 404
        assert test_client.get("/health").json()["status"] in ("ok", "loading")

    def test_frontend_not_built(self, test_client, tmp_path, monkeypatch):
        from src import server
        monkeypatch.setattr(server, "_FRONTEND_DIST", tmp_path / "missing")
        resp = test_client.get("/")
        assert resp.status_code == 404 and "npm run build" in resp.json()["detail"]
