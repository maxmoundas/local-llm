"""Background task runtime: event log, reconnects, cancel, questions/approvals, persistence, endpoints."""
from __future__ import annotations

import threading
import time

import pytest

from src import usage
from src.tasks import store
from src.tasks.manager import TERMINAL, TaskManager


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _drain(mgr, task_id, after=-1):
    return list(mgr.subscribe(task_id, after, heartbeat=0.05))


class TestEventLog:
    def test_events_in_order_and_terminal_last(self):
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: [ctx.emit("status", text="a"), ctx.emit("token", content="x")])
        events = [e for e in _drain(mgr, tid) if e["type"] != "ping"]
        assert [e["type"] for e in events] == ["task_start", "status", "token", "usage", TERMINAL]
        assert [e["seq"] for e in events] == list(range(5))
        assert events[-1]["status"] == "done"

    def test_reconnect_gets_only_later_events(self):
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: [ctx.emit("status", text=str(i)) for i in range(5)])
        _drain(mgr, tid)
        later = _drain(mgr, tid, after=3)
        assert [e["seq"] for e in later][0] == 4

    def test_live_subscriber_sees_events_as_they_happen(self):
        mgr = TaskManager()
        gate = threading.Event()

        def work(ctx):
            ctx.emit("status", text="first")
            gate.wait(2)
            ctx.emit("status", text="second")

        tid = mgr.start("c", "m", work)
        seen = []
        for e in mgr.subscribe(tid, heartbeat=0.05):
            seen.append(e)
            if e.get("text") == "first":
                gate.set()
        texts = [e.get("text") for e in seen if e["type"] == "status"]
        assert texts == ["first", "second"]

    def test_heartbeat_while_quiet(self):
        mgr = TaskManager()
        gate = threading.Event()
        tid = mgr.start("c", "m", lambda ctx: gate.wait(2))
        it = mgr.subscribe(tid, heartbeat=0.02)
        first = [next(it) for _ in range(2)]
        gate.set()
        list(it)  # let the task finish before the test's temp DB goes away
        assert first[0]["type"] == "task_start" and first[1]["type"] == "ping"

    def test_persisted_without_tokens_and_served_after_eviction(self, monkeypatch):
        import src.tasks.manager as m
        monkeypatch.setattr(m, "_KEEP_FINISHED_S", 0)
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: [ctx.emit("status", text="a"), ctx.emit("token", content="x")])
        _drain(mgr, tid)
        mgr._sweep()
        assert tid not in mgr._live
        events = _drain(mgr, tid)
        assert "token" not in [e["type"] for e in events]
        assert events[-1]["type"] == TERMINAL

    def test_failure_is_recorded(self):
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: 1 / 0)
        events = _drain(mgr, tid)
        assert any(e["type"] == "error" for e in events)
        assert store.get(tid)["status"] == "failed"

    def test_usage_is_metered_per_task(self):
        usage.set_price("priced", 1.0, 1.0)
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: usage.record("priced", 1000, 1000))
        events = _drain(mgr, tid)
        spent = next(e for e in events if e["type"] == "usage")
        assert spent["input_tokens"] == 1000 and spent["cost_usd"] == pytest.approx(0.002)
        assert store.get(tid)["usage"]["calls"] == 1


class TestControl:
    def test_cancel_sets_flag_and_status(self):
        mgr = TaskManager()

        def work(ctx):
            while not ctx.cancelled:
                time.sleep(0.01)

        tid = mgr.start("c", "m", work)
        assert mgr.cancel(tid)
        assert _drain(mgr, tid)[-1]["status"] == "cancelled"
        assert not mgr.cancel(tid)  # already finished

    def test_ask_and_reply(self):
        mgr = TaskManager()
        result = {}
        tid = mgr.start("c", "m", lambda ctx: result.update(reply=ctx.ask({"kind": "question", "question": "Q?"})))
        assert _wait(lambda: (mgr.active_for_chat("c") or {}).get("pending"))
        pending = mgr.active_for_chat("c")["pending"]
        assert store.get(tid)["status"] == "waiting"
        assert not mgr.reply(tid, {"request_id": "wrong", "answer": "no"})
        assert mgr.reply(tid, {"request_id": pending["id"], "answer": "yes"})
        types = [e["type"] for e in _drain(mgr, tid)]
        assert result["reply"]["answer"] == "yes"
        assert types.index("input_request") < types.index("input_resolved")

    def test_cancel_while_waiting_ends_task(self):
        mgr = TaskManager()
        tid = mgr.start("c", "m", lambda ctx: ctx.ask({"kind": "question", "question": "Q?"}))
        assert _wait(lambda: (mgr.active_for_chat("c") or {}).get("pending"))
        mgr.cancel(tid)
        assert _drain(mgr, tid)[-1]["status"] == "cancelled"

    def test_on_close_runs_after_task(self):
        mgr = TaskManager()
        closed = threading.Event()
        tid = mgr.start("c", "m", lambda ctx: ctx.on_close(closed.set))
        _drain(mgr, tid)
        assert closed.is_set()

    def test_recover_marks_orphans_interrupted(self):
        store.create("orphan", "c", "m")
        store.append_event("orphan", 0, "status", {"type": "status", "text": "x"})
        assert TaskManager().recover() == ["orphan"]
        assert store.get("orphan")["status"] == "interrupted"
        assert store.events("orphan")[-1]["status"] == "interrupted"


class TestEndpoints:
    @pytest.fixture(autouse=True)
    def no_agent(self, monkeypatch):
        from src import server
        from src.model import LocalLLM
        monkeypatch.setattr(server, "_router", None)
        monkeypatch.setattr(server, "_agent", None)
        monkeypatch.setattr(server, "_memory_enabled", False)
        monkeypatch.setattr(LocalLLM, "stream", lambda self, *a, **kw: iter(["hi ", "there"]))

    def test_message_stream_has_ids_and_task_header(self, test_client):
        chat_id = test_client.post("/api/chats").json()["id"]
        resp = test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"})
        assert resp.headers["x-task-id"]
        assert "id: 0\ndata: " in resp.text and '"task_end"' in resp.text

    def test_events_endpoint_replays_after_seq(self, test_client):
        chat_id = test_client.post("/api/chats").json()["id"]
        tid = test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"}).headers["x-task-id"]
        replay = test_client.get(f"/api/tasks/{tid}/events?after=0").text
        assert "id: 0\n" not in replay and '"task_end"' in replay
        assert test_client.get("/api/tasks/nope/events").status_code == 404

    def test_reply_rejected_when_not_waiting(self, test_client):
        chat_id = test_client.post("/api/chats").json()["id"]
        tid = test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"}).headers["x-task-id"]
        assert test_client.post(f"/api/tasks/{tid}/reply", json={"answer": "a"}).status_code == 409

    def test_task_files_download_and_escape(self, test_client):
        from src.agent import sandbox as sb
        chat_id = test_client.post("/api/chats").json()["id"]
        tid = test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"}).headers["x-task-id"]
        ws = sb.workspace_for(tid)
        (ws / "report.md").write_text("# hi")
        assert test_client.get(f"/api/tasks/{tid}/files/report.md").text == "# hi"
        assert test_client.get(f"/api/tasks/{tid}/files/missing.txt").status_code == 404
        assert test_client.get(f"/api/tasks/{tid}/files/..%2F..%2Fsecret").status_code in (400, 404)

    def test_busy_chat_rejects_second_message(self, test_client, monkeypatch):
        from src import server
        monkeypatch.setattr(server._tasks, "active_for_chat", lambda chat_id: {"task_id": "t", "pending": None})
        chat_id = test_client.post("/api/chats").json()["id"]
        assert test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"}).status_code == 409

    def test_deleting_chat_detaches_its_tasks(self, test_client):
        # Events go, but the task row stays (without the chat) so past spending still counts.
        chat_id = test_client.post("/api/chats").json()["id"]
        tid = test_client.post(f"/api/chats/{chat_id}/message", json={"content": "x"}).headers["x-task-id"]
        test_client.delete(f"/api/chats/{chat_id}")
        assert store.get(tid)["chat_id"] is None
        assert store.for_chat(chat_id) == []
