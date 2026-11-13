"""One chat turn, run as a background task.

Events (each also gets a `seq` from the task manager):

  status          one-line live status ("Thinking", "Writing the answer")
  action          a tool call starting / finishing — the UI upserts by action.id
  sources         web sources gathered
  input_request   the agent is waiting on the user (question or approval)
  input_resolved  …and got a reply
  token           answer text delta
  blocked         a guardrail stopped the input or output
  error           something failed
  memory          the user's saved memory changed because of this message (added / removed facts)
  done            turn finished; carries the chat name, files, and usage
  usage / task_end   emitted by the task manager when the task ends
"""
from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src import attachments as att
from src import chat_store, usage
from src.agent.connectors import ConnectorHub
from src.agent.router import ResearchRouter
from src.agent.runner import RunControls, TaskAgent, build_answer_system
from src.agent import sandbox as sb
from src.agent.sandbox import Sandbox
from src.agent.workspace_tools import list_workspace, workspace_tools
from src.guardrail_service import SUPPORT_SYSTEM_NOTE, GuardrailService
from src.llm_service import CALL_TYPE_CHAT, LLMService
from src.logger import get_logger
from src.providers.base import RefusalError
from src.tasks import store as task_store
from src.tasks.manager import Cancelled, TaskContext

_log = get_logger(__name__)


@dataclass
class TurnDeps:
    """A snapshot of the services and settings one turn runs with."""
    svc: LLMService
    build_messages: Callable[..., list[dict]]                 # (chat, model_id) → provider messages
    guardrails: GuardrailService | None = None
    guard_input: bool = True
    guard_output: bool = False
    # (user message, previous assistant reply) → {"added", "removed"} if memory changed, else None
    update_memory: Callable[[str, str], dict | None] | None = None
    user_memory: str = ""                                     # saved facts about the user, for the agent planner
    router: ResearchRouter | None = None
    agent: TaskAgent | None = None
    sandbox: Sandbox | None = None
    connectors: ConnectorHub | None = None
    agent_model_id: str | None = None   # model for agent steps; None → the chat model
    budget_usd: float | None = None
    max_seconds: float | None = None
    # Ask before code with network access / connector writes once the task has read web or connector content.
    guard_untrusted: bool = True
    # Scheduled runs: nobody is watching, so questions go unanswered and approvals are denied.
    unattended: bool = False
    # chat → files the agent can work on (copied into its workspace): (display name, path on disk).
    # Called when the agent starts, so the current message's attachments are included.
    workspace_files: Callable[[dict], list[tuple[str, Path]]] | None = None


MODES = ("send", "edit", "regenerate")


def run_turn(ctx: TaskContext, chat_id: str, content: str, image_base64: str | None, model_id: str,
             deps: TurnDeps, *, attachments: list[dict] | None = None, mode: str = "send",
             index: int | None = None) -> None:
    """One chat turn.

    mode "send"        append `content` (+ `attachments`, already stored in the chat's files) and answer
    mode "edit"        replace the user message at `index` with `content` (the old one stays a version)
    mode "regenerate"  answer the last user message again; the reply at `index` (if any) stays a version
    """
    try:
        _run_turn(ctx, chat_id, content, image_base64, model_id, deps, list(attachments or []), mode, index)
    finally:
        if mode == "regenerate":
            chat_store.restore_fork(chat_id)  # no-op once the new reply was saved


def _run_turn(ctx: TaskContext, chat_id: str, content: str, image_base64: str | None, model_id: str,
              deps: TurnDeps, attachments: list[dict], mode: str, index: int | None) -> None:
    chat = chat_store.get_chat(chat_id)
    is_new_chat = chat["name"] == "New Chat" and mode == "send"
    support_note = ""

    if image_base64:  # legacy single-image field
        attachments.append(att.save_base64(chat_store.files_dir(chat_id), "image.png", image_base64))

    if mode == "regenerate":
        if index is not None and index < len(chat["messages"]):
            chat = chat_store.fork(chat_id, index)
        last_user = next((m for m in reversed(chat["messages"]) if m["role"] == "user"), None)
        if last_user is None:
            ctx.emit("error", message="There is no message to answer.")
            return
        content = last_user["content"]
        attachments = list(last_user.get("attachments") or [])
    else:
        # Input guardrail — runs before saving so blocked messages aren't stored
        if deps.guardrails is not None and deps.guard_input:
            guard = deps.guardrails.check(content, "input")
            if guard.blocked:
                _log.info("Input blocked by guardrail: %s", guard.reason)
                ctx.emit("blocked", target="input", **({"reason": guard.reason} if guard.reason else {}))
                return
            if guard.action == "support":
                support_note = SUPPORT_SYSTEM_NOTE
        if mode == "edit" and index is not None:
            chat_store.fork(chat_id, index)
        extra = {"attachments": attachments} if attachments else None
        chat = chat_store.append_messages(chat_id, "user", content, model_id=model_id, extra=extra)

    messages = deps.build_messages(chat, model_id)
    messages[0]["content"] += support_note
    options = _options(chat.get("settings") or {})

    if deps.update_memory is not None and mode != "regenerate":
        prev = next((m.get("content") or "" for m in reversed(chat["messages"][:-1]) if m["role"] == "assistant"), "")
        memory_job = _MemoryJob(deps.update_memory, content, prev, chat_id)
    else:
        memory_job = None

    extra: dict = {"task_id": ctx.task_id, "model_id": model_id}
    started = time.time()
    images = [a for a in attachments if a.get("kind") == "image"]

    # Agent — the router decides per message. The planner can't see images, but they're staged in the
    # workspace for code, and the final answer (a vision model) still sees them alongside the notes.
    if deps.router is not None and deps.agent is not None:
        ctx.emit("status", text="Thinking")
        docs = [a for a in attachments if a.get("kind") == "document"]
        route_text = content + (f"\n\n[Attached and already readable: {', '.join(d['name'] for d in docs)}]" if docs else "")
        if images:
            route_text += f"\n\n[Attached images, visible to the answering model: {', '.join(i['name'] for i in images)}]"
        if deps.router.decide(route_text, chat["messages"]).research:
            planner_text = _with_documents(chat, content, docs)
            if images:
                planner_text += (f"\n\n(The user attached image(s): {', '.join(i['name'] for i in images)}. You can't "
                                 "see them, but the model writing the final answer can — gather only what the "
                                 "images can't tell. They're also in the workspace if code needs to process them.)")
            try:
                # The first planner turn must call a tool: weaker models otherwise talk themselves out
                # of acting. An occasional false-positive route costs one cheap lookup; a missed one
                # costs a wrong answer.
                notes, stopped = _run_agent(ctx, chat, planner_text, model_id, deps, extra)
            except Cancelled:
                notes, stopped = "", "cancelled"
            if ctx.cancelled:
                _save(chat_id, "", {**extra, "stopped": True}, ctx, started)
                ctx.emit("done", name=chat_store.get_chat(chat_id)["name"])
                return
            messages[0]["content"] = build_answer_system(messages[0]["content"], notes, stopped)
            ctx.emit("status", text="Writing the answer")

    full_response = ""
    try:
        for token in deps.svc.stream(messages, model_id, call_type=CALL_TYPE_CHAT, context_id=chat_id,
                                     options=options):
            full_response += token
            ctx.emit("token", content=token)
            if ctx.cancelled:
                break
    except RefusalError as exc:
        ctx.emit("error", message=f"The model declined this request. {exc}".strip())
        return
    except Exception as exc:
        # LLMService already logged the error; just surface it to the client.
        ctx.emit("error", message=str(exc))
        if full_response:
            _save(chat_id, full_response, {**extra, "stopped": True}, ctx, started)
        return

    if ctx.cancelled:
        if full_response or extra.get("actions"):
            _save(chat_id, full_response, {**extra, "stopped": True}, ctx, started)
        ctx.emit("done", name=chat_store.get_chat(chat_id)["name"])
        return

    # Output guardrail check — runs after full response is received
    if deps.guardrails is not None and deps.guard_output:
        guard = deps.guardrails.check(full_response, "output")
        if guard.blocked:
            _log.info("Output blocked by guardrail: %s", guard.reason)
            ctx.emit("blocked", target="output", **({"reason": guard.reason} if guard.reason else {}))
            return  # do not store the blocked response

    # Memory extraction runs alongside the answer and usually finishes first. Wait briefly so the
    # reply can show what was remembered; if it's slow, it still saves — just without the note.
    if memory_job is not None:
        change = memory_job.wait(_MEMORY_WAIT_S)
        if change and (change.get("added") or change.get("removed")):
            extra["memory"] = change
            ctx.emit("memory", **change)

    _save(chat_id, full_response, extra, ctx, started)

    # On the first exchange, generate a descriptive title via the cheapest model.
    if is_new_chat:
        chat_store.rename_chat(chat_id, deps.svc.generate_title(content))

    ctx.emit("done", name=chat_store.get_chat(chat_id)["name"], files=extra.get("files", []))


PLANNER_DOC_CHARS = 8000


def _with_documents(chat: dict, content: str, docs: list[dict]) -> str:
    """The user's message as the agent planner sees it: attached documents' text included (clipped)."""
    if not docs:
        return content
    root = chat_store.files_dir(chat["id"])
    parts = []
    for d in docs:
        text = att.text_of(root, d)
        clipped = text[:PLANNER_DOC_CHARS] + (f"\n[… {len(text) - PLANNER_DOC_CHARS:,} more characters; the full text "
                                              "goes to the final answer]" if len(text) > PLANNER_DOC_CHARS else "")
        parts.append(f'<document name="{d["name"]}">\n{clipped}\n</document>')
    return "\n\n".join([*parts, content])


def _options(settings: dict) -> dict | None:
    """Per-chat model options saved in chat["settings"]."""
    out = {k: settings[k] for k in ("temperature", "effort") if settings.get(k) is not None}
    return out or None


_MEMORY_WAIT_S = 8.0


class _MemoryJob:
    """Runs the memory updater on a daemon thread and keeps its result.

    It can outlive the turn (the turn waits only briefly), so it meters its own LLM calls and
    records them as separate "memory" usage instead of adding to the turn's meter.
    """

    def __init__(self, fn: Callable[[str, str], dict | None], content: str, prev: str,
                 chat_id: str | None = None) -> None:
        self._result: dict | None = None
        self._chat_id = chat_id
        self._thread = threading.Thread(target=self._run, args=(fn, content, prev), daemon=True, name="memory-update")
        self._thread.start()

    def _run(self, fn: Callable[[str, str], dict | None], content: str, prev: str) -> None:
        with usage.attach(()), usage.meter() as m:  # never count toward the turn's own meter
            try:
                self._result = fn(content, prev)
            except Exception:
                _log.exception("Memory update failed")
        if m.calls:
            try:
                task_store.record_usage(self._chat_id, None, m.full_snapshot(), "memory")
            except Exception:
                _log.exception("Could not record memory usage")

    def wait(self, timeout: float) -> dict | None:
        self._thread.join(timeout)
        return self._result


def _run_agent(ctx: TaskContext, chat: dict, content: str, model_id: str, deps: TurnDeps,
               extra: dict, require_first: bool = True) -> tuple[str, str]:
    """Run the agent loop, forwarding its events. Returns (work log, stop reason)."""
    tools = []
    if deps.sandbox is not None:
        tools += workspace_tools(deps.sandbox, ctx.task_id)
        sandbox = deps.sandbox
        ctx.on_close(lambda: sandbox.close(ctx.task_id))
    if deps.connectors is not None:
        tools += deps.connectors.tools

    files_to_stage = deps.workspace_files(chat) if deps.workspace_files and deps.sandbox else []
    uploaded = _stage_files(sb.WORKSPACES_DIR / ctx.task_id, files_to_stage)
    if uploaded:
        content += ("\n\n(Files the user provided are in your workspace and can be opened with code: "
                    + ", ".join(sorted(uploaded)) + ")")

    controls = RunControls(
        cancelled=lambda: ctx.cancelled,
        ask=None if deps.unattended else ctx.ask,
        budget_usd=deps.budget_usd,
        deadline=time.monotonic() + deps.max_seconds if deps.max_seconds else None,
        meter=ctx.meter,
        guard_untrusted=deps.guard_untrusted,
    )
    notes, stopped = "", ""
    actions: list[dict] = []
    for evt in deps.agent.run(chat["messages"], content, deps.agent_model_id or model_id, context_id=ctx.chat_id or "",
                              extra_tools=tools, controls=controls, require_first=require_first,
                              user_memory=deps.user_memory):
        if evt.type == "status":
            ctx.emit("status", text=evt.text)
        elif evt.type == "action":
            ctx.emit("action", action=evt.action)
        elif evt.type == "done":
            notes, stopped, actions = evt.notes, evt.stopped, evt.actions
            if evt.sources:
                extra["sources"] = evt.sources
                ctx.emit("sources", sources=evt.sources)
    if actions:
        extra["actions"] = actions
    files = [f for f in list_workspace(sb.WORKSPACES_DIR / ctx.task_id) if f["path"] not in uploaded]
    if files:
        extra["files"] = files
    return notes, stopped


def _stage_files(ws: Path, files: list[tuple[str, Path]]) -> set[str]:
    """Copy the user's files into the task workspace. Returns the names used."""
    names: set[str] = set()
    for name, src in files:
        if not src.is_file():
            continue
        safe = Path(name).name.replace("/", "_") or src.name
        stem, suffix, n = Path(safe).stem, Path(safe).suffix, 1
        while safe in names:
            n += 1
            safe = f"{stem} ({n}){suffix}"
        ws.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, ws / safe)
        names.add(safe)
    return names


def _save(chat_id: str, content: str, extra: dict, ctx: TaskContext, started: float) -> None:
    extra = {**extra, "elapsed_ms": int((time.time() - started) * 1000), "usage": ctx.meter.snapshot()}
    chat_store.append_messages(chat_id, "assistant", content, extra=extra)
