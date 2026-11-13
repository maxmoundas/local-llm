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
  done            turn finished; carries the chat name, files, and usage
  usage / task_end   emitted by the task manager when the task ends
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from src import chat_store
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
from src.tasks.manager import Cancelled, TaskContext

_log = get_logger(__name__)


@dataclass
class TurnDeps:
    """A snapshot of the services and settings one turn runs with."""
    svc: LLMService
    build_messages: Callable[[dict], list[dict]]
    guardrails: GuardrailService | None = None
    guard_input: bool = True
    guard_output: bool = False
    update_memory: Callable[[str], None] | None = None
    router: ResearchRouter | None = None
    agent: TaskAgent | None = None
    sandbox: Sandbox | None = None
    connectors: ConnectorHub | None = None
    agent_model_id: str | None = None   # model for agent steps; None → the chat model
    budget_usd: float | None = None
    max_seconds: float | None = None


def run_turn(ctx: TaskContext, chat_id: str, content: str, image_base64: str | None, model_id: str,
             deps: TurnDeps) -> None:
    chat = chat_store.get_chat(chat_id)
    is_new_chat = chat["name"] == "New Chat"
    support_note = ""

    # Input guardrail — runs before appending so blocked messages aren't stored
    if deps.guardrails is not None and deps.guard_input:
        guard = deps.guardrails.check(content, "input")
        if guard.blocked:
            _log.info("Input blocked by guardrail: %s", guard.reason)
            ctx.emit("blocked", target="input", **({"reason": guard.reason} if guard.reason else {}))
            return
        if guard.action == "support":
            support_note = SUPPORT_SYSTEM_NOTE

    chat = chat_store.append_messages(chat_id, "user", content, model_id=model_id)
    messages = deps.build_messages(chat)
    messages[0]["content"] += support_note

    if deps.update_memory is not None:
        threading.Thread(target=deps.update_memory, args=(content,), daemon=True).start()

    extra: dict = {"task_id": ctx.task_id}
    started = time.time()

    # Agent — the router decides per message; images skip it (tools can't see them).
    if deps.router is not None and deps.agent is not None and not image_base64:
        ctx.emit("status", text="Thinking")
        if deps.router.decide(content, chat["messages"]).research:
            try:
                # The first planner turn must call a tool: weaker models otherwise talk themselves out
                # of acting. An occasional false-positive route costs one cheap lookup; a missed one
                # costs a wrong answer.
                notes, stopped = _run_agent(ctx, chat, content, model_id, deps, extra)
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
        for token in deps.svc.stream(messages, model_id, image_base64=image_base64,
                                     call_type=CALL_TYPE_CHAT, context_id=chat_id):
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

    _save(chat_id, full_response, extra, ctx, started)

    # On the first exchange, generate a descriptive title via the cheapest model.
    if is_new_chat:
        chat_store.rename_chat(chat_id, deps.svc.generate_title(content))

    ctx.emit("done", name=chat_store.get_chat(chat_id)["name"], files=extra.get("files", []))


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

    controls = RunControls(
        cancelled=lambda: ctx.cancelled,
        ask=ctx.ask,
        budget_usd=deps.budget_usd,
        deadline=time.monotonic() + deps.max_seconds if deps.max_seconds else None,
        meter=ctx.meter,
    )
    notes, stopped = "", ""
    actions: list[dict] = []
    for evt in deps.agent.run(chat["messages"], content, deps.agent_model_id or model_id, context_id=ctx.chat_id or "",
                              extra_tools=tools, controls=controls, require_first=require_first):
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
    files = list_workspace(sb.WORKSPACES_DIR / ctx.task_id)
    if files:
        extra["files"] = files
    return notes, stopped


def _save(chat_id: str, content: str, extra: dict, ctx: TaskContext, started: float) -> None:
    extra = {**extra, "elapsed_ms": int((time.time() - started) * 1000), "usage": ctx.meter.snapshot()}
    chat_store.append_messages(chat_id, "assistant", content, extra=extra)
