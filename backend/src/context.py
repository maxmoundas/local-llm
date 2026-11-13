"""Turn a saved chat into provider-ready messages that fit the model's context window.

    build(chat, system, model, ...) -> list[dict]

1. Each message's attachments are expanded: documents become <document> blocks in the text,
   images go in a neutral `images` list (providers convert them). Only the most recent images
   are sent to vision models; older ones, and all images for text-only models, become a note.
2. If the whole history doesn't fit `context_window - max_output - system`, the oldest turns
   are folded into a summary written by the utility model. The summary is cached on the chat
   (`context_summary`) with a fingerprint of what it covers, so it's only extended — not
   rewritten — as the chat grows, and is redone if an edit changes the summarized part.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from src import attachments
from src.logger import get_logger
from src.providers.base import ModelOption

_log = get_logger(__name__)

CHARS_PER_TOKEN = 3.5      # deliberately conservative across tokenizers
IMAGE_TOKENS = 1600
RECENT_IMAGE_MESSAGES = 3  # user messages whose images are still sent to vision models
SAFETY_TOKENS = 1024
SUMMARY_RESERVE = 1500

SUMMARY_SECTION = """

<earlier_conversation_summary>
The start of this conversation no longer fits in context. Summary of it:
{summary}
</earlier_conversation_summary>"""


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _msg_tokens(m: dict) -> int:
    return estimate_tokens(m.get("content") or "") + IMAGE_TOKENS * len(m.get("images") or []) + 4


def document_block(name: str, text: str) -> str:
    return f'<document name="{name}">\n{text}\n</document>'


def _expand(msg: dict, files_root: Path | None, send_images: bool, vision: bool) -> dict:
    """One saved message → neutral provider message (text with documents, plus images)."""
    text = msg.get("content") or ""
    docs, images, notes = [], [], []
    for f in msg.get("attachments") or []:
        if f.get("kind") == "document" and files_root is not None:
            docs.append(document_block(f["name"], attachments.text_of(files_root, f)))
        elif f.get("kind") == "image":
            b64 = attachments.image_base64(files_root, f) if (files_root is not None and send_images) else None
            if b64:
                images.append(b64)
            elif vision:
                notes.append(f"[Image attached earlier: {f['name']} — no longer shown]")
            else:
                notes.append(f"[Image attached: {f['name']} — the current model can't view images]")
    content = "\n\n".join([*docs, *notes, text] if (docs or notes) else [text])
    out = {"role": msg["role"], "content": content}
    if images:
        out["images"] = images
    return out


def _fingerprint(messages: list[dict]) -> str:
    h = hashlib.sha1()
    for m in messages:
        h.update(m["role"].encode())
        h.update((m.get("content") or "").encode())
    return h.hexdigest()[:16]


def _merge(msgs: list[dict]) -> list[dict]:
    """Drop empty turns and merge consecutive same-role turns (strict alternation for some providers)."""
    out: list[dict] = []
    for m in msgs:
        if not m.get("content") and not m.get("images"):
            continue
        if out and out[-1]["role"] == m["role"]:
            prev = out[-1]
            merged = {"role": m["role"], "content": f"{prev['content']}\n\n{m['content']}".strip()}
            imgs = [*(prev.get("images") or []), *(m.get("images") or [])]
            if imgs:
                merged["images"] = imgs
            out[-1] = merged
        else:
            out.append(dict(m))
    return out


def budget_for(model: ModelOption | None) -> int:
    """Input tokens available for system + history."""
    if model is None:
        return 28_000
    reserve = min(model.max_output_tokens, model.context_window // 4)
    return max(2_000, model.context_window - reserve - SAFETY_TOKENS)


def build(
    chat: dict,
    system: str,
    model: ModelOption | None,
    files_root: Path | None = None,
    summarize: Callable[[list[dict], str], str] | None = None,
    save_summary: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Provider messages for `chat`: system first, then as much history as fits."""
    vision = bool(model and model.vision)
    saved = [m for m in chat.get("messages", []) if m["role"] in ("user", "assistant")]

    # Which user messages may still send their images.
    image_msgs = [i for i, m in enumerate(saved) if m["role"] == "user"
                  and any(f.get("kind") == "image" for f in m.get("attachments") or [])]
    recent = set(image_msgs[-RECENT_IMAGE_MESSAGES:]) if vision else set()
    expanded = [_expand(m, files_root, i in recent, vision) for i, m in enumerate(saved)]

    budget = budget_for(model) - estimate_tokens(system)
    costs = [_msg_tokens(m) for m in expanded]
    if sum(costs) <= budget:
        return [{"role": "system", "content": system}, *_merge(expanded)]

    # Keep the newest messages that fit, leaving room for a summary of the rest.
    room = budget - SUMMARY_RESERVE
    cut, used = len(expanded), 0
    while cut > 0 and used + costs[cut - 1] <= room:
        cut -= 1
        used += costs[cut]
    if cut == len(expanded):  # even the latest message alone doesn't fit: trim its text
        last = dict(expanded[-1])
        keep_chars = int(max(room - IMAGE_TOKENS * len(last.get("images") or []), 500) * CHARS_PER_TOKEN)
        last["content"] = (last["content"][:keep_chars // 2] + "\n\n[… middle of this message omitted: too long "
                           "for the model's context window …]\n\n" + last["content"][-keep_chars // 2:])
        expanded[-1] = last
        cut = len(expanded) - 1
    while cut < len(expanded) - 1 and expanded[cut]["role"] != "user":
        cut += 1  # the kept part must start with a user turn

    summary = _summary_for(chat, saved[:cut], summarize, save_summary)
    note = SUMMARY_SECTION.format(summary=summary) if summary else (
        f"\n\n[The first {cut} messages of this conversation are omitted: they don't fit in the context window.]")
    _log.info("Context: %d of %d messages kept, %d %s", len(expanded) - cut, len(expanded), cut,
              "summarized" if summary else "dropped")
    return [{"role": "system", "content": system + note}, *_merge(expanded[cut:])]


def _summary_for(chat: dict, dropped: list[dict], summarize, save_summary) -> str:
    if not dropped or summarize is None:
        return ""
    cached = chat.get("context_summary") or {}
    upto = cached.get("upto", 0)
    try:
        if cached and upto == len(dropped) and cached.get("fp") == _fingerprint(dropped):
            return cached["text"]
        if cached and upto < len(dropped) and cached.get("fp") == _fingerprint(dropped[:upto]):
            text = summarize(dropped[upto:], cached["text"])        # extend the existing summary
        else:
            text = summarize(dropped, "")
    except Exception as exc:
        _log.warning("History summary failed, dropping old messages instead: %s", exc)
        return ""
    if text and save_summary is not None:
        save_summary({"upto": len(dropped), "fp": _fingerprint(dropped), "text": text})
    return text
