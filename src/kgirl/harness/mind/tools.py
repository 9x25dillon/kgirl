"""The tool-call protocol between the chat model and kgirl.

The model calls a tool by replying with one line:

    <tool>NAME {"arg": "value"}</tool>

That form is backend-neutral (plain text, so every model and the scripted test backend can produce it)
and easy to hold back while streaming. Small local models drift from it, so `parse_tool_call` also reads
what they tend to write instead:

    <tool_call>{"name": "NAME", "arguments": {...}}</tool_call>     the format Qwen models were trained on
    <tool>NAME {"spec": "line one
    line two"}                                                       no closing tag; raw newlines in a string

A call that is recognisably a call but cannot be read comes back with `error` set, so the chat loop can
tell the model what was wrong instead of showing the user a stray tag.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

_OPEN = re.compile(r"<\s*(tool_call|tool)\s*>", re.I)
_CLOSE = re.compile(r"<\s*/\s*(tool_call|tool)\s*>", re.I)
_NAME = re.compile(r"\s*([A-Za-z_][\w-]*)")
_DECODER = json.JSONDecoder(strict=False)      # strict=False: a model may put real newlines inside a JSON string

CALL_FORMAT = '<tool>NAME {"arg": "value"}</tool>'
STOP = ["</tool>", "</tool_call>"]             # generation stops here: the model must not invent the tool's result


def call_complete(text: str) -> bool:
    """True once a reply holds a whole tool call (for backends that ignore stop sequences)."""
    start = _OPEN.search(text or "")
    return bool(start and _CLOSE.search(text, start.end()))


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    prefix: str = ""        # what the model wrote before the call (already streamed to the user)
    error: str = ""         # why the call could not be read ("" when it could)

    def render(self) -> str:
        """The canonical form kept in the transcript, so the model sees its own calls written correctly."""
        return f"<tool>{self.name} {json.dumps(self.args, ensure_ascii=False)}</tool>"


@dataclass(frozen=True)
class Tool:
    name: str
    usage: str                               # the arguments as the model should write them
    help: str
    run: Callable[[dict], str]


def parse_tool_call(raw: str) -> ToolCall | None:
    """The first tool call in a reply, or None when the reply is an answer."""
    raw = raw or ""
    start = _OPEN.search(raw)
    if not start:
        return None
    prefix = raw[:start.start()].strip()
    body = raw[start.end():]
    end = _CLOSE.search(body)
    body = (body[:end.start()] if end else body).strip()
    body = re.sub(r"^```[\w-]*\s*|\s*```$", "", body).strip()
    if not body:
        return ToolCall("", prefix=prefix, error="the tool call is empty")
    if body.startswith("{"):                  # {"name": ..., "arguments": {...}}
        obj, err = _object(body)
        if obj is None:
            # A streamed call is often cut off after its name but before the arguments object closes.
            # Keep the name so the chat loop can give precise retry feedback instead of losing the call.
            partial = re.search(r'["\'](?:name|tool)["\']\s*:\s*["\']([A-Za-z_][\w-]*)', body, re.I)
            name = partial.group(1).lower() if partial else ""
            return ToolCall(name, prefix=prefix, error=err)
        name = obj.get("name") or obj.get("tool") or ""
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, str):             # some models write the arguments as a JSON string
            args, err = _object(args)
    else:                                     # NAME {json}
        named = _NAME.match(body)
        if not named:
            return ToolCall("", prefix=prefix, error="the tool call does not start with a tool name")
        name, rest = named.group(1), body[named.end():].strip()
        args, err = _object(rest) if rest else ({}, "")
    name = name.strip().lower() if isinstance(name, str) else ""
    if not name:
        return ToolCall("", prefix=prefix, error="the tool call names no tool")
    if not isinstance(args, dict):
        return ToolCall(name, prefix=prefix, error=err or "the arguments must be a JSON object")
    return ToolCall(name, args, prefix)


def _object(text: str) -> tuple[dict | None, str]:
    """The first JSON object in `text`, or (None, why not)."""
    at = text.find("{")
    if at < 0:
        return None, "the arguments must be a JSON object, for example {\"query\": \"...\"}"
    try:
        obj, _ = _DECODER.raw_decode(text, at)
    except ValueError as exc:
        return None, f"the arguments are not valid JSON ({exc})"
    if not isinstance(obj, dict):
        return None, "the arguments must be a JSON object"
    return obj, ""


def strip_calls(text: str) -> str:
    """A reply with every tool call (or the start of one) removed: what may be shown as an answer."""
    start = _OPEN.search(text or "")
    return (text or "")[:start.start()].strip() if start else (text or "").strip()


def tools_doc(tools: list[Tool]) -> str:
    lines = ["TOOLS. To call one, reply with exactly one line and nothing after it:", CALL_FORMAT,
             "You'll get the result, then continue. One tool per reply. When you have what you need, answer."]
    lines += [f"- {t.name} {t.usage}: {t.help}" for t in tools]
    return "\n".join(lines)
