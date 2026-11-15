"""Helpers for reading what local models write: strip reasoning, pull out JSON and code."""

from __future__ import annotations

import json
import re

_THINK = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.S)
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.S)


def strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks that reasoning-distilled models emit before the answer."""
    return _THINK.sub("", text or "").strip()


def code_blocks(text: str, lang: str | None = None) -> list[str]:
    """Fenced code blocks, optionally only those tagged with `lang` (untagged blocks always count)."""
    out = []
    for tag, body in _FENCE.findall(strip_reasoning(text)):
        if lang is None or not tag or tag.lower() in (lang, lang[:2]):
            out.append(body.rstrip() + "\n")
    return out


def python_code(text: str) -> str:
    """The Python module in a reply: the longest python block, else the whole reply if it parses."""
    blocks = code_blocks(text, "python")
    if blocks:
        return max(blocks, key=len)
    body = strip_reasoning(text)
    try:
        compile(body, "<reply>", "exec")
        return body.rstrip() + "\n"
    except SyntaxError:
        return ""


def first_json(text: str):
    """The first JSON object or array in a reply (fenced or bare); None if there isn't one."""
    text = strip_reasoning(text)
    for body in code_blocks(text, "json") + [text]:
        dec = json.JSONDecoder()
        for i, ch in enumerate(body):
            if ch in "{[":
                try:
                    obj, _ = dec.raw_decode(body, i)
                    return obj
                except ValueError:
                    continue
    return None


def one_line(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
