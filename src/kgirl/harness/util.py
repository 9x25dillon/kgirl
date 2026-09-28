"""Small pure helpers shared by every harness layer.

Nothing here does I/O except `harness_home`, which only resolves a path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

_IDENT_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def harness_home() -> Path:
    """Root for persistent harness state (indexes, memory, traces)."""
    return Path(os.environ.get("KGIRL_HOME", Path.home() / ".kgirl"))


def now() -> float:
    return time.time()


def sha1(text: str | bytes) -> str:
    data = text.encode("utf-8", "replace") if isinstance(text, str) else text
    return hashlib.sha1(data).hexdigest()


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token, never zero)."""
    return max(1, (len(text) + 3) // 4)


def split_identifier(name: str) -> list[str]:
    """`parseHTTPResponse_v2` -> ['parse', 'http', 'response', 'v', '2']."""
    out: list[str] = []
    for part in re.split(r"[_\W]+", name):
        if part:
            out.extend(m.group(0).lower() for m in _IDENT_SPLIT.finditer(part))
    return out


def search_terms(*texts: str) -> str:
    """Identifier-aware bag of words for full-text indexing."""
    seen: dict[str, None] = {}
    for text in texts:
        for w in _WORD.findall(text or ""):
            seen.setdefault(w.lower(), None)
            for piece in split_identifier(w):
                seen.setdefault(piece, None)
    return " ".join(seen)


def fts_query(text: str, max_terms: int = 12) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    terms: dict[str, None] = {}
    for w in _WORD.findall(text or ""):
        for t in [w.lower(), *split_identifier(w)]:
            if len(t) > 1 and t not in _STOP:
                terms.setdefault(t, None)
    picked = list(terms)[:max_terms]
    return " OR ".join(f'"{t}"' for t in picked)


def shingles(text: str, k: int = 3) -> set[str]:
    words = [w.lower() for w in _WORD.findall(text or "")]
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def similarity(a: str, b: str) -> float:
    """Word-bigram Jaccard: robust to small insertions in short rules."""
    return jaccard(shingles(a, 2), shingles(b, 2))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def first_paragraph(doc: str | None, limit: int = 300) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n", 1)[0]
    return clip(" ".join(para.split()), limit)


def unique(items: Iterable) -> list:
    return list(dict.fromkeys(items))


_STOP = frozenset(
    "a an and are as at be by do does for from how i in is it of on or the this to what "
    "where which who why with my me we you your our can should would could will".split()
)
