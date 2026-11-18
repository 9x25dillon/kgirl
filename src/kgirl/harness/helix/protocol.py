"""Trailers: the few machine-read lines a model writes, and the checks that make them evidence.

A block may end with ASCII trailer lines, each a tag and one JSON value:

    TUNE: {"quote": "...", "axis": "rigor", "d": 0.1, "because": "..."}      or  TUNE: none
    DEFECT: {"quote": "...", "repair": "..."}
    CONCUR: {"evidence": "..."}
    CAL: {"p": 0.7, "method": "...", "falsifier": "...", "drops_below": 0.4}
    UNRESOLVED: [1, 3]

What the model writes is a claim; what this module returns is the claim after checking. A quote must occur
word for word in the text it is about, so a re-steer or a defect cannot be earned by paraphrase. A
calibration without its method and falsifier is void. Nothing here raises on bad model output: a trailer
that fails a check comes back with `ok=False` and the reason, for the log.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .state import AXES, MAX_DELTA

TAGS = ("TUNE", "DEFECT", "CONCUR", "CAL", "UNRESOLVED")
MIN_QUOTE_WORDS = 3
MAX_PASSES = 2               # tuning passes per block
MAX_DEFECTS = 5
MALFORMED = object()
_TAG = re.compile(r"^[ \t>*`#-]*(" + "|".join(TAGS) + r")\**[ \t]*:\**[ \t]*", re.M)
# What a trailer leaves behind once it is cut out. Each pattern is narrow on purpose: a reply's own code
# blocks and horizontal rules must survive.
_FENCE = re.compile(r"^[ \t]*```\w*[ \t]*$", re.M)
_EMPTY_FENCES = re.compile(r"^[ \t]*```\w*[ \t]*\n(?:[ \t]*\n)*[ \t]*```[ \t]*$\n?", re.M)
_LEFTOVER = re.compile(r"^[ \t]*(?:\*{1,2}|`{1,2}|[#*]+[ \t]*(?:" + "|".join(TAGS) + r"|Trailers?)[*: \t]*)[ \t]*$\n?",
                       re.M | re.I)
_PLUS = re.compile(r"(:\s*)\+(?=[\d.])")
_DROP = str.maketrans({"*": None, "`": None, "“": '"', "”": '"', "‘": "'", "’": "'"})


@dataclass(frozen=True)
class Trailer:
    tag: str
    value: object            # a dict or list, None for `none`, or MALFORMED


@dataclass(frozen=True)
class Tune:
    ok: bool
    axis: str = ""
    d: float = 0.0
    quote: str = ""
    because: str = ""
    reason: str = ""         # why it was rejected, or a note such as a clamped delta


@dataclass(frozen=True)
class Defect:
    ok: bool
    quote: str = ""
    repair: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Review:
    defects: tuple = ()      # every DEFECT trailer, valid or void
    concur: str | None = None        # the evidence of a valid CONCUR
    reason: str = ""         # why a CONCUR was void

    @property
    def valid(self) -> list:
        return [d for d in self.defects if d.ok]

    @property
    def void(self) -> bool:
        return not self.valid and self.concur is None


@dataclass(frozen=True)
class Calibration:
    ok: bool
    p: float = 0.0
    method: str = ""
    falsifier: str = ""
    drops_below: float | None = None
    reason: str = ""


# ----------------------------------------------------------------------------------------------- splitting

def _json_end(s: str) -> int | None:
    """Index just past the JSON object or array that starts at s[0], or None when it never closes."""
    depth, quoted, escaped = 0, False, False
    for i, ch in enumerate(s):
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _noise_only(s: str) -> bool:
    """Whether s is nothing but what may sit between trailers of one block: blank lines, code-fence lines
    and leftover decoration. A line of prose is not noise, so it ends the trailing block."""
    return not _LEFTOVER.sub("", _FENCE.sub("", s)).strip()


def split(text: str) -> tuple[str, list[Trailer], int]:
    """Separate a reply into its body and its trailers. Returns (body, trailers, trailer characters).

    Only a trailing block counts: the trailers must be a suffix of the reply, separated by noise alone, with
    noise alone after the last one. A tag line with prose after it is body, not a trailer, so a model that
    writes about `TUNE:` mid-answer does not steer anything. The JSON may sit on the tag's line or start on
    the next one, with or without a code fence, and a `+0.1` is read as 0.1: small models write both."""
    trailers: list[Trailer] = []
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _TAG.finditer(text):
        if m.start() < last:                       # inside a value that was already consumed
            continue
        at = m.end()
        lead = re.match(r"\s*(?:```\w*\s*)?", text[at:])
        start = at + lead.end()
        none = re.match(r"\s*none\b[^\n]*", text[at:], re.I)
        if none:
            value, end = None, at + none.end()
        elif text[start:start + 1] in ("{", "["):
            n = _json_end(text[start:])
            try:
                value = json.loads(_PLUS.sub(r"\1", text[start:start + n])) if n else MALFORMED
            except ValueError:
                value = MALFORMED
            end = start + n if n else (text.find("\n", at) if "\n" in text[at:] else len(text))
        else:
            value, end = MALFORMED, text.find("\n", at) if "\n" in text[at:] else len(text)
        if value is not MALFORMED and value is not None and not isinstance(value, (dict, list)):
            value = MALFORMED
        end += re.match(r"[ \t*`]*", text[end:]).end()                 # emphasis closed after the value
        trailers.append(Trailer(m.group(1), value))
        spans.append((m.start(), end))
        last = end
    keep = len(spans)                              # keep only the trailing block of candidate trailers
    if spans and _noise_only(text[spans[-1][1]:]):
        keep = len(spans) - 1
        while keep > 0 and _noise_only(text[spans[keep - 1][1]:spans[keep][0]]):
            keep -= 1
    trailers, spans = trailers[keep:], spans[keep:]
    body, pos = [], 0
    for a, b in spans:
        body.append(text[pos:a])
        pos = b
    body.append(text[pos:])
    clean = "".join(body)
    if spans:
        clean = _LEFTOVER.sub("", _EMPTY_FENCES.sub("", clean))
        fences = list(_FENCE.finditer(clean))
        if len(fences) % 2:                        # the fence a trailer's JSON sat in lost its partner
            last = fences[-1]
            clean = clean[:last.start()] + clean[last.end():]
    return clean.strip(), trailers, sum(b - a for a, b in spans)


# ----------------------------------------------------------------------------------------------- quotes

def _norm(s: str) -> str:
    return " ".join(s.translate(_DROP).split())


def clean_quote(quote: str) -> str:
    return _norm(quote).strip("\"' .…")


def quoted(quote: str, source: str) -> bool:
    """Whether `quote` occurs word for word in `source` (whitespace, emphasis marks and quote style aside)
    and is long enough to point at something."""
    q = clean_quote(quote)
    return len(q.split()) >= MIN_QUOTE_WORDS and q in _norm(source)


def _text(obj, key: str) -> str:
    v = obj.get(key) if isinstance(obj, dict) else None
    return v.strip() if isinstance(v, str) else ""


def _number(x) -> float | None:
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return None
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return float(x)
    return None


# ----------------------------------------------------------------------------------------------- parsers

def parse_tunes(trailers: list[Trailer], block: str) -> list[Tune]:
    """Every TUNE trailer of a block, checked: a known axis, a finite non-zero delta (clamped to MAX_DELTA),
    a reason, and a quote that is in the block. `TUNE: none` yields nothing."""
    out: list[Tune] = []
    for t in trailers:
        if t.tag != "TUNE" or t.value is None:
            continue
        if t.value is MALFORMED or not isinstance(t.value, dict):
            out.append(Tune(False, reason="trailer is not a JSON object"))
            continue
        axis, quote, because = _text(t.value, "axis").lower(), _text(t.value, "quote"), _text(t.value, "because")
        d = _number(t.value.get("d"))
        if axis not in AXES:
            out.append(Tune(False, reason=f"unknown axis {axis!r}"))
        elif d is None or d == 0:
            out.append(Tune(False, axis, reason="d is not a non-zero number"))
        elif not because:
            out.append(Tune(False, axis, d, quote, reason="no reason given"))
        elif not quoted(quote, block):
            out.append(Tune(False, axis, d, quote, because,
                            reason=f"the quote is not a span of at least {MIN_QUOTE_WORDS} words from the block"))
        else:
            held = min(MAX_DELTA, max(-MAX_DELTA, d))
            out.append(Tune(True, axis, held, clean_quote(quote), because,
                            reason=f"d {d} clamped to {held}" if held != d else ""))
    return out


def parse_review(trailers: list[Trailer], draft: str, seen: str | None = None) -> Review:
    """The critic's trailers, checked. A defect needs a repair and a quote found in `draft`: the expert
    blocks themselves, not the harness's notes about them. A concurrence needs evidence that is in nothing the
    critic was shown (`seen`, default the draft)."""
    seen = draft if seen is None else seen
    defects: list[Defect] = []
    concur, reason = None, ""
    for t in trailers:
        if t.tag == "DEFECT":
            if t.value is MALFORMED or not isinstance(t.value, dict):
                defects.append(Defect(False, reason="trailer is not a JSON object"))
                continue
            quote, repair = _text(t.value, "quote"), _text(t.value, "repair")
            if not repair:
                defects.append(Defect(False, quote, reason="no repair proposed"))
            elif not quoted(quote, draft):
                defects.append(Defect(False, quote, repair, reason="the quote is not in the draft"))
            elif len([d for d in defects if d.ok]) >= MAX_DEFECTS:
                defects.append(Defect(False, quote, repair, reason=f"more than {MAX_DEFECTS} defects"))
            else:
                defects.append(Defect(True, clean_quote(quote), repair))
        elif t.tag == "CONCUR":
            evidence = _text(t.value, "evidence") if isinstance(t.value, dict) else ""
            if not evidence:
                reason = "concurrence gives no evidence"
            elif clean_quote(evidence) in _norm(seen):
                reason = "concurrence repeats the draft instead of adding evidence"
            else:
                concur = evidence
    return Review(tuple(defects), concur, reason)


def parse_calibration(trailers: list[Trailer]) -> Calibration:
    """The first CAL trailer: a probability with the method that produced it and what would falsify it."""
    for t in trailers:
        if t.tag != "CAL":
            continue
        if t.value is MALFORMED or not isinstance(t.value, dict):
            return Calibration(False, reason="trailer is not a JSON object")
        p, low = _number(t.value.get("p")), _number(t.value.get("drops_below"))
        method, falsifier = _text(t.value, "method"), _text(t.value, "falsifier")
        if p is None or not 0.0 <= p <= 1.0:
            return Calibration(False, reason="p is not a probability")
        if not method or not falsifier:
            return Calibration(False, p, method, falsifier, low,
                               reason="a number without its " + ("method" if not method else "falsifier"))
        if low is not None and not 0.0 <= low < p:
            return Calibration(False, p, method, falsifier, low, reason="drops_below must be a probability under p")
        return Calibration(True, p, method, falsifier, low)
    return Calibration(False, reason="no CAL trailer")


def parse_unresolved(trailers: list[Trailer], n: int) -> list[int] | None:
    """The 1-based numbers of the conflicts the synthesis left open, or None when it did not say (missing or
    malformed trailer, or a number outside 1..n): the caller then keeps every conflict open."""
    for t in trailers:
        if t.tag != "UNRESOLVED":
            continue
        if not isinstance(t.value, list):
            return None
        if any(not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= n for i in t.value):
            return None
        return sorted(set(t.value))
    return None
