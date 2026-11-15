"""Reflection: turn what was said into what is remembered.

After every few exchanges (and when a conversation ends) the reflector reads the turns since its last
pass, together with the rolling summary, and asks the model for a small JSON record:

    summary      the conversation so far, rewritten (this is what later prompts carry)
    title        a short name for the thread
    threads      open questions and unfinished tasks
    facts        durable facts about the user, their projects and their machine
    preferences  how the user wants kgirl to work
    lessons      what helped or hurt in kgirl's own answers
    feedback     did the user approve, correct, or neither, of kgirl's last answer

How it becomes habit (and why it cannot run away): facts, preferences and lessons are *proposed* to the
Curator with source "reflection". A proposal is staged; an independent later observation of the same
thing reinforces it; the curator promotes it only once support >= 2 and its utility lower bound clears
the policy, and decays and retires it when it stops helping. One conversation can suggest a habit; only
repetition makes it one. Anything the user states explicitly ("remember ...") is stored active at once.

Feedback closes the loop: the memory fragments an answer was built from are credited with a win when the
user approves and a loss when the user corrects it, so useful memories rise in recall and bad ones sink.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..soup import Curator, Soup
from .conversations import ConversationStore, Turn
from .text import first_json, one_line

REFLECT_SYSTEM = """You maintain the long-term memory of kgirl, a local coding and writing assistant.
Read the conversation excerpt and the previous summary, then return ONLY one JSON object:
{"summary": "<the whole conversation so far in <= 120 words: goals, decisions, results, current state>",
 "title": "<3-7 word title>",
 "threads": ["<open question or unfinished task>", ...],
 "facts": ["<durable fact about the user, their projects, tools or machine>", ...],
 "preferences": ["<how the user wants the assistant to behave or write>", ...],
 "lessons": ["<a reusable lesson about what helped or failed in the assistant's answers>", ...],
 "feedback": "positive" | "negative" | "none"}
Rules: only include what the excerpt actually shows; no guesses. Each item is one self-contained sentence
that still makes sense months later (name the project, no "this" or "it"). Empty lists are fine.
"feedback" judges the user's reaction to the assistant's most recent answer."""

_POS = re.compile(r"\b(thanks|thank you|perfect|great|works|worked|nice|exactly|awesome|love it|that'?s right|"
                  r"correct|good job|well done)\b", re.I)
_NEG = re.compile(r"\b(wrong|incorrect|doesn'?t work|didn'?t work|not what i|no,|nope|broken|fails?|failed|"
                  r"that'?s not|you misunderstood|stop|bad)\b", re.I)
_REMEMBER = re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind)(?:\s+that)?[:,]?\s+(.+)$", re.I | re.S)

LLM = Callable[[str, str, int, float], str]   # (system, prompt, max_tokens, temperature) -> text


def quick_feedback(text: str) -> str:
    """Cheap per-turn read of the user's reaction; reflection refines it later."""
    neg, pos = bool(_NEG.search(text)), bool(_POS.search(text))
    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "positive"
    return "none"


def explicit_memory(text: str) -> str | None:
    m = _REMEMBER.match(text)
    return " ".join(m.group(1).split()) if m else None


@dataclass
class Reflection:
    summary: str = ""
    title: str = ""
    threads: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    feedback: str = "none"
    proposed: list[tuple[int, str]] = field(default_factory=list)

    @classmethod
    def parse(cls, obj) -> "Reflection":
        if not isinstance(obj, dict):
            return cls()

        def items(key: str) -> list[str]:
            v = obj.get(key) or []
            v = v if isinstance(v, list) else [v]
            return [one_line(str(x), 280) for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:6]

        fb = str(obj.get("feedback", "none")).lower()
        return cls(one_line(str(obj.get("summary", "")), 1200), one_line(str(obj.get("title", "")), 80),
                   items("threads"), items("facts"), items("preferences"), items("lessons"),
                   fb if fb in ("positive", "negative") else "none")


class Reflector:
    def __init__(self, llm: LLM, store: ConversationStore, soup: Soup, curator: Curator, every: int = 3):
        self.llm = llm
        self.store = store
        self.soup = soup
        self.curator = curator
        self.every = every
        self._lock = threading.Lock()

    def due(self, conv_id: int) -> bool:
        conv = self.store.get(conv_id)
        if not conv:
            return False
        fresh = self.store.turns(conv_id, since_id=conv.reflected_turn, roles=("user",))
        return len(fresh) >= self.every

    def credit_last_answer(self, conv_id: int, reaction: str) -> list[int]:
        """Credit the memory the previous assistant answer was built from. Returns the credited ids."""
        if reaction not in ("positive", "negative"):
            return []
        turns = self.store.turns(conv_id, roles=("assistant",))
        if not turns or not turns[-1].recalled:
            return []
        ids = [i for i in turns[-1].recalled if isinstance(i, int)]
        self.soup.credit(ids, success=reaction == "positive")
        return ids

    def reflect(self, conv_id: int, force: bool = False) -> Reflection | None:
        with self._lock:
            conv = self.store.get(conv_id)
            if not conv:
                return None
            turns = self.store.turns(conv_id, since_id=conv.reflected_turn, roles=("user", "assistant"))
            if not turns or (not force and sum(t.role == "user" for t in turns) < self.every):
                return None
            prompt = (f"PREVIOUS SUMMARY:\n{conv.summary or '(none: new conversation)'}\n\n"
                      f"CONVERSATION EXCERPT:\n{render_turns(turns, 6000)}")
            try:
                raw = self.llm(REFLECT_SYSTEM, prompt, 900, 0.1)
            except Exception:  # noqa: BLE001 - reflection must never break a chat
                return None
            ref = Reflection.parse(first_json(raw))
            if not ref.summary:
                return None
            self.store.update(conv_id, summary=ref.summary, title=conv.title or ref.title or None,
                              threads=ref.threads, reflected_turn=turns[-1].id)
            self._episode(conv_id, ref, conv.scope)
            src = f"reflection:conv{conv_id}"
            for kind, texts in (("fact", ref.facts), ("preference", ref.preferences), ("abstraction", ref.lessons)):
                for text in texts:
                    fid, how = self.curator.propose(text, kind=kind, tags="conversation", source=src, scope=conv.scope)
                    ref.proposed.append((fid, how))
            if ref.feedback == "negative" and ref.lessons:
                fid, how = self.curator.propose(f"avoid: {ref.lessons[0]}", kind="antipattern", tags="conversation",
                                                source=src, scope=conv.scope)
                ref.proposed.append((fid, how))
            return ref

    def _episode(self, conv_id: int, ref: Reflection, scope: str) -> int:
        """One active `episode` fragment per conversation, replaced as the summary grows."""
        marker = f'{{"conv": {conv_id}}}'
        title = f"{ref.title}: " if ref.title else ""
        text = " ".join(f"conversation #{conv_id} {title}{ref.summary}".split())
        rows = self.soup.db.execute("SELECT id, text FROM fragments WHERE kind='episode' AND data=? AND status='active'",
                                    (marker,)).fetchall()
        for row in rows:
            if row["text"] == text:
                return row["id"]
            self.soup.set_status(row["id"], "retired", "superseded by a newer summary")
        fid = self.soup.add(text, kind="episode", tags="conversation", source="conversation", scope=scope, data=marker)
        self.soup.set_status(fid, "active", "episode summary")   # an identical retired summary comes back
        return fid


def render_turns(turns: list[Turn], budget_chars: int) -> str:
    lines, used = [], 0
    for t in reversed(turns):
        body = t.content if len(t.content) < 1600 else t.content[:1500] + " …[cut]"
        line = f"{t.role.upper()}: {body}"
        if lines and used + len(line) > budget_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n\n".join(reversed(lines))
