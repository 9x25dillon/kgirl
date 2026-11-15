"""Mind: kgirl as a conversational agent that remembers, uses tools, learns from feedback and reflects.

One user turn:

    1. feedback   the user's reaction to the previous answer credits (or debits) the memories it used
    2. explicit   "remember ..." is stored at once, as the user's own words (active, source "user")
    3. context    system prompt = persona + tools + what kgirl knows about the user (active preferences)
                  + memory recalled for this message (facts, lessons, skills, episodes of other conversations)
                  + matching turns from earlier conversations + this conversation's rolling summary and
                  open threads; messages = the most recent turns that fit the budget
    4. act        the model answers, or calls a tool with <tool>name {json}</tool>; results come back as a
                  message and it continues, up to `max_steps` tool calls
    5. record     the answer is stored with the ids of every memory it was built from (for step 1 next time)
    6. reflect    every few user turns, the reflector rewrites the summary and proposes facts, preferences
                  and lessons to the curator (see reflect.py): repetition turns them into habits

Nothing here changes code or weights. kgirl improves by what it keeps, what it recalls, and what the
user's reactions teach the recall ranking.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..hermes.router import CHAT, SMITH
from ..llm import Message
from ..util import clip
from .builder import Builder
from .conversations import ConversationStore
from .reflect import Reflector, explicit_memory, quick_feedback
from .text import one_line, strip_reasoning
from .writer import Writer

PERSONA = """You are kgirl, the user's own local AI engineer and writing partner. You run entirely on their
machine. You know their repositories through the Atlas index and you remember past conversations.
How you work:
- Think like a senior engineer: understand the goal, name assumptions, prefer simple, elegant, correct
  solutions, and say plainly when you are unsure or when something is missing.
- Use what you remember (MEMORY, EARLIER CONVERSATIONS, THIS CONVERSATION) when it is relevant, and say
  so ("last time we ..."). Never invent memories, files or results.
- Use a tool whenever the answer depends on the user's code or on something you need to look up or build.
- Be direct and concrete. Code in fenced blocks. No filler."""

TOOLS_DOC = """TOOLS. To call one, reply with exactly one line and nothing else:
<tool>NAME {"arg": "value"}</tool>
You'll get the result, then continue. Call at most one tool per reply.
- search_code {"query": "...", "repo": "optional repo name"}: find symbols and code in the user's repos
- read_file {"repo": "...", "path": "...", "start": 1, "end": 80}: read lines of an indexed file
- recall {"query": "..."}: search long-term memory and earlier conversations
- remember {"text": "...", "kind": "fact|preference|abstraction"}: propose something worth keeping
- build {"spec": "..."}: turn a precise spec into a tested Python module (slow: a few minutes)
- write {"brief": "...", "words": 800}: plan, draft and revise a long piece of text (slow)
- repos {}: list the indexed repositories"""

_TOOL = re.compile(r"<tool>\s*([a-z_]+)\s*(\{.*?\})?\s*</tool>", re.S)


@dataclass
class Reply:
    text: str
    conv_id: int
    recalled: list[int] = field(default_factory=list)
    tools: list[tuple[str, dict]] = field(default_factory=list)
    stored: list[str] = field(default_factory=list)


class Mind:
    def __init__(self, assistant, store: ConversationStore | None = None, reflect_every: int = 3,
                 history_budget: int = 2400, memory_budget: int = 700, max_steps: int = 4,
                 background_reflection: bool = True):
        self.a = assistant
        self.store = store or ConversationStore(assistant.home / "conversations.db")
        self.reflector = Reflector(self._llm_text, self.store, assistant.soup, assistant.curator, reflect_every)
        self.history_budget = history_budget
        self.memory_budget = memory_budget
        self.max_steps = max_steps
        self.background = background_reflection
        self._reflecting: threading.Thread | None = None
        self.last_build = None

    # ------------------------------------------------------------------ model access

    def _llm_text(self, system: str, prompt: str, max_tokens: int, temperature: float, role: str = CHAT) -> str:
        out = self.a.bus.request("llm.complete", {"role": role, "system": system, "max_tokens": max_tokens,
                                                  "temperature": temperature,
                                                  "messages": [Message("user", prompt)]}, sender="mind")
        return strip_reasoning(out.text)

    def _code_llm(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        return self._llm_text(system, prompt, max_tokens, temperature, role=SMITH)

    # ------------------------------------------------------------------ conversations

    def start(self, title: str = "", scope: str = "*") -> int:
        return self.store.new(title, scope)

    def resume(self) -> int:
        conv = self.store.latest()
        return conv.id if conv else self.start()

    # ------------------------------------------------------------------ one turn

    def chat(self, conv_id: int, text: str, on_text: Callable[[str], None] | None = None,
             on_event: Callable[[str], None] | None = None) -> Reply:
        say = on_event or (lambda msg: None)
        text = text.strip()
        reaction = quick_feedback(text)
        credited = self.reflector.credit_last_answer(conv_id, reaction)
        if credited:
            say(f"{'+' if reaction == 'positive' else '-'} credited memories {credited}")
        stored = []
        fact = explicit_memory(text)
        if fact:
            kind = "preference" if re.search(r"\b(i (prefer|like|want|hate)|always|never|don'?t|please)\b", fact, re.I) \
                else "fact"
            fid = self.a.remember(fact, kind=kind, tags="conversation")
            stored.append(f"{kind}#{fid}")
            say(f"remembered {kind} #{fid}")
        self.store.append(conv_id, "user", text)

        system, recalled = self._system(conv_id, text)
        tools_used: list[tuple[str, dict]] = []
        answer = ""
        for step in range(self.max_steps + 1):
            msgs = self._messages(conv_id)
            raw = self._generate(system, msgs, on_text)
            call = _TOOL.search(raw)
            if not call or step == self.max_steps:
                answer = _TOOL.sub("", raw).strip() or raw.strip()
                break
            name, args = call.group(1), _args(call.group(2))
            say(f"tool {name} {json.dumps(args, ensure_ascii=False)[:120]}")
            result = self._tool(name, args, say)
            tools_used.append((name, args))
            self.store.append(conv_id, "tool", f"[{name}] {clip(result, 6000)}", meta={"tool": name, "args": args})
        answer = answer or "(no answer)"
        self.store.append(conv_id, "assistant", answer, recalled=recalled, meta={"tools": [t for t, _ in tools_used]})
        self._maybe_reflect(conv_id)
        return Reply(answer, conv_id, recalled, tools_used, stored)

    def end(self, conv_id: int) -> None:
        """Finish a conversation: wait for any reflection, then reflect over whatever is left."""
        if self._reflecting and self._reflecting.is_alive():
            self._reflecting.join(timeout=300)
        self.reflector.reflect(conv_id, force=True)
        self.a.curator.step()

    # ------------------------------------------------------------------ context

    def _system(self, conv_id: int, text: str) -> tuple[str, list[int]]:
        soup = self.a.soup
        conv = self.store.get(conv_id)
        parts = [PERSONA, TOOLS_DOC]
        recalled: list[int] = []
        profile = latest_profile(soup)
        if profile:
            parts.append(f"WHO THE USER IS (kgirl's consolidated profile):\n{profile.text}")
            recalled.append(profile.id)
        prefs = soup.recall(text, budget_tokens=260, kinds=("preference",))
        if prefs.fragments:
            parts.append("WHAT THE USER WANTS (their preferences):\n" + "\n".join(f"- {f.text}" for f in prefs.fragments))
            recalled += prefs.ids
        mem = soup.recall(text, budget_tokens=self.memory_budget,
                          kinds=("fact", "abstraction", "antipattern", "skill", "episode", "note"),
                          exclude=tuple(recalled))
        if mem.fragments:
            parts.append("MEMORY (recalled for this message; may be partial):\n" + mem.render())
            recalled += mem.ids
        older = [t for t in self.store.search(text, limit=4, exclude_conv=conv_id) if t.role in ("user", "assistant")]
        if older:
            parts.append("EARLIER CONVERSATIONS (matching turns):\n" + "\n".join(
                f"- conv #{t.conv_id} {t.role}: {one_line(t.content, 260)}" for t in older[:4]))
        if conv and (conv.summary or conv.threads):
            s = f"THIS CONVERSATION SO FAR: {conv.summary}" if conv.summary else ""
            if conv.threads:
                s += "\nOPEN THREADS:\n" + "\n".join(f"- {t}" for t in conv.threads)
            parts.append(s.strip())
        return "\n\n".join(parts), recalled

    def _messages(self, conv_id: int) -> list[Message]:
        out: list[Message] = []
        for t in self.store.window(conv_id, self.history_budget):
            if t.role == "tool":
                out.append(Message("user", f"TOOL RESULT {t.content}"))
            else:
                out.append(Message(t.role, t.content))
        # Ollama and Anthropic want alternation that starts with the user: merge neighbours of the same role.
        merged: list[Message] = []
        for m in out:
            if merged and merged[-1].role == m.role:
                merged[-1] = Message(m.role, merged[-1].content + "\n\n" + m.content)
            else:
                merged.append(m)
        while merged and merged[0].role != "user":
            merged.pop(0)
        return merged

    def _generate(self, system: str, msgs: list[Message], on_text: Callable[[str], None] | None) -> str:
        """Stream the reply; hold text back once a tool call starts so the user never sees the tag."""
        buf, shown, hold = "", 0, False
        for piece in self.a.router.stream(CHAT, system, msgs, max_tokens=1600, temperature=0.4):
            buf += piece
            if not on_text or hold:
                continue
            visible = strip_reasoning(buf) if "</think>" in buf or "<think>" not in buf else ""
            cut = visible.find("<tool")
            if cut >= 0:
                hold = True
                visible = visible[:cut]
            else:   # keep back a partial "<to" at the end
                tail = visible.rfind("<")
                if tail >= 0 and "<tool"[: len(visible) - tail] == visible[tail:]:
                    visible = visible[:tail]
            if len(visible) > shown:
                on_text(visible[shown:])
                shown = len(visible)
        if on_text and shown:
            on_text("\n")
        return strip_reasoning(buf)

    # ------------------------------------------------------------------ tools

    def _tool(self, name: str, args: dict, say: Callable[[str], None]) -> str:
        a = self.a
        try:
            if name == "search_code":
                ctx, cites = a.context_pack(str(args.get("query", "")), args.get("repo") or None, budget=2200)
                return ctx or "no matches; the repo may not be indexed"
            if name == "read_file":
                start = int(args.get("start", 1))
                return a.atlas.read_source(str(args["repo"]), str(args["path"]), start, int(args.get("end", start + 80)))
            if name == "recall":
                q = str(args.get("query", ""))
                rec = a.soup.recall(q, budget_tokens=700)
                old = self.store.search(q, limit=6)
                return ((rec.render() or "(no stored memory)") + "\n\nconversation turns:\n" +
                        ("\n".join(f"- conv #{t.conv_id} {t.role}: {one_line(t.content, 300)}" for t in old) or "(none)"))
            if name == "remember":
                kind = args.get("kind", "fact")
                kind = kind if kind in ("fact", "preference", "abstraction") else "fact"
                fid, how = a.curator.propose(one_line(str(args.get("text", "")), 300), kind=kind, tags="conversation",
                                             source="reflection:chat")
                return f"{how} {kind} #{fid} (becomes active once it is observed again or the user promotes it)"
            if name == "repos":
                return "\n".join(f"{r['name']} ({r['nfiles']} files)" for r in a.atlas.repos()) or "no repos indexed"
            if name == "build":
                say("building: tests first, several candidates, agreement selection (takes a few minutes)")
                res = self.builder().build(str(args.get("spec", "")))
                self.last_build = res
                return f"{res.render()}\n\n```python\n{res.code}```" if res.code else res.render()
            if name == "write":
                say("writing: plan, draft by section, critique, revise")
                ctx = a.soup.recall(str(args.get("brief", "")), budget_tokens=500,
                                    kinds=("preference", "fact", "episode")).render()
                d = Writer(self._llm_text).write(str(args.get("brief", "")), ctx, _int(args.get("words")),
                                                 progress=lambda m: say(m))
                return d.markdown()
        except (KeyError, ValueError, TypeError, OSError, FileNotFoundError) as exc:
            return f"tool {name} failed: {exc}"
        return f"unknown tool {name!r}; use one of the tools listed"

    def builder(self, **kw) -> Builder:
        return Builder(self._code_llm, self.a.soup,
                       emit=lambda topic, payload: self.a.bus.publish(topic, payload, "builder"), **kw)

    # ------------------------------------------------------------------ reflection

    def _maybe_reflect(self, conv_id: int) -> None:
        if not self.reflector.due(conv_id):
            return
        if not self.background:
            self.reflector.reflect(conv_id)
            return
        if self._reflecting and self._reflecting.is_alive():
            return
        self._reflecting = threading.Thread(target=self.reflector.reflect, args=(conv_id,), daemon=True,
                                            name="kgirl-reflect")
        self._reflecting.start()

    def recap(self, conv_id: int) -> str:
        conv = self.store.get(conv_id)
        if not conv:
            return "no such conversation"
        lines = [f"conversation #{conv.id}: {conv.title or '(untitled)'}",
                 f"summary: {conv.summary or '(not reflected yet)'}"]
        lines += [f"open: {t}" for t in conv.threads]
        lines.append(f"turns: {self.store.count(conv.id)}")
        return "\n".join(lines)


def latest_profile(soup):
    """The newest active profile note written by `evolve` (None before the first consolidation)."""
    row = soup.db.execute("SELECT id FROM fragments WHERE kind='note' AND tags='profile' AND status='active'"
                          " ORDER BY id DESC LIMIT 1").fetchone()
    return soup.get(row["id"]) if row else None


def _args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None

