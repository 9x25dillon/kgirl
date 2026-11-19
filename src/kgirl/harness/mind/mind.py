"""Mind: kgirl as a conversational agent that remembers, uses tools, learns from feedback and reflects.

One user turn:

    1. feedback   the opening of the user's message is read as a verdict on the previous answer; a clear
                  one credits (or debits) the memories that answer was built from
    2. explicit   "remember that ..." is stored at once, as the user's own words (active, source "user")
    3. context    system prompt = persona + the user's rules + tools + what kgirl knows about the user
                  (profile, preferences) + memory relevant to this message (facts, lessons, skills, episodes
                  of other conversations) + earlier conversations on the same topic (by tag) or with
                  matching turns + the indexed code the message names + this conversation's rolling
                  summary, open commitments and open threads; messages = the most recent turns.
                  One token budget covers all of it (budget.py): the rules are never cut, the rest is
                  trimmed by priority, and what was cut is recorded in the turn's meta and shown by /context
    4. act        the model answers, or calls a tool with <tool>name {json}</tool>; the result comes back
                  as a message and it continues, up to `max_steps` tool calls
    5. record     the transcript keeps what happened: the text before each call, the call, the result, and
                  the final answer with the ids of every memory it was built from (for step 1 next time)
                  and the system prompt it was generated from
    6. reflect    every few user turns, the reflector rewrites the summary and proposes facts, preferences
                  and lessons to the curator (see reflect.py): repetition turns them into habits

With a workspace attached, kgirl can also look around a project directory, change files (each change
approved by the user and undoable) and run commands the user allowed (see workspace.py). The `task` tool
hands a code change to a swarm of sandboxed agents; the verified diff is applied only by `/apply`.

Nothing here changes weights. kgirl improves by what it keeps, what it recalls, and what the user's
reactions teach the recall ranking.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..atlas.secrets import secret_reason
from ..hermes.router import BUILD, CHAT
from ..jev.swarm import apply_edits
from ..llm import Message
from ..util import clip, estimate_tokens
from .builder import Builder
from .budget import PRIORITY, Block, fit, render_report
from .conversations import ConversationStore, clean_tags, tags_in
from .reflect import Reflector, episode_marker, explicit_memory, observed_in, quick_feedback, set_observed_in
from .rules import check as rules_check, render_rules, revision_prompt, standing_rules, warning as rules_warning
from .text import one_line, strip_reasoning
from .tools import CALL_FORMAT, STOP, Tool, call_complete, parse_tool_call, strip_calls, tools_doc
from .workspace import Workspace, WorkspaceError, allowlisted_commands
from .writer import Draft, Writer, style_guide

PERSONA = """You are kgirl, the user's own local AI engineer and writing partner. You run entirely on their
machine. You know their repositories through the Atlas index and you remember past conversations.
How you work:
- Think like a senior engineer: understand the goal, name assumptions, prefer simple, elegant, correct
  solutions, and say plainly when you are unsure or when something is missing.
- Use what you remember (MEMORY, EARLIER CONVERSATIONS, THIS CONVERSATION) when it is relevant, and say
  so ("last time we ..."). Never invent memories, files or results.
- Use a tool whenever the answer depends on the user's code or files, or on something you need to look
  up, run or build. Look before you change: read a file before you edit it.
- Only say you changed, ran or verified something when a TOOL RESULT shows it.
- The user's RULES are binding. Never write code or commands that break them, not even in an example or a
  "full script". If the task seems to need such a step, leave it out and say which step the user should run.
- Be direct and concrete. Code in fenced blocks. No filler."""

NO_MORE_TOOLS = ("\n\nYou have used every tool call available for this message. Answer now from what you have; "
                 "do not call another tool. Say what is still unknown.")
OUT_OF_STEPS = "(I used up my tool steps before reaching an answer. Say \"continue\" and I will carry on.)"
DRAFT_SAVED = "(saved as draft #{id}; `kgirl write --resume {id}` continues one)"   # last line of the write tool's result
OLD_TOOL_CHARS = 500          # what stays of a tool result from an earlier exchange
PROMPT_SHARE = 0.92           # of the room left for the prompt: estimates run low for code, hence the headroom
# Rules v2 takes the active rules as a third argument; the older check() does not.
_CHECK_TAKES_RULES = len(inspect.signature(rules_check).parameters) >= 3

# Things in a message that may name code: `backticked`, paths and file names, snake_case and CamelCase names.
_NAMED = re.compile(r"`([^`\n]{2,80})`"
                    r"|((?:[\w.-]+/)+[\w.-]+\.\w{1,5}|\b[\w-]+\.(?:py|ts|tsx|js|jl|rs|go|c|cpp|h|sh|md|toml|json|ya?ml))\b"
                    r"|\b([A-Za-z_]\w*(?:_\w+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")


@dataclass
class Prompt:
    """One turn's prompt: the system text, the memory ids it was built from, and what to record in the turn's meta
    (`budget`: block sizes and what was cut; `related`: the earlier conversations picked, and why)."""
    system: str
    recalled: list[int]
    meta: dict


@dataclass
class Reply:
    text: str
    conv_id: int
    recalled: list[int] = field(default_factory=list)
    tools: list[tuple[str, dict]] = field(default_factory=list)
    stored: list[str] = field(default_factory=list)
    verdict: str = "none"         # what this message said about the previous answer


class Mind:
    def __init__(self, assistant, store: ConversationStore | None = None, reflect_every: int = 3,
                 history_budget: int | None = None, memory_budget: int | None = None, max_steps: int = 8,
                 background_reflection: bool = True, workspace: Workspace | None = None,
                 verify_cmd: list[str] | None = None, context_tokens: int | None = None):
        self.a = assistant
        self.store = store or ConversationStore(assistant.home / "conversations.db")
        self.reflector = Reflector(self._llm_text, self.store, assistant.soup, assistant.curator, reflect_every)
        # Budgets follow the context window the chat model is loaded with, so a larger window is used
        # and a smaller one is not overrun. Estimates run low for code, hence the headroom.
        ctx = context_tokens or assistant.router.context_tokens(CHAT) or 4096
        self.context_tokens = ctx
        self.answer_tokens = max(400, min(1600, ctx // 5))
        self.history_budget = history_budget or max(600, int(ctx * 0.26))
        self.memory_budget = memory_budget or max(200, int(ctx * 0.08))
        self.tool_chars = max(1500, int(ctx * 0.6))
        self.max_steps = max_steps
        self.background = background_reflection
        self.workspace = workspace
        self.verify_cmd = verify_cmd          # set by the user (--verify or /verify), never by the model
        self._reflecting: threading.Thread | None = None
        self._say: Callable[[str], None] = lambda msg: None
        self.last_build = None
        self._rules: list = []               # the standing rules in the latest system prompt
        self._history_cap: int | None = None  # tokens of older turns the budget left, for this turn
        self._budgets: dict[int, dict] = {}   # conversation -> the budget meta of its latest prompt
        self._conv: int | None = None         # the conversation being answered (tools that save work use it)
        self.last_draft: int | None = None    # the draft artifact written last, even if the writing was interrupted
        self._audit_seq: int | None = None    # the newest workspace audit event already recorded in a tool turn
        self.last_task = None                 # (SwarmResult, repo root) waiting for /apply
        self.last_edit = None                 # (BuildResult, workspace path, original text) waiting for /apply

    # ------------------------------------------------------------------ model access

    def _llm_text(self, system: str, prompt: str, max_tokens: int, temperature: float, role: str = CHAT) -> str:
        out = self.a.bus.request("llm.complete", {"role": role, "system": system, "max_tokens": max_tokens,
                                                  "temperature": temperature,
                                                  "messages": [Message("user", prompt)]}, sender="mind")
        return strip_reasoning(out.text)

    def _code_llm(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        return self._llm_text(system, prompt, max_tokens, temperature, role=BUILD)

    # ------------------------------------------------------------------ conversations

    def scope(self) -> str:
        """The memory scope of a conversation started here: the workspace's repo or directory name, else "*"."""
        if self.workspace is None:
            return "*"
        for r in self.a.atlas.repos():
            if Path(r["root"]).resolve() == self.workspace.root:
                return r["name"]
        return self.workspace.name or "*"

    def start(self, title: str = "", scope: str | None = None) -> int:
        return self.store.new(title, scope or self.scope())

    def resume(self) -> int:
        """The latest conversation of this workspace (of any, without a workspace), or a new one."""
        scope = self.scope()
        conv = self.store.latest(None if scope == "*" else scope)
        return conv.id if conv else self.start(scope=scope)

    def forget(self, conv_id: int) -> str:
        """Delete a conversation: its turns, its episode summary, and proposals only it supported."""
        if not self.store.get(conv_id):
            return f"no conversation #{conv_id}"
        turns = self.store.delete(conv_id)
        soup = self.a.soup
        rows = soup.db.execute("SELECT id FROM fragments WHERE status != 'retired' AND ((kind='episode' AND data=?)"
                               " OR (status='staged' AND source LIKE 'reflection:conv%'))",
                               (episode_marker(conv_id),)).fetchall()
        retired = 0
        for row in rows:
            frag = soup.get(row["id"])
            if frag.kind != "episode":
                seen = observed_in(frag)
                if conv_id not in seen:
                    continue
                if seen - {conv_id}:     # other conversations restated it: it stands on them, only this trace goes
                    set_observed_in(soup, frag.id, seen - {conv_id})
                    continue
            soup.set_status(frag.id, "retired", f"conversation #{conv_id} deleted")
            retired += 1
        return f"deleted conversation #{conv_id}: {turns} turns, {retired} memory fragments retired"

    # ------------------------------------------------------------------ one turn

    def chat(self, conv_id: int, text: str, on_text: Callable[[str], None] | None = None,
             on_event: Callable[[str], None] | None = None) -> Reply:
        say = self._say = on_event or (lambda msg: None)
        self._conv = conv_id
        text = text.strip()
        verdict = quick_feedback(text)
        credited = self.reflector.credit_answer(conv_id, verdict)
        if credited is not None:
            say(f"{'+' if verdict == 'positive' else '-'} read as feedback on the last answer"
                + (f"; credited memories {credited}" if credited else ""))
        stored = []
        memo = explicit_memory(text)
        if memo:
            fid = self.a.remember(memo[0], kind=memo[1], tags="conversation")
            stored.append(f"{memo[1]}#{fid}")
            say(f"remembered {memo[1]} #{fid}")
        self.store.append(conv_id, "user", text)

        prompt = self._prompt(conv_id, text)
        system, recalled = prompt.system, prompt.recalled
        used: list[tuple[str, dict]] = []
        answer = ""
        for _ in range(self.max_steps):
            raw = self._generate(system, self._messages(conv_id), on_text)
            call = parse_tool_call(raw)
            if call is None:
                answer = raw.strip()
                break
            if call.error:
                written = clip(raw[len(call.prefix):].strip(), 400)     # what the model wrote, for the record
                result = f"that tool call could not be read: {call.error}. Write exactly: {CALL_FORMAT}"
            else:
                written = call.render()
                say(f"tool {call.name} {clip(json.dumps(call.args, ensure_ascii=False), 120)}")
                result = self._run_tool(call.name, call.args)
                used.append((call.name, call.args))
            self.store.append(conv_id, "assistant", f"{call.prefix}\n{written}".strip(),
                              meta={"partial": True, "tool": call.name})
            audit = self._audit_new()
            self.store.append(conv_id, "tool", f"[{call.name or 'error'}] {result}",
                              meta={"tool": call.name, "args": call.args, **({"audit": audit} if audit else {})})
        else:   # the tool budget is spent: one reply without tools
            answer = strip_calls(self._generate(system + NO_MORE_TOOLS, self._messages(conv_id), on_text))
            if not answer:
                answer = OUT_OF_STEPS
                if on_text:
                    on_text(answer + "\n")
        if not answer:
            answer = "(the model returned nothing; try rephrasing)"
            if on_text:
                on_text(answer + "\n")
        answer, rules_meta = self._enforce_rules(conv_id, system, text, answer, on_text)
        tid = self.store.append(conv_id, "assistant", answer, recalled=recalled,
                                meta={"tools": [name for name, _ in used], **rules_meta, **prompt.meta})
        self.store.save_context(tid, system)
        self._maybe_reflect(conv_id)
        return Reply(answer, conv_id, recalled, used, stored, verdict if credited is not None else "none")

    def _enforce_rules(self, conv_id: int, system: str, text: str, answer: str,
                       on_text: Callable[[str], None] | None) -> tuple[str, dict]:
        """Check the answer's code against the user's rules; on a hit, revise once, and warn if it still breaks one."""
        hits = self._rules_check(answer, text)
        if not hits:
            return answer, {}
        self._say("rules check: " + "; ".join(f"{v.name} ({v.snippet})" for v in hits) + " -> revising")
        if on_text:
            on_text("\n\n---\n*kgirl's rules check caught a step you asked it not to take; revising...*\n\n")
        msgs = self._messages(conv_id) + [Message("assistant", answer),
                                          Message("user", revision_prompt(hits, self._rules))]
        try:
            revised = strip_calls(self._generate(system + NO_MORE_TOOLS, msgs, on_text)).strip()
        except Exception as exc:  # noqa: BLE001 - a failed revision must not lose the answer
            self._say(f"rules revision failed: {type(exc).__name__}")
            revised = ""
        final = revised or answer
        left = self._rules_check(final, text)
        if left:
            note = rules_warning(left)
            final += note
            if on_text:
                on_text(note)
        return final, {"rules_hits": [v.name for v in hits], "rules_left": [v.name for v in left],
                       "rules_revised": bool(revised)}

    def _audit_new(self) -> list[dict]:
        """Workspace changes and runs since the last tool turn, as plain records for that turn's meta["audit"]
        (the workspace keeps its own append-only audit; this only copies what is new, and nothing without one)."""
        since = getattr(self.workspace, "audit_since", None)
        if since is None:
            return []
        events = since(self._audit_seq or 0)
        if events:
            self._audit_seq = max(e.seq for e in events)
        return [dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(e) for e in events]

    def _rules_check(self, answer: str, text: str) -> list:
        return rules_check(answer, text, self._rules) if _CHECK_TAKES_RULES else rules_check(answer, text)

    def feedback(self, conv_id: int, verdict: str) -> str:
        """The user's explicit verdict on the last answer (/good, /bad). It overrides an inferred one."""
        ids = self.reflector.credit_answer(conv_id, verdict, source="command", override=True)
        if ids is None:
            return "there is no answer to judge yet"
        return f"noted: {verdict}" + (f"; credited memories {ids}" if ids else "; that answer used no stored memory")

    def end(self, conv_id: int) -> None:
        """Finish a conversation: wait for any reflection, then reflect over whatever is left."""
        if self._reflecting and self._reflecting.is_alive():
            self._reflecting.join(timeout=300)
        self.reflector.reflect(conv_id, force=True)
        self.a.curator.step()

    # ------------------------------------------------------------------ context

    @property
    def prompt_limit(self) -> int:
        """Tokens the whole prompt (system text and messages) may use: the context window minus the answer's room."""
        return int((self.context_tokens - self.answer_tokens) * PROMPT_SHARE)

    def _system(self, conv_id: int, text: str) -> tuple[str, list[int]]:
        p = self._prompt(conv_id, text)
        return p.system, p.recalled

    def _prompt(self, conv_id: int, text: str) -> Prompt:
        """Gather every block, fit them to the one budget, and assemble the system prompt."""
        soup = self.a.soup
        conv = self.store.get(conv_id)
        scope = conv.scope if conv and conv.scope != "*" else None
        limit = self.prompt_limit
        # Stable blocks first (persona, rules, tools, workspace, the date): the model server reuses its work on
        # a prompt prefix that did not change, and everything recalled per message comes after.
        # The user's rules come right after the persona and in every prompt: recall by similarity would only
        # surface "never push" when the message happens to look like it. They are never cut here; the cap on
        # how many are gathered is a safety valve, not a budget.
        rules = standing_rules(soup, scope, budget_tokens=max(200, limit // 4))
        self._rules = rules
        blocks = [Block("persona", [PERSONA], fixed=True)]
        if rules:
            blocks.append(Block("rules", [render_rules(rules)], fixed=True))
        blocks.append(Block("tools", [tools_doc(list(self.tools().values()))], fixed=True))
        if self.workspace is not None:
            blocks.append(Block("workspace", [self.workspace.describe()], fixed=True))
        blocks.append(Block("date", [f"Today is {time.strftime('%A, %Y-%m-%d')}."], fixed=True))

        ids: dict[str, list[int]] = {}               # the memory behind each block's items, in item order
        profile = latest_profile(soup)
        if profile:
            blocks.append(Block("who", [ln for ln in profile.text.splitlines() if ln.strip()] or [profile.text],
                                "WHO THE USER IS (kgirl's consolidated profile):", PRIORITY["who"]))
            ids["who"] = [profile.id]
        prefs = soup.recall(text, budget_tokens=max(120, self.memory_budget // 2), kinds=("preference",), scope=scope,
                            exclude=tuple(f.id for f in rules))
        if prefs.fragments:
            blocks.append(Block("wants", [f"- {f.text}" for f in prefs.fragments],
                                "WHAT THE USER WANTS (their preferences):", PRIORITY["wants"]))
            ids["wants"] = prefs.ids
        mem = soup.recall(text, budget_tokens=self.memory_budget,
                          kinds=("fact", "abstraction", "antipattern", "skill", "episode", "note"),
                          scope=scope, exclude=tuple(([profile.id] if profile else []) + prefs.ids),
                          fallback=False)
        if mem.fragments:
            blocks.append(Block("memory", [f"- {f.render()}" for f in mem.fragments],
                                "MEMORY (recalled for this message; may be partial):", PRIORITY["memory"]))
            ids["memory"] = mem.ids
        earlier, related = self._earlier(conv_id, text, conv, scope)
        if earlier:
            blocks.append(Block("earlier", earlier, "EARLIER CONVERSATIONS (same topic, or matching turns):",
                                PRIORITY["earlier"]))
        named = self._named_code(text, scope)
        if named:
            blocks.append(Block("code", named.splitlines(),
                                "CODE THIS MESSAGE NAMES (from the Atlas index; read_file shows the source):",
                                PRIORITY["code"]))
        so_far = self._so_far(conv)
        if so_far:
            blocks.append(Block("so_far", so_far, "THIS CONVERSATION SO FAR:", PRIORITY["so_far"]))

        # The messages: this exchange is whole and never cut; older turns fill what the budget leaves.
        entries = self._entries(conv_id)
        now = [body for current, _, body in entries if current]
        older, used = [], 0
        for current, _, body in reversed(entries):       # newest first; at most the history budget
            cost = estimate_tokens(body)
            if not current:
                if used + cost > self.history_budget:
                    break
                older.append(body)
                used += cost
        blocks.append(Block("now", now, fixed=True))
        blocks.append(Block("history", older, priority=PRIORITY["history"], clip=False))
        blocks.append(Block("tool_room", [], fixed=True, reserve=self.tool_chars // 4))   # room for the biggest tool result
        fitted = fit(blocks, limit)
        self._history_cap = sum(estimate_tokens(b) for b in fitted.kept["history"])

        recalled: list[int] = []
        for name, owners in ids.items():                  # memory behind items that were cut was not shown
            recalled += owners[:len(fitted.kept[name])]    # (the profile is one id behind all its lines)
        kept_related = related[:len(fitted.kept.get("earlier", []))]
        parts = [fitted.text(n) for n in ("persona", "rules", "tools", "workspace", "date", "who", "wants", "memory",
                                          "earlier", "code", "so_far")]
        meta = {"budget": fitted.meta(), "related": kept_related}
        self._budgets[conv_id] = meta["budget"]
        return Prompt("\n\n".join(p for p in parts if p), recalled, meta)

    def _earlier(self, conv_id: int, text: str, conv, scope: str | None) -> tuple[list[str], list[dict]]:
        """Earlier conversations worth showing: first those on the same topic (tags written at reflection that this
        conversation or this message shares), then turns that match the message's words. Returns the lines and,
        for each, why it was picked (kept in the turn's meta)."""
        items: list[str] = []
        why: list[dict] = []
        want = clean_tags((conv.tags if conv else []) + tags_in(text, self.store.tag_counts()))
        for c, shared in self.store.related(want, exclude_conv=conv_id, scope=scope, limit=3):
            name = f"conv #{c.id}" + (f' "{one_line(c.title, 60)}"' if c.title else "")
            items.append(f"- {name} (same topic: {', '.join(shared)}): "
                         f"{one_line(c.summary or c.title or '(no summary yet)', 240)}")
            why.append({"conv": c.id, "via": "tags", "tags": shared})
        for t in self.store.search(text, limit=3, exclude_conv=conv_id):
            items.append(f"- conv #{t.conv_id} {t.role}: {one_line(t.content, 260)}")
            why.append({"conv": t.conv_id, "via": "words", "turn": t.id})
        return items, why

    def _so_far(self, conv) -> list[str]:
        """This conversation's state as prompt lines, most important first: the summary, what was promised and is
        still open, then open questions."""
        if conv is None:
            return []
        items = [conv.summary] if conv.summary else []
        owed = [f"- {c.label} ({c.who + ': ' if c.who else ''}{c.text})" for c in conv.open_commitments]
        if owed:
            items += [f"OPEN COMMITMENTS (not done yet; keep them in mind):\n{owed[0]}", *owed[1:]]
        if conv.threads:
            items += [f"OPEN THREADS:\n- {conv.threads[0]}", *(f"- {t}" for t in conv.threads[1:])]
        return items

    def context_report(self, conv_id: int) -> str:
        """What the latest prompt of this conversation held: each block's size and what the budget cut."""
        meta = self._budgets.get(conv_id)
        if meta is None:
            last = self.store.last_answer(conv_id)
            meta = last.meta.get("budget") if last else None
        return render_report(meta or {}, self.context_tokens, self.answer_tokens)

    def doctor(self) -> str:
        """The health checks (kgirl doctor), one line each, run in-process and read-only. Says so when this
        build has no doctor."""
        try:
            from ..doctor import collect
        except ImportError:
            return "kgirl doctor is not available in this build"
        checks = collect(self.a.home)
        lines = [f"{c.status:<4} {c.name}: {c.detail}" + (f" - fix: {c.fix}" if c.fix else "") for c in checks]
        bad = sum(c.status == "FAIL" for c in checks)
        return "\n".join(lines + [f"{len(checks)} checks, {bad} failing" if bad else f"{len(checks)} checks, none failing"])

    def close_commitment(self, conv_id: int, which: str) -> str:
        """The user says a commitment is done (/done C2, or a piece of its text)."""
        done = self.store.close_commitments(conv_id, [which], by="user")
        if done:
            return "closed: " + "; ".join(f"{c.label} {c.text}" for c in done)
        conv = self.store.get(conv_id)
        left = ", ".join(c.label for c in conv.open_commitments) if conv else ""
        return f"no open commitment matches {which!r}" + (f" (open: {left})" if left else " (none are open)")

    def _named_code(self, text: str, repo: str | None, limit: int = 8) -> str:
        """Index lines for the symbols and files a message names exactly. Exact matches only: this block is
        there so the model need not guess where something lives, not to pad the prompt with look-alikes."""
        atlas = self.a.atlas
        names = list(dict.fromkeys(next(g for g in m.groups() if g).strip() for m in _NAMED.finditer(text)))[:10]
        lines: list[str] = []
        for name in names:
            if len(lines) >= limit:
                break
            found: list[str] = []
            if "/" in name or "." in name:            # a path or a file name (a dotted symbol falls through)
                sql = ("SELECT r.name AS repo, f.path FROM files f JOIN repos r ON r.id=f.repo_id"
                       " WHERE (f.path=? OR f.path LIKE ?)" + (" AND r.name=?" if repo else "") + " LIMIT 3")
                args = (name, "%/" + name, repo) if repo else (name, "%/" + name)
                found = [f"- {r['repo']}:{r['path']} [file]" for r in atlas.db.execute(sql, args)]
            if not found:
                rows = atlas.find_symbol(name, repo) or (atlas.find_symbol(name) if repo else [])
                found = [f"- {r['repo']}:{r['path']}:{r['line']} [{r['kind']}] {clip(r['signature'] or r['qualname'], 110)}"
                         for r in rows[:3]]
            lines += [ln for ln in found if ln not in lines]
        return "\n".join(lines[:limit])

    def _entries(self, conv_id: int) -> list[tuple[bool, str, str]]:
        """The recent turns as (in the current exchange, role, body), oldest first. Tool results shrink as they
        age: full for the newest one of this exchange, a third for its earlier ones, a stub once it is over."""
        turns = self.store.window(conv_id, self.history_budget * 4)
        last_user = max((t.id for t in turns if t.role == "user"), default=0)
        newest_tool = max((t.id for t in turns if t.role == "tool"), default=0)
        out: list[tuple[bool, str, str]] = []
        for t in turns:
            if t.role == "tool":
                cap = OLD_TOOL_CHARS if t.id < last_user else (self.tool_chars if t.id == newest_tool
                                                               else self.tool_chars // 3)
                role, body = "user", "TOOL RESULT " + _clip_result(t.content, cap)
            else:
                role, body = t.role, t.content
            out.append((t.id >= last_user, role, body))
        return out

    def _messages(self, conv_id: int) -> list[Message]:
        """The recent turns as chat messages. The current exchange is always whole; older turns fill the room the
        prompt budget left for them (the history budget when no prompt has been built), newest first."""
        cap = self.history_budget if self._history_cap is None else self._history_cap
        picked: list[Message] = []
        used = 0
        for current, role, body in reversed(self._entries(conv_id)):
            cost = estimate_tokens(body)
            if picked and not current and used + cost > cap:
                break
            picked.append(Message(role, body))
            if not current:
                used += cost
        picked.reverse()
        # Ollama and Anthropic want alternation that starts with the user: merge neighbours of the same role.
        merged: list[Message] = []
        for m in picked:
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
        for piece in self.a.router.stream(CHAT, system, msgs, max_tokens=self.answer_tokens, temperature=0.4,
                                          stop=STOP):
            buf += piece
            if call_complete(buf):
                break           # a backend that ignored the stop sequence: don't let it write the tool's result
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

    def tools(self) -> dict[str, Tool]:
        """The tools this session offers, in the order they are described to the model."""
        ws = self.workspace
        verifiers = ", ".join(shlex.join(c) for c in self._verifiers()) or "none yet: the user sets one with /verify"
        tools = [Tool("search_code", '{"query": "...", "repo": "optional"}',
                      "find symbols and code across the indexed repos", self._t_search_code)]
        if ws is not None:
            runnable = ", ".join(shlex.join(c) for c in ws.allowed_commands()) or "none yet"
            tools += [
                Tool("read_file", '{"path": "...", "start": 1, "end": 120}',
                     'read numbered lines of a workspace file (add "repo" to read a file of another indexed repo)',
                     self._t_read_file),
                Tool("ls", '{"path": ".", "depth": 1}', "list a workspace directory", self._t_ls),
                Tool("grep", '{"pattern": "regex", "path": "."}', "search file contents in the workspace", self._t_grep),
                Tool("edit", '{"path": "...", "search": "exact lines from the file", "replace": "new lines"}',
                     "change a workspace file; search must match exactly once", self._t_edit),
                Tool("write_file", '{"path": "...", "content": "..."}', "create a workspace file, or replace one whole",
                     self._t_write_file),
                Tool("run", '{"command": "..."}',
                     f"run a command in the workspace, no shell (allowed: {runnable}; others need the user's OK)",
                     self._t_run)]
        else:
            tools.append(Tool("read_file", '{"repo": "...", "path": "...", "start": 1, "end": 120}',
                              "read numbered lines of an indexed file", self._t_read_file))
        tools += [
            Tool("task", '{"goal": "...", "verify": "command"}',
                 f"have sandboxed agents make a code change and prove it with the verifier ({verifiers}); slow; "
                 "returns a diff the user applies with /apply", self._t_task),
            Tool("recall", '{"query": "..."}', "search long-term memory and earlier conversations", self._t_recall),
            Tool("remember", '{"text": "...", "kind": "fact|preference|abstraction"}', "propose something worth keeping",
                 self._t_remember),
            Tool("build", '{"spec": "...", "edit": "optional workspace file to change", "tests": ["optional test files"]}',
                 "turn a precise spec into a tested Python module, or with edit into a diff of that file for the "
                 "user to review (slow: minutes)", self._t_build),
            Tool("write", '{"brief": "...", "words": 800}', "plan, draft and revise a long piece of text (slow)",
                 self._t_write),
            Tool("repos", "{}", "list the indexed repositories", self._t_repos)]
        return {t.name: t for t in tools}

    def _run_tool(self, name: str, args: dict) -> str:
        tool = self.tools().get(name)
        if tool is None:
            return f"unknown tool {name!r}; the tools are: {', '.join(self.tools())}"
        try:
            return tool.run(args) or "(no output)"
        except WorkspaceError as exc:
            return f"{name} refused: {exc}"
        except KeyError as exc:
            return f"{name} failed: missing argument {exc}; use {tool.usage}"
        except Exception as exc:  # noqa: BLE001 - a failing tool is a result for the model, not a crash
            return f"{name} failed: {type(exc).__name__}: {exc}"

    def call_tool(self, name: str, args: dict, on_event: Callable[[str], None] | None = None) -> str:
        """Run one tool on the user's behalf (slash commands use this)."""
        if on_event is not None:
            self._say = on_event
        return self._run_tool(name, args)

    def _t_search_code(self, args: dict) -> str:
        ctx, _ = self.a.context_pack(str(args["query"]), args.get("repo") or None, budget=2200)
        return ctx or "no matches; the repo may not be indexed"

    def _t_read_file(self, args: dict) -> str:
        path, start, end = str(args["path"]), _int(args.get("start")) or 1, _int(args.get("end"))
        repo = args.get("repo")
        if not repo:
            if self.workspace is None:
                raise ValueError('no workspace is attached, so read_file needs "repo": an indexed repository')
            return self.workspace.read(path, start, end)
        text = self.a.atlas.read_source(str(repo), path, start, end or start + 120)
        why = secret_reason(path, text)
        if why:
            raise ValueError(f"{repo}:{path} looks like it holds a secret ({why}); kgirl keeps it out of the conversation")
        return text

    def _t_ls(self, args: dict) -> str:
        return self.workspace.ls(str(args.get("path") or "."), _int(args.get("depth")) or 1)

    def _t_grep(self, args: dict) -> str:
        return self.workspace.grep(str(args["pattern"]), str(args.get("path") or "."))

    def _t_edit(self, args: dict) -> str:
        return self.workspace.edit(str(args["path"]), str(args["search"]), str(args.get("replace", "")))

    def _t_write_file(self, args: dict) -> str:
        return self.workspace.write(str(args["path"]), str(args["content"]))

    def _t_run(self, args: dict) -> str:
        return self.workspace.run(str(args.get("command") or args.get("cmd") or ""))

    def _t_recall(self, args: dict) -> str:
        q = str(args.get("query", ""))
        rec = self.a.soup.recall(q, budget_tokens=700, fallback=False)
        old = self.store.search(q, limit=6)
        topical = self.store.related(tags_in(q, self.store.tag_counts()), exclude_conv=self._conv, limit=3)
        out = ((rec.render() or "(no stored memory matches)") + "\n\nconversation turns:\n" +
               ("\n".join(f"- conv #{t.conv_id} {t.role}: {one_line(t.content, 300)}" for t in old) or "(none)"))
        if topical:
            out += "\n\nearlier conversations on this topic:\n" + "\n".join(
                f"- conv #{c.id} {c.title or '(untitled)'} (tags: {', '.join(shared)}): {one_line(c.summary, 240)}"
                for c, shared in topical)
        return out

    def _t_remember(self, args: dict) -> str:
        kind = args.get("kind", "fact")
        kind = kind if kind in ("fact", "preference", "abstraction") else "fact"
        text = one_line(str(args["text"]), 300)
        if not text:
            raise ValueError("remember needs text")
        fid, how = self.a.curator.propose(text, kind=kind, tags="conversation", source="reflection:chat")
        return f"{how} {kind} #{fid} (becomes active once it is observed again or the user promotes it)"

    def _t_repos(self, args: dict) -> str:
        return "\n".join(f"{r['name']} ({r['nfiles']} files)" for r in self.a.atlas.repos()) or "no repos indexed"

    def _t_build(self, args: dict) -> str:
        spec = str(args["spec"]).strip()
        if not spec:
            raise ValueError("build needs a spec")
        if args.get("edit"):
            return self._build_edit(spec, str(args["edit"]), args.get("tests"))
        self._say("building: tests first, several candidates, agreement selection (takes a few minutes)")
        res = self.builder().build(spec)
        self.last_build = res
        head = res.summary() if hasattr(res, "summary") else res.render()      # builder v2's short chat summary
        return f"{head}\n\n```python\n{res.code}```" if res.code else head

    def _build_edit(self, spec: str, path: str, tests) -> str:
        """Edit an existing workspace file from a spec. The builder samples diffs and runs the file's own tests plus the
        spec's; the winner comes back as a diff for review and is never applied here (/apply does it, with approval)."""
        ws = self.workspace
        if ws is None:
            raise ValueError("edit needs a project directory: no workspace is attached")
        target = ws.resolve(path)
        if not target.is_file():
            raise ValueError(f"{path} is not a file in the workspace")
        names = [tests] if isinstance(tests, str) else [t for t in (tests or []) if isinstance(t, str)]
        test_files = [ws.resolve(t) for t in names]
        original = target.read_text(encoding="utf-8", errors="replace")
        self._say(f"editing {ws.rel(target)}: diffs sampled, existing and spec tests run (takes a few minutes)")
        res = self._edit_build(spec, target, test_files)
        self.last_build = res
        self.last_edit = None
        head = res.summary() if hasattr(res, "summary") else res.render()
        if not (res.code and getattr(res, "diff", "")):
            return head
        self.last_edit = (res, ws.rel(target), original)
        return (f"{head}\n\n```diff\n{res.diff.rstrip()}\n```\n"
                "This diff is NOT applied. Tell the user what it changes; they apply it with /apply "
                "(it asks first, and /undo takes it back).")

    def _edit_build(self, spec: str, target: Path, tests: list[Path]):
        """The builder's edit mode: its chat entry point `edit_build` when it has one, else `Builder.edit`."""
        try:
            from .builder import edit_build
        except ImportError:
            edit_build = None
        if edit_build is not None:
            scope = self.scope()
            return edit_build(spec, target, tests or None, builder=self.builder(), atlas=self.a.atlas,
                              repo=None if scope == "*" else scope, scope=scope)
        builder = self.builder()
        if not hasattr(builder, "edit"):
            raise ValueError("this build of kgirl has no edit mode for the build tool")
        return builder.edit(spec, target, tests=tests, scope=self.scope())

    def _t_write(self, args: dict) -> str:
        brief = str(args["brief"]).strip()
        if not brief:
            raise ValueError("write needs a brief")
        self._say("writing: outline, draft by section, critique, revise")
        conv = self._conv if self._conv is not None else self.resume()
        draft, art = self.write(conv, brief, _int(args.get("words")), progress=self._say)
        return f"{draft.markdown()}\n{DRAFT_SAVED.format(id=art)}"

    def write(self, conv_id: int, brief: str = "", words: int | None = None, revise: bool = True,
              resume: int | None = None, progress: Callable[[str], None] | None = None) -> tuple[Draft, int]:
        """Write a long piece in this conversation, or continue draft `resume`. Every step is saved as a
        conversation artifact (kind "draft"), so an interrupted piece is not lost. Returns (draft, artifact id)."""
        saved = None
        if resume is not None:
            art = self.store.artifact(resume)
            if art is None or art["kind"] != "draft":
                raise ValueError(f"there is no draft #{resume} (see `kgirl write --list`)")
            conv_id, saved, art_id = art["conv_id"], Draft.from_dict(art["data"]), resume
        else:
            art_id = None
        context, style, ids = ("", "", []) if saved else self._writing_context(brief)

        def keep(d: Draft) -> None:
            nonlocal art_id
            art_id = self.store.save_artifact(conv_id, "draft", d.title or one_line(d.brief, 60), d.to_dict(), art_id)
            self.last_draft = art_id

        writer = Writer(self._llm_text, emit=lambda topic, payload: self.a.bus.publish(topic, payload, "writer"))
        draft = writer.write(brief, context, words, revise, progress or self._say, style=style, resume=saved,
                             checkpoint=keep)
        if not saved:
            draft.recalled = ids
            keep(draft)
        return draft, art_id

    def _writing_context(self, brief: str) -> tuple[str, str, list[int]]:
        """What the writer is told besides the brief: facts and episodes to use (CONTEXT) and the user's voice
        (STYLE: their profile from evolve and the preferences that fit this brief). Also the memory ids used."""
        soup = self.a.soup
        scope = self.scope()
        scope = None if scope == "*" else scope
        context = soup.recall(brief, budget_tokens=500, kinds=("fact", "episode"), scope=scope,
                              fallback=False).render()
        prefs = soup.recall(brief, budget_tokens=350, kinds=("preference",), scope=scope)
        profile = latest_profile(soup)
        style = style_guide(profile.text if profile else "", [f.text for f in prefs.fragments])
        return context, style, ([profile.id] if profile else []) + prefs.ids

    def drafts(self, conv_id: int | None = None) -> list[tuple[int, int, str, str]]:
        """Saved drafts, newest first, as (id, conversation, title, where it stands)."""
        return [(a["id"], a["conv_id"], a["title"], Draft.from_dict(a["data"]).progress())
                for a in self.store.artifacts(conv_id, "draft")]

    def _t_task(self, args: dict) -> str:
        goal = str(args["goal"]).strip()
        if not goal:
            raise ValueError("task needs a goal")
        root, name = self._task_repo(args.get("repo"))
        verify = self._verifier(args.get("verify"))
        self._say(f"swarm: sandboxed agents work on {name}, verified by `{shlex.join(verify)}` (takes minutes)")
        res = self.a.task(goal, root, verify, repo_name=name, origin="chat")
        self.last_task = (res, Path(root)) if res.winner is not None else None
        if res.winner is None:
            return res.render() + "\nNo agent produced a change that passes the verifier."
        return res.render() + "\nThis diff is NOT applied yet. Tell the user what it changes; they apply it with /apply."

    def builder(self, **kw) -> Builder:
        return Builder(self._code_llm, self.a.soup,
                       emit=lambda topic, payload: self.a.bus.publish(topic, payload, "builder"), **kw)

    # ------------------------------------------------------------------ swarm tasks

    def _verifiers(self) -> list[list[str]]:
        """Verifier commands a model may name: the user's session verifier, then the allowlist."""
        allowed = self.workspace.allowed_commands() if self.workspace is not None else allowlisted_commands()
        out = [self.verify_cmd] if self.verify_cmd else []
        return out + [c for c in allowed if c not in out]

    def _verifier(self, requested) -> list[str]:
        choices = self._verifiers()
        listed = ", ".join(f"`{shlex.join(c)}`" for c in choices)
        if requested:
            argv = shlex.split(str(requested))
            if argv in choices:
                return argv
            raise ValueError(f"verify must be exactly one of: {listed}" if choices else
                             "no verifier is allowed yet; ask the user to set one with /verify <command>")
        if self.verify_cmd:
            return self.verify_cmd
        raise ValueError(f"task needs \"verify\": one of {listed}" if choices else
                         "task needs a verifier, and none is allowed yet; ask the user to set one with /verify <command>")

    def _task_repo(self, repo) -> tuple[str, str]:
        """(root, name) of the repo a swarm task works on: a named indexed repo, else the workspace."""
        if repo:
            for r in self.a.atlas.repos():
                if r["name"].lower() == str(repo).lower():
                    return r["root"], r["name"]
            raise ValueError(f"{repo!r} is not an indexed repo (see the repos tool)")
        if self.workspace is None:
            raise ValueError('no workspace is attached, so task needs "repo": an indexed repository')
        return str(self.workspace.root), self.scope() if self.scope() != "*" else self.workspace.name

    def apply_task(self) -> str:
        """Write what is waiting for the user's /apply: the last swarm's verified diff, and the last edit-mode build.
        What a swarm run taught is promoted, because a person has now accepted it."""
        if self.last_task is None and self.last_edit is None:
            return "there is no verified swarm diff waiting"
        lines = []
        if self.last_task is not None:
            lines.append(self._apply_swarm())
        if self.last_edit is not None:
            lines.append(self._apply_edit())
        return "\n".join(lines)

    def _apply_swarm(self) -> str:
        res, root = self.last_task
        applied, conflicts = apply_edits(res.winner.edited, root)
        if applied:
            if self.workspace is not None and self.workspace.root == Path(root).resolve():
                for path in applied:
                    self.workspace.record(path, *res.winner.edited[path])
            for fid, how in res.proposals:
                if how == "staged trajectory":
                    self.a.soup.set_status(fid, "active", "its diff was applied by the user")
                    self.a.soup.credit([fid], success=True)
            self.last_task = None
        lines = [f"applied to {root}: {', '.join(applied)}"] if applied else ["nothing applied"]
        if conflicts:
            lines.append("NOT applied (the file changed since the agents copied it): " + ", ".join(conflicts))
        return "\n".join(lines)

    def _apply_edit(self) -> str:
        """Write the edit-mode result through the workspace, so the user approves it and /undo can take it back.
        Refused when the file is no longer what the builder started from."""
        res, rel, original = self.last_edit
        ws = self.workspace
        if ws is None:
            self.last_edit = None
            return f"edit of {rel} not applied: no workspace is attached"
        try:
            now = ws.resolve(rel).read_text(encoding="utf-8", errors="replace")
            if now != original:
                self.last_edit = None
                return f"edit of {rel} NOT applied: the file changed since the builder read it; build the edit again"
            out = ws.write(rel, res.code)
        except WorkspaceError as exc:
            return f"edit of {rel} not applied: {exc}"          # (declined: it stays waiting for another /apply)
        self.last_edit = None
        return out

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
        lines = [f"conversation #{conv.id}: {conv.title or '(untitled)'}" + (f"  [{conv.scope}]" if conv.scope != "*" else ""),
                 f"summary: {conv.summary or '(not reflected yet)'}"]
        if conv.tags:
            lines.append("topics: " + ", ".join(conv.tags))
        lines += [f"open: {t}" for t in conv.threads]
        lines += [f"{c.label} [{'open' if c.open else 'done'}] {c.who + ': ' if c.who else ''}{c.text}"
                  for c in conv.commitments]
        lines.append(f"turns: {self.store.count(conv.id)}")
        return "\n".join(lines)


def latest_profile(soup):
    """The newest active profile note written by `evolve` (None before the first consolidation)."""
    row = soup.db.execute("SELECT id FROM fragments WHERE kind='note' AND tags='profile' AND status='active'"
                          " ORDER BY id DESC LIMIT 1").fetchone()
    return soup.get(row["id"]) if row else None


def _clip_result(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return f"{text[:cap]}\n…[{len(text) - cap} more characters not shown; call the tool again with a narrower range]"


def _int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
