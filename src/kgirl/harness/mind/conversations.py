"""Conversation memory: every turn kgirl has, stored, searchable and summarized.

    conversations  one row per thread: title, rolling summary, open threads, scope, timestamps
    turns          every message in order, with the memory ids it was answered from and a small `meta` record
    turn_fts       full-text index over what was said, so a new conversation can recall old ones
    contexts       the system prompt each answer was generated from (for a faithful fine-tune export)
    artifacts      work products of a conversation (writer drafts), so a long piece can be resumed

Beyond the rolling summary, reflection keeps two more kinds of state per conversation: open `threads`
(questions and unfinished tasks) and `commitments` ("I'll ...", "next time ...", from either side), each
with an owner and a status, closed once the outcome is stated or the user says so. `tags` name what the
conversation was about; `related()` uses them to find earlier conversations on the same topic.

Roles, in the order one exchange stores them:

    user        what the person typed
    assistant   meta.partial = true: text the model wrote before a tool call, and the call itself
    tool        the tool's result
    assistant   the final answer; `recalled` lists the memory it was built from, and meta.feedback records
                the person's verdict once there is one

The rolling summary is what lets a small model "understand" a long conversation: the prompt carries the
summary plus the most recent turns that fit a token budget, instead of the whole transcript.
Lives beside soup.db in the harness home (private, 0700; see docs/PRIVACY.md).
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..util import estimate_tokens, fts_query, harness_home, now, search_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY, title TEXT DEFAULT '', summary TEXT DEFAULT '', threads TEXT DEFAULT '[]',
  scope TEXT DEFAULT '*', started REAL, updated REAL, reflected_turn INTEGER DEFAULT 0,
  commitments TEXT DEFAULT '[]', tags TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY, conv_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
  ts REAL, recalled TEXT DEFAULT '[]', meta TEXT DEFAULT '{}');
CREATE INDEX IF NOT EXISTS ix_turns_conv ON turns(conv_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(terms, content, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS contexts (turn_id INTEGER PRIMARY KEY, system TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY, conv_id INTEGER NOT NULL, kind TEXT NOT NULL, title TEXT DEFAULT '',
  data TEXT NOT NULL, created REAL, updated REAL);
CREATE INDEX IF NOT EXISTS ix_artifacts_conv ON artifacts(conv_id, kind, id);
"""
MIGRATIONS = (("conversations", "commitments", "TEXT DEFAULT '[]'"), ("conversations", "tags", "TEXT DEFAULT '[]'"))

MAX_COMMITMENTS = 20       # open ones first; this many closed ones are kept for the record
MAX_TAGS = 8
RELATED_SCAN = 500         # `related` looks at the most recently used conversations
_TAG = re.compile(r"[a-z0-9][a-z0-9_-]{1,23}")

ROLES = ("user", "assistant", "tool", "system")


@dataclass(frozen=True)
class Turn:
    id: int
    conv_id: int
    role: str
    content: str
    ts: float
    recalled: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    @property
    def partial(self) -> bool:
        """An assistant turn that ended in a tool call, not the answer."""
        return bool(self.meta.get("partial"))

    @property
    def is_answer(self) -> bool:
        return self.role == "assistant" and not self.partial

    @property
    def verdict(self) -> str:
        """The person's verdict on this answer: "positive", "negative" or "" when there is none yet."""
        fb = self.meta.get("feedback")
        return fb.get("verdict", "") if isinstance(fb, dict) else ""


@dataclass(frozen=True)
class Exchange:
    """One user message and everything kgirl did about it."""
    user: Turn
    steps: tuple[Turn, ...]          # partial assistant turns and tool results, in order
    answer: Turn | None              # None while unanswered (or when the answer was interrupted)


@dataclass(frozen=True)
class Commitment:
    """Something said would be done: by kgirl ("I'll check that") or by the user ("next time I will ...")."""
    id: int                    # per conversation; shown as C<id>
    text: str
    who: str = ""              # "kgirl", "user", or "" when unclear
    status: str = "open"       # "open" | "done"
    turn: int = 0              # the turn the reflection that found it ended at
    closed_by: str = ""        # once done: "reflection" (an outcome was stated or the user said so) or "user" (/done)

    @property
    def label(self) -> str:
        return f"C{self.id}"

    @property
    def open(self) -> bool:
        return self.status == "open"

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "who": self.who, "status": self.status, "turn": self.turn,
                "closed_by": self.closed_by}

    @classmethod
    def from_dict(cls, d) -> "Commitment | None":
        try:
            text = " ".join(str(d["text"]).split())
            if not text:
                return None
            return cls(int(d["id"]), text, str(d.get("who", "")), "done" if d.get("status") == "done" else "open",
                       int(d.get("turn") or 0), str(d.get("closed_by", "")))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    summary: str
    threads: list[str]
    scope: str
    started: float
    updated: float
    reflected_turn: int
    commitments: list[Commitment] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def open_commitments(self) -> list[Commitment]:
        return [c for c in self.commitments if c.open]


class ConversationStore:
    def __init__(self, db_path: str | os.PathLike | None = None):
        path = Path(db_path) if db_path else harness_home() / "conversations.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        for table, col, decl in MIGRATIONS:       # a database made before this column existed
            if col not in {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                with self.db:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ threads

    def new(self, title: str = "", scope: str = "*") -> int:
        t = now()
        with self._lock, self.db:
            return self.db.execute("INSERT INTO conversations(title, scope, started, updated) VALUES (?,?,?,?)",
                                   (title, scope or "*", t, t)).lastrowid

    def get(self, conv_id: int) -> Conversation | None:
        r = self.db.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        return _conv(r) if r else None

    def latest(self, scope: str | None = None) -> Conversation | None:
        """The most recently used conversation, optionally only among those of one scope."""
        if scope is None:
            r = self.db.execute("SELECT * FROM conversations ORDER BY updated DESC, id DESC LIMIT 1").fetchone()
        else:
            r = self.db.execute("SELECT * FROM conversations WHERE scope=? ORDER BY updated DESC, id DESC LIMIT 1",
                                (scope,)).fetchone()
        return _conv(r) if r else None

    def list(self, limit: int = 20) -> list[Conversation]:
        return [_conv(r) for r in self.db.execute(
            "SELECT * FROM conversations ORDER BY updated DESC, id DESC LIMIT ?", (limit,))]

    def update(self, conv_id: int, *, title: str | None = None, summary: str | None = None,
               threads: list[str] | None = None, reflected_turn: int | None = None,
               scope: str | None = None, tags: list[str] | None = None) -> None:
        sets, args = [], []
        for col, val in (("title", title), ("summary", summary), ("reflected_turn", reflected_turn), ("scope", scope)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        if threads is not None:
            sets.append("threads=?")
            args.append(json.dumps(threads[:12], ensure_ascii=False))
        if tags is not None:
            sets.append("tags=?")
            args.append(json.dumps(clean_tags(tags), ensure_ascii=False))
        if not sets:
            return
        with self._lock, self.db:
            self.db.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id=?", (*args, conv_id))

    def delete(self, conv_id: int) -> int:
        """Remove a conversation and everything stored about its turns. Returns how many turns it had."""
        with self._lock, self.db:
            ids = [r["id"] for r in self.db.execute("SELECT id FROM turns WHERE conv_id=?", (conv_id,))]
            self.db.executemany("DELETE FROM turn_fts WHERE rowid=?", [(i,) for i in ids])
            self.db.executemany("DELETE FROM contexts WHERE turn_id=?", [(i,) for i in ids])
            self.db.execute("DELETE FROM turns WHERE conv_id=?", (conv_id,))
            self.db.execute("DELETE FROM artifacts WHERE conv_id=?", (conv_id,))
            self.db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        return len(ids)

    # ------------------------------------------------------------------ commitments

    def add_commitments(self, conv_id: int, items: list[tuple[str, str]], turn: int = 0) -> list[Commitment]:
        """Record new commitments as (text, who). One that restates an open commitment is not added twice.
        Returns the ones that were added."""
        added: list[Commitment] = []
        with self._lock, self.db:
            have = self._commitments(conv_id)
            if have is None:
                return added
            nxt = max((c.id for c in have), default=0) + 1
            for text, who in items:
                text = " ".join(str(text).split())[:240]
                if not text or any(c.open and same_commitment(c.text, text) for c in have + added):
                    continue
                added.append(Commitment(nxt, text, who if who in ("kgirl", "user") else "", "open", turn))
                nxt += 1
            if added:
                self._save_commitments(conv_id, have + added)
        return added

    def close_commitments(self, conv_id: int, which: list[int | str], by: str = "reflection") -> list[Commitment]:
        """Mark open commitments done. `which` holds ids (3, "C3", "3") or a distinctive piece of the text.
        Returns the ones that were closed."""
        closed: list[Commitment] = []
        with self._lock, self.db:
            have = self._commitments(conv_id)
            if not have:
                return closed
            out = []
            for c in have:
                hit = c.open and any(_names(c, w) for w in which)
                if hit:
                    c = Commitment(c.id, c.text, c.who, "done", c.turn, by)
                    closed.append(c)
                out.append(c)
            if closed:
                self._save_commitments(conv_id, out)
        return closed

    def _commitments(self, conv_id: int) -> list[Commitment] | None:
        r = self.db.execute("SELECT commitments FROM conversations WHERE id=?", (conv_id,)).fetchone()
        return None if r is None else _commitments(r["commitments"])

    def _save_commitments(self, conv_id: int, items: list[Commitment]) -> None:
        opened = [c for c in items if c.open][-MAX_COMMITMENTS:]
        done = [c for c in items if not c.open][-MAX_COMMITMENTS:]
        keep = sorted(opened + done, key=lambda c: c.id)
        self.db.execute("UPDATE conversations SET commitments=? WHERE id=?",
                        (json.dumps([c.to_dict() for c in keep], ensure_ascii=False), conv_id))

    # ------------------------------------------------------------------ related conversations

    def tag_counts(self) -> dict[str, int]:
        """How many of the recent conversations carry each tag."""
        counts: dict[str, int] = {}
        for r in self.db.execute("SELECT tags FROM conversations ORDER BY updated DESC, id DESC LIMIT ?",
                                 (RELATED_SCAN,)):
            for tag in _tags(r["tags"]):
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def related(self, tags: list[str], exclude_conv: int | None = None, scope: str | None = None,
                limit: int = 3) -> list[tuple[Conversation, list[str]]]:
        """Earlier conversations on the same topic: those sharing tags with `tags`, best first, with the shared tags.

        A shared tag counts for more the fewer conversations carry it (a tag everything has says nothing).
        Ties go to the more recently used conversation. With `scope`, only conversations of that scope or of
        no scope (`*`) qualify, like recalled memory."""
        want = set(clean_tags(tags))
        if not want:
            return []
        rows = self.db.execute("SELECT * FROM conversations ORDER BY updated DESC, id DESC LIMIT ?",
                               (RELATED_SCAN,)).fetchall()
        df: dict[str, int] = {}
        for r in rows:
            for tag in _tags(r["tags"]):
                df[tag] = df.get(tag, 0) + 1
        scored: list[tuple[float, int, Conversation, list[str]]] = []
        for rank, r in enumerate(rows):
            if r["id"] == exclude_conv or (scope is not None and (r["scope"] or "*") not in ("*", scope)):
                continue
            shared = [t for t in _tags(r["tags"]) if t in want]
            if shared:
                weight = sum(math.log(1 + len(rows) / df[t]) for t in shared)
                scored.append((weight, -rank, _conv(r), shared))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [(c, shared) for _, _, c, shared in scored[:limit]]

    # ------------------------------------------------------------------ artifacts

    def save_artifact(self, conv_id: int, kind: str, title: str, data: dict, artifact_id: int | None = None) -> int:
        """Keep a work product of a conversation (a writer draft); with `artifact_id`, replace that one.
        Saving one does not count as conversation activity: it never changes which conversation chat resumes."""
        t = now()
        blob = json.dumps(data, ensure_ascii=False)
        with self._lock, self.db:
            if artifact_id is not None and self.db.execute("SELECT 1 FROM artifacts WHERE id=?",
                                                           (artifact_id,)).fetchone():
                self.db.execute("UPDATE artifacts SET title=?, data=?, updated=? WHERE id=?",
                                (title, blob, t, artifact_id))
                return artifact_id
            return self.db.execute("INSERT INTO artifacts(conv_id, kind, title, data, created, updated)"
                                   " VALUES (?,?,?,?,?,?)", (conv_id, kind, title, blob, t, t)).lastrowid

    def artifact(self, artifact_id: int) -> dict | None:
        """{"id", "conv_id", "kind", "title", "data", "created", "updated"}, or None."""
        r = self.db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return _artifact(r) if r else None

    def artifacts(self, conv_id: int | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT * FROM artifacts WHERE 1=1", []
        if conv_id is not None:
            sql, args = sql + " AND conv_id=?", args + [conv_id]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [_artifact(r) for r in self.db.execute(sql + " ORDER BY updated DESC, id DESC LIMIT ?", (*args, limit))]

    # ------------------------------------------------------------------ turns

    def append(self, conv_id: int, role: str, content: str, recalled: list[int] | None = None,
               meta: dict | None = None, searchable: bool | None = None) -> int:
        """Store one turn. By default what the user said and what kgirl finally answered are searchable;
        tool results and the half-turns that end in a tool call are not."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        meta = meta or {}
        if searchable is None:
            searchable = role in ("user", "assistant") and not meta.get("partial")
        t = now()
        with self._lock, self.db:
            tid = self.db.execute("INSERT INTO turns(conv_id, role, content, ts, recalled, meta) VALUES (?,?,?,?,?,?)",
                                  (conv_id, role, content, t, json.dumps(recalled or []),
                                   json.dumps(meta, ensure_ascii=False))).lastrowid
            if searchable:
                self.db.execute("INSERT INTO turn_fts(rowid, terms, content) VALUES (?,?,?)",
                                (tid, search_terms(content), content))
            self.db.execute("UPDATE conversations SET updated=? WHERE id=?", (t, conv_id))
        return tid

    def turn(self, turn_id: int) -> Turn | None:
        r = self.db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
        return _turn(r) if r else None

    def set_meta(self, turn_id: int, **updates) -> None:
        """Merge keys into a turn's meta record (for example the verdict on an answer)."""
        with self._lock, self.db:
            row = self.db.execute("SELECT meta FROM turns WHERE id=?", (turn_id,)).fetchone()
            if row is None:
                return
            try:
                meta = json.loads(row["meta"] or "{}")
            except ValueError:
                meta = {}
            meta.update(updates)
            self.db.execute("UPDATE turns SET meta=? WHERE id=?", (json.dumps(meta, ensure_ascii=False), turn_id))

    def turns(self, conv_id: int, since_id: int = 0, roles: tuple[str, ...] | None = None) -> list[Turn]:
        sql, args = "SELECT * FROM turns WHERE conv_id=? AND id>?", [conv_id, since_id]
        if roles:
            sql += f" AND role IN ({','.join('?' * len(roles))})"
            args.extend(roles)
        return [_turn(r) for r in self.db.execute(sql + " ORDER BY id", args)]

    def last(self, conv_id: int, role: str | None = None) -> Turn | None:
        sql, args = "SELECT * FROM turns WHERE conv_id=?", [conv_id]
        if role:
            sql, args = sql + " AND role=?", args + [role]
        r = self.db.execute(sql + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return _turn(r) if r else None

    def last_answer(self, conv_id: int, before_id: int | None = None) -> Turn | None:
        """The newest final answer in a conversation (optionally the newest one stored before `before_id`)."""
        sql, args = "SELECT * FROM turns WHERE conv_id=? AND role='assistant'", [conv_id]
        if before_id is not None:
            sql, args = sql + " AND id<?", args + [before_id]
        for r in self.db.execute(sql + " ORDER BY id DESC LIMIT 12", args):
            t = _turn(r)
            if not t.partial:
                return t
        return None

    def exchanges(self, conv_id: int, since_id: int = 0) -> list[Exchange]:
        """The conversation as exchanges: each user message with its tool steps and its final answer."""
        out: list[Exchange] = []
        user: Turn | None = None
        steps: list[Turn] = []
        for t in self.turns(conv_id, since_id):
            if t.role == "user":
                if user is not None:
                    out.append(Exchange(user, tuple(steps), None))
                user, steps = t, []
            elif user is None:
                continue                      # output of a slash command, or the tail of an older exchange
            elif t.is_answer:
                out.append(Exchange(user, tuple(steps), t))
                user, steps = None, []
            else:
                steps.append(t)
        if user is not None:
            out.append(Exchange(user, tuple(steps), None))
        return out

    def window(self, conv_id: int, budget_tokens: int, roles: tuple[str, ...] = ("user", "assistant", "tool")
               ) -> list[Turn]:
        """The most recent turns that fit the budget, oldest first. The newest turn is always included."""
        out, used = [], 0
        for r in self.db.execute(f"SELECT * FROM turns WHERE conv_id=? AND role IN ({','.join('?' * len(roles))})"
                                 " ORDER BY id DESC", (conv_id, *roles)):
            t = _turn(r)
            if out and used + t.tokens > budget_tokens:
                break
            out.append(t)
            used += t.tokens
        return out[::-1]

    def search(self, query: str, limit: int = 6, exclude_conv: int | None = None) -> list[Turn]:
        """Turns from any conversation that match the query (BM25), best first."""
        q = fts_query(query, max_terms=14)
        if not q:
            return []
        sql = ("SELECT t.* FROM turn_fts JOIN turns t ON t.id = turn_fts.rowid WHERE turn_fts MATCH ?"
               + (" AND t.conv_id != ?" if exclude_conv is not None else "") + " ORDER BY bm25(turn_fts) LIMIT ?")
        args = (q, exclude_conv, limit) if exclude_conv is not None else (q, limit)
        return [_turn(r) for r in self.db.execute(sql, args)]

    def count(self, conv_id: int | None = None) -> int:
        if conv_id is None:
            return self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM turns WHERE conv_id=?", (conv_id,)).fetchone()[0]

    def recent_since(self, ts: float, roles: tuple[str, ...] = ("user", "assistant")) -> list[Turn]:
        return [_turn(r) for r in self.db.execute(
            f"SELECT * FROM turns WHERE ts>=? AND role IN ({','.join('?' * len(roles))}) ORDER BY id",
            (ts, *roles))]

    # ------------------------------------------------------------------ contexts and export

    def save_context(self, turn_id: int, system: str) -> None:
        """Keep the system prompt an answer was generated from: with it, the exchange is a training example
        in the exact form the model was used in."""
        with self._lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO contexts(turn_id, system) VALUES (?,?)", (turn_id, system))

    def context(self, turn_id: int) -> str | None:
        r = self.db.execute("SELECT system FROM contexts WHERE turn_id=?", (turn_id,)).fetchone()
        return r["system"] if r else None

    def transcript(self, conv_id: int) -> str:
        """The whole conversation as Markdown (the person's own export of their data)."""
        conv = self.get(conv_id)
        if not conv:
            return ""
        lines = [f"# {conv.title or f'Conversation {conv.id}'}", "",
                 f"Started {_stamp(conv.started)}; last message {_stamp(conv.updated)}; scope `{conv.scope}`."]
        if conv.summary:
            lines += ["", f"**Summary:** {conv.summary}"]
        if conv.tags:
            lines += ["", "**Topics:** " + ", ".join(conv.tags)]
        if conv.threads:
            lines += ["", "**Open threads:**", *(f"- {t}" for t in conv.threads)]
        if conv.commitments:
            lines += ["", "**Commitments:**", *(f"- [{'x' if not c.open else ' '}] {c.label} {c.text}"
                                                   + (f" ({c.who})" if c.who else "") for c in conv.commitments)]
        drafts = self.artifacts(conv_id, "draft")
        if drafts:
            lines += ["", "**Drafts:**", *(f"- #{d['id']} {d['title'] or '(untitled)'} ({d['data'].get('stage', '?')})"
                                           for d in drafts)]
        for t in self.turns(conv_id):
            label = {"user": "You", "assistant": "kgirl", "tool": "Tool result", "system": "System"}[t.role]
            if t.partial:
                label = "kgirl (calling a tool)"
            lines += ["", f"## {label} · {_stamp(t.ts)}", ""]
            lines.append(f"```\n{t.content}\n```" if t.role == "tool" else t.content)
        return "\n".join(lines) + "\n"


def _stamp(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"


def clean_tags(tags) -> list[str]:
    """Topic tags as stored: lowercase words, 2-24 characters, no repeats, at most MAX_TAGS."""
    out: dict[str, None] = {}
    for t in tags or []:
        t = re.sub(r"[\s_]+", "-", str(t).strip().lower()).strip("-")
        if _TAG.fullmatch(t):
            out.setdefault(t, None)
    return list(out)[:MAX_TAGS]


def _words(text: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z0-9]+", text.lower())]


def same_commitment(a: str, b: str) -> bool:
    """Two wordings of one commitment: equal word sets, or one nearly inside the other."""
    x, y = set(_words(a)), set(_words(b))
    if not x or not y:
        return False
    return len(x & y) / min(len(x), len(y)) >= 0.8 and len(x & y) / len(x | y) >= 0.5


def _names(c: Commitment, which) -> bool:
    which = str(which).strip()
    m = re.fullmatch(r"[Cc]?(\d+)", which)
    if m:
        return int(m.group(1)) == c.id
    return len(which) >= 4 and which.lower() in c.text.lower()


def tags_in(text: str, known: list[str] | dict[str, int]) -> list[str]:
    """The known tags a piece of text mentions (whole words, plural-insensitive; "error-handling" is "error handling")."""
    words = _words(text)
    padded = " " + " ".join(words) + " "
    return [t for t in known if " " + " ".join(_words(t.replace("-", " "))) + " " in padded]


def _tags(raw) -> list[str]:
    try:
        v = json.loads(raw or "[]")
    except ValueError:
        return []
    return [t for t in v if isinstance(t, str)] if isinstance(v, list) else []


def _commitments(raw) -> list[Commitment]:
    try:
        v = json.loads(raw or "[]")
    except ValueError:
        return []
    items = [Commitment.from_dict(d) for d in v] if isinstance(v, list) else []
    return [c for c in items if c is not None]


def _artifact(r: sqlite3.Row) -> dict:
    try:
        data = json.loads(r["data"])
    except ValueError:
        data = {}
    return {"id": r["id"], "conv_id": r["conv_id"], "kind": r["kind"], "title": r["title"] or "",
            "data": data if isinstance(data, dict) else {}, "created": r["created"], "updated": r["updated"]}


def _conv(r: sqlite3.Row) -> Conversation:
    try:
        threads = json.loads(r["threads"] or "[]")
    except ValueError:
        threads = []
    return Conversation(r["id"], r["title"] or "", r["summary"] or "", threads if isinstance(threads, list) else [],
                        r["scope"] or "*", r["started"], r["updated"], r["reflected_turn"] or 0,
                        _commitments(r["commitments"]), _tags(r["tags"]))


def _turn(r: sqlite3.Row) -> Turn:
    try:
        recalled = json.loads(r["recalled"] or "[]")
        meta = json.loads(r["meta"] or "{}")
    except ValueError:
        recalled, meta = [], {}
    return Turn(r["id"], r["conv_id"], r["role"], r["content"], r["ts"], recalled if isinstance(recalled, list) else [],
                meta if isinstance(meta, dict) else {})
