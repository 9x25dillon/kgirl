"""Conversation memory: every turn kgirl has, stored, searchable and summarized.

    conversations  one row per thread: title, rolling summary, open threads, timestamps
    turns          every message (user, assistant, tool) in order, with the memory ids it was answered from
    turn_fts       full-text index over turns, so a new conversation can recall what was said in old ones

The rolling summary is what lets a small model "understand" a long conversation: the prompt carries the
summary plus the most recent turns that fit a token budget, instead of the whole transcript.
Lives beside soup.db in the harness home (private, 0700; see docs/PRIVACY.md).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..util import estimate_tokens, fts_query, harness_home, now, search_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY, title TEXT DEFAULT '', summary TEXT DEFAULT '', threads TEXT DEFAULT '[]',
  scope TEXT DEFAULT '*', started REAL, updated REAL, reflected_turn INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY, conv_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
  ts REAL, recalled TEXT DEFAULT '[]', meta TEXT DEFAULT '{}');
CREATE INDEX IF NOT EXISTS ix_turns_conv ON turns(conv_id, id);
CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(terms, content, tokenize='unicode61 remove_diacritics 2');
"""

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

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ threads

    def new(self, title: str = "", scope: str = "*") -> int:
        t = now()
        with self._lock, self.db:
            return self.db.execute("INSERT INTO conversations(title, scope, started, updated) VALUES (?,?,?,?)",
                                   (title, scope, t, t)).lastrowid

    def get(self, conv_id: int) -> Conversation | None:
        r = self.db.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        return _conv(r) if r else None

    def latest(self) -> Conversation | None:
        r = self.db.execute("SELECT * FROM conversations ORDER BY updated DESC LIMIT 1").fetchone()
        return _conv(r) if r else None

    def list(self, limit: int = 20) -> list[Conversation]:
        return [_conv(r) for r in self.db.execute("SELECT * FROM conversations ORDER BY updated DESC LIMIT ?",
                                                  (limit,))]

    def update(self, conv_id: int, *, title: str | None = None, summary: str | None = None,
               threads: list[str] | None = None, reflected_turn: int | None = None) -> None:
        sets, args = [], []
        for col, val in (("title", title), ("summary", summary), ("reflected_turn", reflected_turn)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        if threads is not None:
            sets.append("threads=?")
            args.append(json.dumps(threads[:12], ensure_ascii=False))
        if not sets:
            return
        with self._lock, self.db:
            self.db.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id=?", (*args, conv_id))

    def delete(self, conv_id: int) -> None:
        with self._lock, self.db:
            ids = [r["id"] for r in self.db.execute("SELECT id FROM turns WHERE conv_id=?", (conv_id,))]
            self.db.executemany("DELETE FROM turn_fts WHERE rowid=?", [(i,) for i in ids])
            self.db.execute("DELETE FROM turns WHERE conv_id=?", (conv_id,))
            self.db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

    # ------------------------------------------------------------------ turns

    def append(self, conv_id: int, role: str, content: str, recalled: list[int] | None = None,
               meta: dict | None = None) -> int:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        t = now()
        with self._lock, self.db:
            tid = self.db.execute("INSERT INTO turns(conv_id, role, content, ts, recalled, meta) VALUES (?,?,?,?,?,?)",
                                  (conv_id, role, content, t, json.dumps(recalled or []),
                                   json.dumps(meta or {}, ensure_ascii=False))).lastrowid
            if role in ("user", "assistant"):
                self.db.execute("INSERT INTO turn_fts(rowid, terms, content) VALUES (?,?,?)",
                                (tid, search_terms(content), content))
            self.db.execute("UPDATE conversations SET updated=? WHERE id=?", (t, conv_id))
        return tid

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


def _conv(r: sqlite3.Row) -> Conversation:
    try:
        threads = json.loads(r["threads"] or "[]")
    except ValueError:
        threads = []
    return Conversation(r["id"], r["title"] or "", r["summary"] or "", threads, r["scope"] or "*",
                        r["started"], r["updated"], r["reflected_turn"] or 0)


def _turn(r: sqlite3.Row) -> Turn:
    try:
        recalled = json.loads(r["recalled"] or "[]")
        meta = json.loads(r["meta"] or "{}")
    except ValueError:
        recalled, meta = [], {}
    return Turn(r["id"], r["conv_id"], r["role"], r["content"], r["ts"], recalled, meta)
