"""Soup: one shared memory pool every agent reads from and feeds back into.

A Fragment is a small, self-contained unit of knowledge:

    fact         "numbskull_dual_orchestrator is imported by 8 kgirl files"
    abstraction  "to add an Ollama backend, implement Backend.available/complete"
    trajectory   a compressed accepted agent run (goal, op sequence, diff, verdict)
    preference   "user wants Claude for summaries, local models for code"
    antipattern  "editing tests to make them pass was rejected"
    note         anything else

Recall is *just-in-time*: at task arrival, fragments are ranked by
relevance x utility x recency and packed greedily into a token budget with a
near-duplicate filter. Nothing is recalled that does not fit.

Utility is a Beta(1,1)-prior posterior over outcomes of tasks the fragment was
recalled into (credit assignment). That makes memory *task adaptive* without
touching model weights: fragments that help get recalled more; fragments that
co-occur with failures sink. (Confidence: Level 0 mechanism — bandit-style
credit assignment; whether it improves a given local model is Level 2 and is
what `soup stats` + the ledger let you measure.)
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from ..util import estimate_tokens, fts_query, harness_home, jaccard, now, search_terms, sha1, shingles

KINDS = ("fact", "abstraction", "trajectory", "preference", "antipattern", "note")
STATUSES = ("staged", "active", "retired")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, tags TEXT DEFAULT '',
  source TEXT DEFAULT '', scope TEXT DEFAULT '*', status TEXT NOT NULL DEFAULT 'active',
  support INTEGER DEFAULT 1, wins REAL DEFAULT 0, losses REAL DEFAULT 0, uses INTEGER DEFAULT 0,
  created REAL, last_used REAL, hash TEXT UNIQUE, data TEXT);
CREATE INDEX IF NOT EXISTS ix_frag_status ON fragments(status, kind);
CREATE VIRTUAL TABLE IF NOT EXISTS frag_fts USING fts5(terms, text, tags, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS ledger (
  seq INTEGER PRIMARY KEY, ts REAL, op TEXT, fragment_id INTEGER, before TEXT, after TEXT, detail TEXT);
"""


@dataclass(frozen=True)
class Fragment:
    id: int
    kind: str
    text: str
    tags: str
    source: str
    scope: str
    status: str
    support: int
    wins: float
    losses: float
    uses: int
    created: float
    last_used: float | None
    data: str | None = None

    @property
    def evidence(self) -> float:
        return self.wins + self.losses

    @property
    def utility(self) -> float:
        return (self.wins + 1.0) / (self.evidence + 2.0)

    def bound(self, z: float) -> float:
        """Normal-approx bound on utility; z<0 lower, z>0 upper."""
        n = self.evidence + 2.0
        m = self.utility
        return min(1.0, max(0.0, m + z * math.sqrt(m * (1 - m) / n)))

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def render(self) -> str:
        tag = f" [{self.tags}]" if self.tags else ""
        return f"({self.kind}#{self.id} u={self.utility:.2f}){tag} {self.text}"


@dataclass(frozen=True)
class Recall:
    query: str
    fragments: tuple[Fragment, ...]
    tokens: int
    budget: int

    @property
    def ids(self) -> list[int]:
        return [f.id for f in self.fragments]

    def render(self) -> str:
        return "\n".join(f"- {f.render()}" for f in self.fragments)


class Soup:
    def __init__(self, db_path: str | os.PathLike | None = None, half_life_days: float = 30.0):
        path = Path(db_path) if db_path else harness_home() / "soup.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.half_life = half_life_days * 86400.0
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ writes

    def add(self, text: str, kind: str = "note", tags: str | list[str] = "", source: str = "", scope: str = "*",
            status: str = "active", data: str | None = None) -> int:
        """Insert a fragment; an exact duplicate reinforces the existing one instead."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        text = " ".join(text.split()) if kind != "trajectory" else text.strip()
        if not text:
            raise ValueError("empty fragment")
        tags = ",".join(tags) if isinstance(tags, (list, tuple)) else tags
        h = sha1(f"{kind}\x00{scope}\x00{text}")
        with self._lock, self.db:
            row = self.db.execute("SELECT id FROM fragments WHERE hash=?", (h,)).fetchone()
            if row:
                self.db.execute("UPDATE fragments SET support = support + 1 WHERE id=?", (row["id"],))
                self._log("reinforce", row["id"], detail=source)
                return row["id"]
            fid = self.db.execute(
                "INSERT INTO fragments(kind, text, tags, source, scope, status, created, hash, data)"
                " VALUES (?,?,?,?,?,?,?,?,?)", (kind, text, tags, source, scope, status, now(), h, data)).lastrowid
            self.db.execute("INSERT INTO frag_fts(rowid, terms, text, tags) VALUES (?,?,?,?)",
                            (fid, search_terms(text, tags), text, tags))
            self._log("add", fid, after=status, detail=f"{kind} {source}")
            return fid

    def credit(self, ids: list[int], success: bool, weight: float = 1.0) -> None:
        if not ids:
            return
        col = "wins" if success else "losses"
        with self._lock, self.db:
            self.db.executemany(f"UPDATE fragments SET {col} = {col} + ? WHERE id=?", [(weight, i) for i in ids])
            self._log("credit", None, detail=f"{col}+{weight} -> {ids[:20]}")

    def reinforce(self, fid: int, detail: str = "") -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE fragments SET support = support + 1 WHERE id=?", (fid,))
            self._log("reinforce", fid, detail=detail)

    def set_status(self, fid: int, status: str, reason: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(status)
        with self._lock, self.db:
            row = self.db.execute("SELECT status FROM fragments WHERE id=?", (fid,)).fetchone()
            if not row or row["status"] == status:
                return
            self.db.execute("UPDATE fragments SET status=? WHERE id=?", (status, fid))
            self._log("status", fid, before=row["status"], after=status, detail=reason)

    def decay(self, gamma: float) -> None:
        """Shrink all evidence toward the prior (regularization: old wins must be re-earned)."""
        with self._lock, self.db:
            self.db.execute("UPDATE fragments SET wins = wins * ?, losses = losses * ? WHERE status != 'retired'",
                            (gamma, gamma))
            self._log("decay", None, detail=f"gamma={gamma}")

    def _log(self, op: str, fid: int | None, before: str = "", after: str = "", detail: str = "") -> None:
        self.db.execute("INSERT INTO ledger(ts, op, fragment_id, before, after, detail) VALUES (?,?,?,?,?,?)",
                        (now(), op, fid, before, after, detail[:500]))

    # ------------------------------------------------------------------ reads

    def get(self, fid: int) -> Fragment | None:
        row = self.db.execute("SELECT * FROM fragments WHERE id=?", (fid,)).fetchone()
        return _frag(row) if row else None

    def all(self, status: str | None = None, kind: str | None = None) -> list[Fragment]:
        sql, args = "SELECT * FROM fragments WHERE 1=1", []
        if status:
            sql, args = sql + " AND status=?", args + [status]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        return [_frag(r) for r in self.db.execute(sql + " ORDER BY id", args)]

    def similar(self, text: str, kind: str | None = None, limit: int = 10) -> list[tuple[Fragment, float]]:
        q = fts_query(text, max_terms=16)
        if not q:
            return []
        sql = ("SELECT f.* FROM frag_fts JOIN fragments f ON f.id = frag_fts.rowid WHERE frag_fts MATCH ?"
               + (" AND f.kind=?" if kind else "") + " AND f.status != 'retired' ORDER BY bm25(frag_fts) LIMIT ?")
        rows = self.db.execute(sql, (q, kind, limit) if kind else (q, limit)).fetchall()
        sh = shingles(text, 2)
        return sorted(((_frag(r), jaccard(sh, shingles(r["text"], 2))) for r in rows), key=lambda t: -t[1])

    def recall(self, query: str, budget_tokens: int = 600, kinds: tuple[str, ...] | None = None,
               scope: str | None = None, statuses: tuple[str, ...] = ("active",), exclude: tuple[int, ...] = (),
               candidates: int = 40, diversity: float = 0.6, offset: int = 0) -> Recall:
        """Just-in-time recall under a hard token budget.

        `offset` rotates the candidate list so parallel agents can be given
        different (diverse) memory slices of the same pool.
        """
        clauses = [f"f.status IN ({','.join('?' * len(statuses))})"]
        args: list = list(statuses)
        if kinds:
            clauses.append(f"f.kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        if scope:
            clauses.append("f.scope IN ('*', ?)")
            args.append(scope)
        where = " AND ".join(clauses)
        q = fts_query(query, max_terms=16)
        rows: list[tuple[sqlite3.Row, float]] = []
        if q:
            for r in self.db.execute(
                    f"SELECT f.*, bm25(frag_fts) AS rank FROM frag_fts JOIN fragments f ON f.id=frag_fts.rowid"
                    f" WHERE frag_fts MATCH ? AND {where} ORDER BY rank LIMIT ?", [q, *args, candidates]):
                rows.append((r, -float(r["rank"])))
        if len(rows) < 3:  # sparse query: fall back to the most useful fragments in scope
            seen = {r["id"] for r, _ in rows}
            for r in self.db.execute(
                    f"SELECT f.* FROM fragments f WHERE {where} ORDER BY (f.wins+1.0)/(f.wins+f.losses+2.0) DESC,"
                    f" f.support DESC LIMIT ?", [*args, candidates]):
                if r["id"] not in seen:
                    rows.append((r, 0.0))
        if not rows:
            return Recall(query, (), 0, budget_tokens)
        top = max(s for _, s in rows) or 1.0
        t = now()
        scored: list[tuple[float, Fragment]] = []
        for r, s in rows:
            f = _frag(r)
            if f.id in exclude:
                continue
            rel = 0.25 + 0.75 * (s / top if top > 0 else 0.0)
            age = t - (f.last_used or f.created or t)
            recency = 0.5 ** (age / self.half_life) if self.half_life > 0 else 1.0
            support = 1.0 + 0.1 * math.log1p(f.support)
            scored.append((rel * (0.35 + 0.65 * f.utility) * (0.5 + 0.5 * recency) * support, f))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        if offset and scored:
            k = offset % len(scored)
            scored = scored[k:] + scored[:k]
        picked: list[Fragment] = []
        picked_sh: list[set[str]] = []
        used = 0
        for _, f in scored:
            if used + f.tokens > budget_tokens:
                continue
            sh = shingles(f.text, 2)
            if any(jaccard(sh, p) >= diversity for p in picked_sh):
                continue
            picked.append(f)
            picked_sh.append(sh)
            used += f.tokens
        if picked:
            with self._lock, self.db:
                self.db.executemany("UPDATE fragments SET uses = uses + 1, last_used = ? WHERE id=?",
                                    [(t, f.id) for f in picked])
        return Recall(query, tuple(picked), used, budget_tokens)

    def stats(self) -> dict:
        rows = self.db.execute("SELECT status, kind, COUNT(*) AS n, AVG((wins+1.0)/(wins+losses+2.0)) AS u"
                               " FROM fragments GROUP BY status, kind ORDER BY status, kind").fetchall()
        return {f"{r['status']}/{r['kind']}": {"n": r["n"], "mean_utility": round(r["u"], 3)} for r in rows}

    def ledger(self, since: float = 0.0, limit: int = 200) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM ledger WHERE ts >= ? ORDER BY seq DESC LIMIT ?",
                               (since, limit)).fetchall()


def _frag(r: sqlite3.Row) -> Fragment:
    return Fragment(r["id"], r["kind"], r["text"], r["tags"] or "", r["source"] or "", r["scope"] or "*",
                    r["status"], r["support"], r["wins"], r["losses"], r["uses"], r["created"], r["last_used"],
                    r["data"])
