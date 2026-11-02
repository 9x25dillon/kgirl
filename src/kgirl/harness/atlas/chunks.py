"""Full-text chunk index: the text that Atlas deliberately does not keep.

Chunks are cut at Atlas symbol boundaries, so a hit cites the same
`repo:path:line` that `Atlas.search` does. The output is one self-contained
SQLite file, built on the desktop and copied to a slower read-only store:

    meta       (key, value)
    chunks     (id, repo, path, lang, kind, symbol, start_line, end_line, text)
    chunks_fts contentless FTS5(symbol, path, body, terms)  rowid = chunks.id

    python -m kgirl.harness.atlas.chunks . chunks.db --repo kgirl
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..util import fts_query, now, search_terms
from .parse import PARSER_VERSION, FileFacts, extract
from .store import _git_head, _walk

MAX_LINES = 160  # a longer span is split into overlapping windows
OVERLAP = 20
BM25 = "bm25(chunks_fts, 4.0, 0.7, 1.0, 0.3)"  # symbol, path, body, terms

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, repo TEXT NOT NULL, path TEXT NOT NULL, lang TEXT, kind TEXT, symbol TEXT,
  start_line INTEGER, end_line INTEGER, text TEXT NOT NULL);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  symbol, path, body, terms, content = '', tokenize = 'unicode61 remove_diacritics 2');
"""


@dataclass(frozen=True)
class Chunk:
    kind: str
    symbol: str
    start_line: int
    end_line: int
    text: str


def chunk_file(text: str, facts: FileFacts) -> list[Chunk]:
    """Cover the whole file, starting a new chunk at each symbol's first line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    by_line: dict[int, tuple[str, str]] = {}
    for s in facts.symbols:
        if 1 <= s.line <= len(lines):
            by_line.setdefault(s.line, (s.kind, s.qualname))
    starts = sorted({1, *by_line})
    out: list[Chunk] = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines)
        kind, symbol = by_line.get(start, ("preamble", ""))
        first = start
        while first <= stop:
            last = min(stop, first + MAX_LINES - 1)
            body = "\n".join(lines[first - 1:last])
            if body.strip():
                out.append(Chunk(kind, symbol, first, last, body))
            if last == stop:
                break
            first = last + 1 - OVERLAP
    return out


def build(root: str | os.PathLike, db_path: str | os.PathLike, repo: str | None = None) -> dict[str, int]:
    """Rebuild `db_path` from scratch. The old file stays in place until the new one is complete."""
    root = Path(root).resolve()
    repo = repo or root.name
    target = Path(db_path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(str(tmp))
    try:
        db.executescript(SCHEMA)
        nfiles = nchunks = 0
        with db:
            for rel, text, lang in _walk(root):
                nfiles += 1
                for c in chunk_file(text, extract(rel, text, lang)):
                    cid = db.execute(
                        "INSERT INTO chunks(repo, path, lang, kind, symbol, start_line, end_line, text)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (repo, rel, lang, c.kind, c.symbol, c.start_line, c.end_line, c.text)).lastrowid
                    db.execute("INSERT INTO chunks_fts(rowid, symbol, path, body, terms) VALUES (?,?,?,?,?)",
                               (cid, c.symbol, rel, c.text, search_terms(c.symbol, rel, c.text)))
                    nchunks += 1
            db.executemany("INSERT INTO meta VALUES (?,?)", [
                ("repo", repo), ("head", _git_head(root)), ("built_at", str(now())),
                ("parser_version", PARSER_VERSION)])
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        db.commit()
        db.execute("ANALYZE")
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    tmp.replace(target)
    return {"files": nfiles, "chunks": nchunks, "bytes": target.stat().st_size}


def search_sql(limit: int = 10) -> str:
    return ("SELECT c.repo, c.path, c.start_line, c.end_line, c.kind, c.symbol, c.text, " + BM25 + " AS rank"
            " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
            f" WHERE chunks_fts MATCH ? ORDER BY rank LIMIT {int(limit)}")


def search(db: sqlite3.Connection, query: str, limit: int = 10) -> list[tuple]:
    q = fts_query(query)
    return db.execute(search_sql(limit), (q,)).fetchall() if q else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root")
    ap.add_argument("db")
    ap.add_argument("--repo")
    args = ap.parse_args()
    print(build(args.root, args.db, args.repo))


if __name__ == "__main__":
    main()
