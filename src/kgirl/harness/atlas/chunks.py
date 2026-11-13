"""Full-text chunk index: the text that Atlas deliberately does not keep.

Chunks are cut at Atlas symbol boundaries, so a hit cites the same
`repo:path:line` that `Atlas.search` does. The output is one self-contained
SQLite file, built on the desktop and copied to a slower read-only store:

    meta       (key, value)
    chunks     (id, repo, path, lang, kind, symbol, start_line, end_line, text)
    chunks_fts contentless FTS5(symbol, path, body, terms)  rowid = chunks.id

    python -m kgirl.harness.atlas.chunks . chunks.db --repo kgirl     # one working tree
    python -m kgirl.harness chunks                                     # every indexed repo, committed files only

Files that look like they hold credentials are left out (see `secrets.py`); `meta.filtered` lists them by
`repo:path (rule)`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..util import fts_query, now, search_terms
from .parse import PARSER_VERSION, FileFacts, detect_language, extract
from .secrets import secret_reason
from .store import MAX_BYTES, SKIP_DIRS, SKIP_SUFFIXES, _git_head, _is_console_shim, _walk

Files = Iterable[tuple[str, str, str]]  # (path, text, language), as store._walk yields them

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


def build(root: str | os.PathLike, db_path: str | os.PathLike, repo: str | None = None) -> dict:
    """Rebuild `db_path` from one working tree. The old file stays in place until the new one is complete."""
    root = Path(root).resolve()
    repo = repo or root.name
    return _build(db_path, [(repo, _walk(root))], {"repo": repo, "head": _git_head(root)})


def build_committed(repos: Sequence[tuple[str, Path]], db_path: str | os.PathLike) -> dict:
    """One index over many repos, from each one's HEAD commit: untracked files and uncommitted edits never get in."""
    heads = {name: _git_head(Path(root)) for name, root in repos}
    missing = sorted(name for name, head in heads.items() if not head)
    if missing:
        raise ValueError(f"not git checkouts with a commit: {', '.join(missing)}")
    return _build(db_path, [(name, committed_files(Path(root))) for name, root in repos],
                  {"repo": "all", "head": json.dumps(heads)})


def committed_files(root: Path) -> Iterator[tuple[str, str, str]]:
    """Files at HEAD, read through `git archive`, filtered the way store._walk filters a working tree."""
    proc = subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    finished = False
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile() or member.size > MAX_BYTES:
                    continue
                parts = member.name.split("/")
                if any(d in SKIP_DIRS or d.startswith(".") or d.endswith(".egg-info") for d in parts[:-1]):
                    continue
                if parts[-1].endswith(SKIP_SUFFIXES):
                    continue
                f = tar.extractfile(member)
                raw = f.read() if f else b""
                if b"\x00" in raw[:4096]:
                    continue
                text = raw.decode("utf-8", errors="replace")
                lang = detect_language(member.name, text[:200])
                if lang and not _is_console_shim(parts[-1], text):
                    yield member.name, text, lang
        finished = True
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.stderr.close()
        code = proc.wait()
    if finished and code != 0:
        raise RuntimeError(f"git archive in {root} failed: {err[:300]}")


def _build(db_path: str | os.PathLike, sources: Sequence[tuple[str, Files]], meta: dict[str, str]) -> dict:
    target = Path(db_path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(str(tmp))
    filtered: list[str] = []
    try:
        db.executescript(SCHEMA)
        nfiles = nchunks = 0
        with db:
            for repo, files in sources:
                for rel, text, lang in files:
                    why = secret_reason(rel, text)
                    if why:
                        filtered.append(f"{repo}:{rel} ({why})")
                        continue
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
                *meta.items(), ("built_at", str(now())), ("parser_version", PARSER_VERSION),
                ("filtered", json.dumps(filtered))])
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        db.commit()
        db.execute("ANALYZE")
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    tmp.replace(target)
    return {"files": nfiles, "chunks": nchunks, "bytes": target.stat().st_size, "filtered": filtered}


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
