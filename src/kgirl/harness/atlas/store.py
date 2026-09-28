"""Atlas persistence: one SQLite file holding the structural graph of every repo.

Schema (all integer ids, foreign keys cascade):

    repos    (id, name, root, origin, head, indexed_at, card)
    files    (id, repo_id, path, lang, sha, size, loc, module, doc, entry, parse_error)
    symbols  (id, file_id, kind, name, qualname, signature, doc, line, end_line, body_hash)
    imports  (id, file_id, target, names, level, line, kind, resolved_file_id)
    refs     (file_id, name)                      -- identifiers a file calls/uses
    sym_fts  FTS5(terms, name, signature, doc, path)  rowid = symbols.id

Ingest is incremental per file (content sha1). Import resolution is global
because an import in kgirl may resolve to a module that lives in numbskull.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..util import fts_query, harness_home, now, search_terms, sha1, stable_json
from .parse import PARSER_VERSION, FileFacts, detect_language, extract

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".env", "dist", "build", ".next",
    ".mypy_cache", ".pytest_cache", ".tox", "site-packages", ".idea", ".vscode", "target", "vendor",
    ".gradle", ".godot", "coverage", ".cache", "releases", "weights", "models",
})
SKIP_SUFFIXES = (".min.js", ".map", ".lock", "-lock.json", ".bundle.js")
MAX_BYTES = 768 * 1024

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS repos (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, root TEXT NOT NULL, origin TEXT,
  head TEXT, indexed_at REAL, card TEXT);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY, repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  path TEXT NOT NULL, lang TEXT, sha TEXT, size INTEGER, loc INTEGER, module TEXT, doc TEXT,
  entry INTEGER DEFAULT 0, parse_error TEXT, UNIQUE(repo_id, path));
CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT, name TEXT, qualname TEXT, signature TEXT, doc TEXT, line INTEGER, end_line INTEGER,
  body_hash TEXT);
CREATE TABLE IF NOT EXISTS imports (
  id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  target TEXT, names TEXT, level INTEGER, line INTEGER, kind TEXT,
  resolved_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL);
CREATE TABLE IF NOT EXISTS refs (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE, name TEXT NOT NULL,
  PRIMARY KEY (file_id, name)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_sym_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS ix_sym_hash ON symbols(body_hash) WHERE body_hash != '';
CREATE INDEX IF NOT EXISTS ix_imp_file ON imports(file_id);
CREATE INDEX IF NOT EXISTS ix_imp_res ON imports(resolved_file_id);
CREATE INDEX IF NOT EXISTS ix_refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS ix_files_module ON files(module);
CREATE VIRTUAL TABLE IF NOT EXISTS sym_fts USING fts5(
  terms, name, signature, doc, path, tokenize = 'unicode61 remove_diacritics 2');
"""


@dataclass(frozen=True)
class IngestStats:
    repo: str
    scanned: int
    changed: int
    removed: int
    symbols: int


@dataclass(frozen=True)
class Hit:
    repo: str
    path: str
    kind: str
    qualname: str
    signature: str
    doc: str
    line: int
    score: float

    def cite(self) -> str:
        return f"{self.repo}:{self.path}:{self.line}"


def module_name(path: str, lang: str) -> str:
    """Dotted module path used for import resolution (`src/kgirl/core/x.py` -> `src.kgirl.core.x`)."""
    p = Path(path)
    if lang == "python":
        parts = list(p.with_suffix("").parts) if p.suffix else list(p.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    if lang == "julia":
        return p.stem
    return str(p.with_suffix(""))


class Atlas:
    """Structural index over many repositories. Thread-safe for reads and ingest."""

    def __init__(self, db_path: str | os.PathLike | None = None):
        path = Path(db_path) if db_path else harness_home() / "atlas.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ ingest

    def ingest(self, root: str | os.PathLike, name: str | None = None, origin: str = "") -> IngestStats:
        root = Path(root).resolve()
        name = name or root.name
        head = _git_head(root)
        with self._lock, self.db:
            row = self.db.execute("SELECT id FROM repos WHERE name = ?", (name,)).fetchone()
            if row:
                repo_id = row["id"]
                self.db.execute("UPDATE repos SET root=?, origin=COALESCE(NULLIF(?, ''), origin), head=? WHERE id=?",
                                (str(root), origin, head, repo_id))
            else:
                repo_id = self.db.execute("INSERT INTO repos(name, root, origin, head) VALUES (?,?,?,?)",
                                          (name, str(root), origin, head)).lastrowid
            known = {r["path"]: (r["id"], r["sha"]) for r in
                     self.db.execute("SELECT id, path, sha FROM files WHERE repo_id=?", (repo_id,))}
            seen: set[str] = set()
            scanned = changed = nsym = 0
            for rel, text, lang in _walk(root):
                scanned += 1
                seen.add(rel)
                digest = sha1(PARSER_VERSION + "\x00" + text)
                prev = known.get(rel)
                if prev and prev[1] == digest:
                    continue
                changed += 1
                if prev:
                    self._drop_file(prev[0])
                facts = extract(rel, text, lang)
                nsym += self._store_file(repo_id, rel, lang, digest, len(text), facts)
            removed = [fid for p, (fid, _) in known.items() if p not in seen]
            for fid in removed:
                self._drop_file(fid)
            self.db.execute("UPDATE repos SET indexed_at=? WHERE id=?", (now(), repo_id))
        if changed or removed:
            self.resolve_imports()
        return IngestStats(name, scanned, changed, len(removed), nsym)

    def _drop_file(self, file_id: int) -> None:
        self.db.execute("DELETE FROM sym_fts WHERE rowid IN (SELECT id FROM symbols WHERE file_id=?)", (file_id,))
        self.db.execute("DELETE FROM files WHERE id=?", (file_id,))

    def _store_file(self, repo_id: int, rel: str, lang: str, digest: str, size: int, f: FileFacts) -> int:
        fid = self.db.execute(
            "INSERT INTO files(repo_id, path, lang, sha, size, loc, module, doc, entry, parse_error)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (repo_id, rel, lang, digest, size, f.loc, module_name(rel, lang), f.doc, int(f.is_entrypoint),
             f.parse_error or None)).lastrowid
        for s in f.symbols:
            sid = self.db.execute(
                "INSERT INTO symbols(file_id, kind, name, qualname, signature, doc, line, end_line, body_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, s.kind, s.name, s.qualname, s.signature, s.doc, s.line, s.end_line, s.body_hash)).lastrowid
            self.db.execute("INSERT INTO sym_fts(rowid, terms, name, signature, doc, path) VALUES (?,?,?,?,?,?)",
                            (sid, search_terms(s.qualname, rel), s.qualname, s.signature, s.doc, rel))
        self.db.executemany(
            "INSERT INTO imports(file_id, target, names, level, line, kind) VALUES (?,?,?,?,?,?)",
            [(fid, i.target, ",".join(i.names), i.level, i.line, i.kind) for i in f.imports])
        self.db.executemany("INSERT OR IGNORE INTO refs(file_id, name) VALUES (?,?)",
                            [(fid, r) for r in f.refs if len(r) > 2])
        return len(f.symbols)

    def resolve_imports(self) -> int:
        from .graph import resolve_all  # local import: graph depends on store types
        with self._lock, self.db:
            return resolve_all(self.db)

    # ------------------------------------------------------------------ queries

    def repos(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM files f WHERE f.repo_id=r.id) AS nfiles,"
            " (SELECT COUNT(*) FROM symbols s JOIN files f ON f.id=s.file_id WHERE f.repo_id=r.id) AS nsymbols"
            " FROM repos r ORDER BY r.name").fetchall()

    def repo_id(self, name: str) -> int | None:
        row = self.db.execute("SELECT id FROM repos WHERE name=? OR lower(name)=lower(?)", (name, name)).fetchone()
        return row["id"] if row else None

    def search(self, query: str, limit: int = 12, repo: str | None = None,
               kinds: tuple[str, ...] | None = None) -> list[Hit]:
        q = fts_query(query)
        if not q:
            return []
        sql = ("SELECT r.name AS repo, f.path, s.kind, s.qualname, s.signature, s.doc, s.line,"
               " bm25(sym_fts, 1.0, 4.0, 1.5, 1.0, 0.7) AS rank"
               " FROM sym_fts JOIN symbols s ON s.id = sym_fts.rowid JOIN files f ON f.id = s.file_id"
               " JOIN repos r ON r.id = f.repo_id WHERE sym_fts MATCH ?")
        args: list = [q]
        if repo:
            sql += " AND r.name = ?"
            args.append(repo)
        if kinds:
            sql += f" AND s.kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit * 3)
        rows = self.db.execute(sql, args).fetchall()
        hits: list[Hit] = []
        per_file: dict[tuple[str, str], int] = {}
        for r in rows:  # diversity: at most 3 symbols per file
            key = (r["repo"], r["path"])
            if per_file.get(key, 0) >= 3:
                continue
            per_file[key] = per_file.get(key, 0) + 1
            hits.append(Hit(r["repo"], r["path"], r["kind"], r["qualname"], r["signature"] or "",
                            r["doc"] or "", r["line"] or 0, -float(r["rank"])))
            if len(hits) >= limit:
                break
        return hits

    def outline(self, repo: str, path: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT s.kind, s.qualname, s.signature, s.doc, s.line, s.end_line FROM symbols s"
            " JOIN files f ON f.id=s.file_id JOIN repos r ON r.id=f.repo_id"
            " WHERE r.name=? AND f.path=? ORDER BY s.line", (repo, path)).fetchall()

    def file_row(self, repo: str, path: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT f.*, r.name AS repo, r.root FROM files f JOIN repos r ON r.id=f.repo_id"
            " WHERE r.name=? AND f.path=?", (repo, path)).fetchone()

    def find_symbol(self, name: str, repo: str | None = None) -> list[sqlite3.Row]:
        """Exact lookup by `name`, `qualname`, or `path::qualname`."""
        path = None
        if "::" in name:
            path, name = name.split("::", 1)
        sql = ("SELECT s.*, f.path, r.name AS repo FROM symbols s JOIN files f ON f.id=s.file_id"
               " JOIN repos r ON r.id=f.repo_id WHERE (s.qualname=? OR s.name=?)")
        args: list = [name, name]
        if path:
            sql += " AND (f.path=? OR f.path LIKE ?)"
            args += [path, "%/" + path]
        if repo:
            sql += " AND r.name=?"
            args.append(repo)
        return self.db.execute(sql + " ORDER BY length(f.path) LIMIT 50", args).fetchall()

    def read_source(self, repo: str, path: str, start: int = 1, end: int | None = None) -> str:
        row = self.file_row(repo, path)
        if not row:
            raise FileNotFoundError(f"{repo}:{path} is not indexed")
        full = Path(row["root"]) / path
        lines = full.read_text(encoding="utf-8", errors="replace").split("\n")
        end = min(len(lines), end or start + 80)
        return "\n".join(f"{n:>5} {lines[n - 1]}" for n in range(max(1, start), end + 1))

    def set_card(self, repo: str, card: dict) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE repos SET card=? WHERE name=?", (stable_json(card), repo))


# ---------------------------------------------------------------------- filesystem


def _walk(root: Path) -> Iterator[tuple[str, str, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
                             and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            if fn.endswith(SKIP_SUFFIXES):
                continue
            full = Path(dirpath) / fn
            try:
                if full.is_symlink() or full.stat().st_size > MAX_BYTES:
                    continue
                raw = full.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
            rel = full.relative_to(root).as_posix()
            lang = detect_language(rel, text[:200])
            if lang and not _is_console_shim(fn, text):
                yield rel, text, lang


def _is_console_shim(filename: str, text: str) -> bool:
    """pip/venv console-script launchers (`#!/…/venv/bin/python` + `sys.exit(main())`).

    Several repos have a virtualenv's bin/ committed at the root; indexing those
    shims would make `tqdm`, `pip`, … look like first-party modules.
    """
    if "." in filename or not text.startswith("#!"):
        return False
    return text.count("\n") < 16 and "sys.exit(" in text and "import" in text


def _git_head(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""
