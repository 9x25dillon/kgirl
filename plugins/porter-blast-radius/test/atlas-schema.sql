-- Atlas schema, copied verbatim from kgirl src/kgirl/harness/atlas/store.py (SCHEMA)
-- at commit 26748c4 (branch claude/dazzling-sagan-0uf4y7). If Atlas changes its schema, update
-- this file and the bridge's REQUIRED_COLUMNS together; the contract test will say which.
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
