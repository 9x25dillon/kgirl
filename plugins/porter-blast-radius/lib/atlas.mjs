// Read-only bridge from a kgirl Atlas database to a coupling model.
//
// Atlas (kgirl/src/kgirl/harness/atlas, commit 26748c4) keeps the structural graph of many
// repos in one SQLite file: files, symbols with normalized body hashes, and imports resolved
// across repos. This module opens that file with SQLite's read-only flag (the engine itself
// refuses writes), checks the columns it relies on, and emits the connector's model format:
//
//   import  provider file/repo -> importer   kinds change + runtime
//   clone   identical symbol bodies          kind change, both directions: a defect fixed in
//                                            one copy still lives in the other (Atlas's own
//                                            "clone" edge semantics)
//
// Atlas resolves a bare Python module name (`from db import x`) to a same-named file in any
// repo when the importer's repo has none. Across repos that is a guess, so such edges are
// flagged in the evidence (bareName) and can be left out with excludeBareNames.

import { cleanText, normalizeRemote } from './discover.mjs';
import { ToolError } from './fsguard.mjs';

// Columns this bridge reads, per Atlas table. A missing one means an incompatible Atlas.
export const REQUIRED_COLUMNS = {
  repos: ['id', 'name', 'origin', 'head'],
  files: ['id', 'repo_id', 'path', 'lang', 'parse_error'],
  symbols: ['id', 'file_id', 'body_hash'],
  imports: ['id', 'file_id', 'resolved_file_id', 'target', 'line'],
};

function openReadOnly(path) {
  const sqlite = process.getBuiltinModule?.('node:sqlite');
  if (!sqlite) throw new ToolError(`reading an Atlas database needs Node 22.13 or later (node:sqlite); this is Node ${process.versions.node}`);
  try {
    return new sqlite.DatabaseSync(path, { readOnly: true });
  } catch (e) {
    throw new ToolError(`cannot open ${path} read-only: ${cleanText(e.message, 120)}`);
  }
}

export function checkAtlasSchema(db) {
  for (const [table, cols] of Object.entries(REQUIRED_COLUMNS)) {
    let have;
    try {
      have = new Set(db.prepare(`PRAGMA table_info(${table})`).all().map((r) => r.name));
    } catch (e) {
      throw new ToolError(`not an Atlas database: ${cleanText(e.message, 120)}`);
    }
    const missing = cols.filter((c) => !have.has(c));
    if (missing.length) throw new ToolError(`not a compatible Atlas database: table ${table} ${have.size ? `lacks ${missing.join(', ')}` : 'is missing'}`);
  }
}

const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '').slice(0, 64) || 'repo';

function repoComponents(db) {
  const stats = new Map(db.prepare(
    "SELECT repo_id, COUNT(*) AS files, SUM(CASE WHEN parse_error IS NOT NULL AND parse_error != '' THEN 1 ELSE 0 END) AS broken FROM files GROUP BY repo_id",
  ).all().map((r) => [r.repo_id, r]));
  const used = new Set();
  const byRepo = new Map();
  const components = db.prepare('SELECT id, name, origin, head FROM repos ORDER BY lower(name), name').all().map((r) => {
    let s = slug(r.name);
    for (let i = 2; used.has(s); i++) s = `${slug(r.name)}-${i}`;
    used.add(s);
    const id = `repo:${s}`;
    byRepo.set(r.id, id);
    const st = stats.get(r.id) ?? { files: 0, broken: 0 };
    return {
      id, layer: 'repo', name: cleanText(r.name, 100), remote: r.origin ? normalizeRemote(r.origin) : undefined,
      commit: /^[0-9a-f]{7,64}$/.test(r.head ?? '') ? r.head.slice(0, 12) : undefined,
      files: Number(st.files), parseErrors: Number(st.broken ?? 0),
    };
  });
  return { components, byRepo };
}

// One node per repo; edge weights are counts of resolved imports and shared clone bodies.
function repoGraph(db, { includeClones, excludeBareNames }) {
  const { components, byRepo } = repoComponents(db);
  const edges = [];
  const evidence = [];
  const imports = db.prepare(`
    SELECT fb.repo_id AS provider, fa.repo_id AS importer, COUNT(*) AS n,
           SUM(CASE WHEN fa.lang = 'python' AND COALESCE(i.level, 0) = 0 AND instr(COALESCE(i.target, ''), '.') = 0 THEN 1 ELSE 0 END) AS bare,
           MIN(fa.path || ' -> ' || fb.path) AS example
    FROM imports i JOIN files fa ON fa.id = i.file_id JOIN files fb ON fb.id = i.resolved_file_id
    WHERE fa.repo_id != fb.repo_id GROUP BY fb.repo_id, fa.repo_id`).all();
  for (const r of imports) {
    const from = byRepo.get(r.provider);
    const to = byRepo.get(r.importer);
    const bareOnly = Number(r.bare) === Number(r.n);
    if (bareOnly && excludeBareNames) continue;
    edges.push([from, to, 'change'], [from, to, 'runtime']);
    evidence.push({ from, to, kind: 'import', count: Number(r.n), bareName: Number(r.bare), via: cleanText(r.example, 200) });
  }
  if (includeClones) {
    const clones = db.prepare(`
      SELECT fa.repo_id AS a, fb.repo_id AS b, COUNT(DISTINCT sa.body_hash) AS n, MIN(fa.path || ' == ' || fb.path) AS example
      FROM symbols sa JOIN symbols sb ON sb.body_hash = sa.body_hash AND sb.id != sa.id
      JOIN files fa ON fa.id = sa.file_id JOIN files fb ON fb.id = sb.file_id
      WHERE sa.body_hash IS NOT NULL AND sa.body_hash != '' AND fa.repo_id < fb.repo_id
      GROUP BY fa.repo_id, fb.repo_id`).all();
    for (const r of clones) {
      const a = byRepo.get(r.a);
      const b = byRepo.get(r.b);
      edges.push([a, b, 'change'], [b, a, 'change']);
      evidence.push({ from: a, to: b, kind: 'clone', count: Number(r.n), via: cleanText(r.example, 200) });
    }
  }
  return { components, edges, evidence };
}

// Files of one repo (optionally under a path prefix) plus the other repos they touch.
function fileGraph(db, { repo, pathPrefix = '', includeClones, excludeBareNames, maxFiles }) {
  const { components: repos, byRepo } = repoComponents(db);
  const target = repos.find((c) => c.id === repo || c.name === repo || c.id === `repo:${repo}`);
  if (!target) throw new ToolError(`repo ${cleanText(repo, 80)} is not in the Atlas; indexed repos: ${repos.map((c) => c.name).slice(0, 40).join(', ')}`);
  const repoId = [...byRepo].find(([, id]) => id === target.id)[0];
  const files = db.prepare("SELECT id, path, parse_error FROM files WHERE repo_id = ? AND substr(path, 1, length(?)) = ? ORDER BY path")
    .all(repoId, pathPrefix, pathPrefix);
  if (files.length > maxFiles) throw new ToolError(`${files.length} files match; the limit is ${maxFiles}. Pass path_prefix, raise max_files, or use granularity "repo"`);
  const name = target.id.slice('repo:'.length);
  const fileNode = new Map(files.map((f) => [f.id, `${name}:${f.path}`]));
  const components = files.map((f) => ({
    id: fileNode.get(f.id), layer: 'file', name: cleanText(f.path, 300),
    ...(f.parse_error ? { parseError: cleanText(f.parse_error, 160) } : {}),
  }));
  const external = new Set();
  const edges = [];
  const evidence = [];
  const seen = new Set();
  const add = (from, to, kind, via) => {
    const key = `${from}>${to}>${kind}`;
    if (from === to || seen.has(key)) return;
    seen.add(key);
    edges.push([from, to, kind]);
    evidence.push({ from, to, kind, via });
  };
  const nodeFor = (fileId, repoOf) => fileNode.get(fileId) ?? (external.add(byRepo.get(repoOf)), byRepo.get(repoOf));
  const rows = db.prepare(`
    SELECT i.file_id AS importer, fa.repo_id AS ra, i.resolved_file_id AS provider, fb.repo_id AS rb, i.line, i.target,
           (fa.lang = 'python' AND COALESCE(i.level, 0) = 0 AND instr(COALESCE(i.target, ''), '.') = 0) AS bare
    FROM imports i JOIN files fa ON fa.id = i.file_id JOIN files fb ON fb.id = i.resolved_file_id
    WHERE fa.repo_id = ? OR fb.repo_id = ?`).all(repoId, repoId);
  for (const r of rows) {
    if (!fileNode.has(r.importer) && !fileNode.has(r.provider)) continue; // outside the prefix
    const crossBare = r.ra !== r.rb && Number(r.bare) === 1;
    if (crossBare && excludeBareNames) continue;
    const from = nodeFor(r.provider, r.rb);
    const to = nodeFor(r.importer, r.ra);
    const via = `import L${r.line}: ${cleanText(r.target, 120)}${crossBare ? ' (bare-name guess across repos)' : ''}`;
    add(from, to, 'change', via);
    add(from, to, 'runtime', via);
  }
  if (includeClones) {
    const clones = db.prepare(`
      SELECT sa.file_id AS fa, sb.file_id AS fb, fb2.repo_id AS rb, COUNT(DISTINCT sa.body_hash) AS n
      FROM symbols sa JOIN files fa2 ON fa2.id = sa.file_id
      JOIN symbols sb ON sb.body_hash = sa.body_hash AND sb.id != sa.id JOIN files fb2 ON fb2.id = sb.file_id
      WHERE fa2.repo_id = ? AND sa.body_hash IS NOT NULL AND sa.body_hash != '' GROUP BY sa.file_id, sb.file_id`).all(repoId);
    for (const r of clones) {
      if (!fileNode.has(r.fa)) continue;
      const other = nodeFor(r.fb, r.rb);
      add(fileNode.get(r.fa), other, 'change', `clone: ${Number(r.n)} identical bod${Number(r.n) === 1 ? 'y' : 'ies'}`);
      add(other, fileNode.get(r.fa), 'change', `clone: ${Number(r.n)} identical bod${Number(r.n) === 1 ? 'y' : 'ies'}`);
    }
  }
  for (const id of external) components.push({ ...repos.find((c) => c.id === id), layer: 'repo' });
  return { components, edges, evidence };
}

export function atlasModel(path, { granularity = 'repo', repo, pathPrefix, includeClones = true, excludeBareNames = false, maxFiles = 400 } = {}) {
  const db = openReadOnly(path);
  try {
    checkAtlasSchema(db);
    const g = granularity === 'file'
      ? fileGraph(db, { repo, pathPrefix, includeClones, excludeBareNames, maxFiles })
      : repoGraph(db, { includeClones, excludeBareNames });
    const parseErrors = Number(db.prepare("SELECT COUNT(*) AS n FROM files WHERE parse_error IS NOT NULL AND parse_error != ''").get().n);
    const model = {
      modelVersion: 1,
      about: `Coupling model read from a kgirl Atlas database (${granularity} granularity). Edge A -> B: B imports A, or A and B share cloned code.`,
      components: g.components, edges: g.edges, controls: {}, capabilities: [], acceptedCycles: [], failureModes: [],
    };
    const warnings = parseErrors ? [`${parseErrors} indexed file(s) failed to parse; their imports are missing, so their dependents are missing too`] : [];
    const guesses = g.evidence.filter((e) => e.kind === 'import' && e.bareName === e.count);
    for (const e of guesses) warnings.push(`${e.from} -> ${e.to} rests only on bare-name resolution (${e.via}); verify it before trusting it, or pass exclude_bare_names`);
    return { model, evidence: g.evidence, warnings };
  } finally {
    db.close();
  }
}
